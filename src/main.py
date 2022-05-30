import glob
import os
import argparse
import os.path as osp

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from trainer import LightningIGMC
from datamodule import IGMCDatamodule
from dataset import IGMCDataset
from models import IGMC
from preprocess import train_val_split_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_pipeline(args):
    seed_everything(42)
    # load data
    print("|Loading data...")
    if args.use_features:
        datasplit_path = (
            'raw_data/mpd/withfeatures.pickle'
        )
    else:
        datasplit_path = (
            'raw_data/mpd/split.pickle'
        )

    (
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,
        val_labels, val_u_indices, val_v_indices, class_values
    ) = train_val_split_data(datasplit_path=datasplit_path, datasplit_from_file=None)

    if args.use_features:
        u_features, v_features = u_features.toarray(), v_features.toarray()
        n_features = u_features.shape[1] + v_features.shape[1]
        print(f'Number of playlist features {u_features.shape[1]}, '
              f'track features {v_features.shape[1]}, '
              f'total features {n_features}')
    else:
        u_features, v_features = None, None
        n_features = 0

    if args.debug:  # use a small number of data to debug
        num_data = 1000
        train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
        val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]

    train_indices = (train_u_indices, train_v_indices)
    val_indices = (val_u_indices, val_v_indices)
    print(f"#train: {len(train_u_indices)}, #val: {len(val_u_indices)}")

    # Dynamically extract enclosing subgraphs
    train_graphs, val_graphs, test_graphs = None, None, None
    root = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data')

    train_graphs = IGMCDataset(
        root,
        adj_train,
        train_indices,
        train_labels,
        args.hop,
        args.sample_ratio,
        args.max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=args.max_train_num
    )

    if not args.testing:
        val_graphs = IGMCDataset(
            root,
            adj_train,
            val_indices,
            val_labels,
            args.hop,
            args.sample_ratio,
            args.max_nodes_per_hop,
            u_features,
            v_features,
            class_values,
            max_num=args.max_val_num
        )

    # Determine testing data (on which data to evaluate the trained model
    if not args.testing:
        test_graphs = val_graphs

    print(f'Using #train graphs: {len(train_graphs)}, #test graphs: {len(test_graphs)}')

    num_relations = len(class_values)


    datamodule = IGMCDatamodule(
        train_graphs,
        val_graphs,
        test_graphs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_features=args.use_features
    )

    model = IGMC(
        train_graphs,
        latent_dim=[32, 32, 32, 32],
        num_bases=4,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        use_features=args.use_features,
        n_side_features=n_features
    )
    total_params = sum(p.numel() for param in model.parameters() for p in param)
    print(f'Total number of parameters: {total_params}')

    model = LightningIGMC(
        lr=args.lr,
        lr_decay_step_size=args.lr_decay_step_size,
        lr_decay_factor=args.lr_decay_factor,
        ARR=args.ARR,
        # Model arguments
        dataset=train_graphs,
        latent_dim=32,
        num_layers=args.num_layers,
        num_bases=4,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        use_features=args.use_features,
        n_side_features=n_features,
    )

    print(f'#Params {sum([p.numel() for p in model.parameters()])}')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        mode='min',
        save_top_k=1
    )
    trainer = Trainer(
        accelerator="auto",
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=f'logs/{args.model}'
    )

    if not args.evaluate:
        trainer.fit(model, datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)
        model = LightningIGMC.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml')

        trainer.test(model=model, datamodule=datamodule)

        loader = datamodule.test_dataloader()

        model.eval()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        y_preds = []
        for batch in tqdm(loader):
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                y_preds.append(out)
        res = {'y_pred': torch.cat(y_preds, dim=0)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IGMC')

    # Experiment arguments
    parser.add_argument("--name", type=str, default="test")

    # Model arguments
    parser.add_argument('--adj-dropout', type=float, default=0.2,
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    parser.add_argument('--use_features', action='store_true',
                        help="whether to use raw node features as additional GNN input")
    parser.add_argument('--num_layers', type=int, default=4)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay-step-size', type=int, default=50,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--ARR', type=float, default=0.001,
                        help="Adjacent rating regularization lambda value")
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='igmc', choices=['igmc'])

    args = parser.parse_args("")
    print(args)

    run_pipeline(args)
