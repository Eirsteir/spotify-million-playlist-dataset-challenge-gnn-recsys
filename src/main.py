import glob
import argparse
import os.path as osp

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from trainer import LightningIGMC
from datamodule import IGMCDatamodule, SpotifyDataModule
from dataset import IGMCDataset
from models import IGMC
from preprocess import train_val_split, load_processed_data
from contants import RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST



def run_pipeline(args):
    seed_everything(42)
    print("|Loading data...")
    
    datamodule = SpotifyDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_features=args.use_features,
        num_hops=args.num_hops, 
        max_nodes_per_hop=args.max_nodes_per_hop,
        platform=args.platform,
        use_test_data=args.evaluate,
        debug=args.debug
    )
    datamodule.prepare_data()
    datamodule.setup()

    model = LightningIGMC(
        lr=args.lr,
        lr_decay_step_size=args.lr_decay_step_size,
        lr_decay_factor=args.lr_decay_factor,
        ARR=args.ARR,
        # Model arguments
        dataset=datamodule.train_dataset,
        latent_dim=[32, 32, 32, 32],
        num_layers=args.num_layers,
        num_bases=4,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        use_features=args.use_features,
        n_side_features=datamodule.n_features,
    )

    print(f'|#Params {sum([p.numel() for p in model.parameters()])}')

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

    # General settings
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument('--evaluate', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--datasplit_from_file", action="store_true", 
                        help="Whether to use data split previosly stored")
    parser.add_argument('--max-train-num', type=int, default=None, 
                    help='set maximum number of train data to use')
    parser.add_argument('--max-val-num', type=int, default=None, 
                        help='set maximum number of val data to use')
    parser.add_argument('--max-test-num', type=int, default=None, 
                        help='set maximum number of test data to use')
    parser.add_argument("--platform", type=str, choices=["azure", "local"], default="local", help="Platform the mpd files are stored")
    parser.add_argument('--mpd_train_dir', type=str, default=RAW_DATA_DIR_TRAIN, help="train mpd path")
    parser.add_argument('--mpd_test_dir', type=str, default=RAW_DATA_DIR_TEST, help="test mpd path")

    # Model arguments
    parser.add_argument('--num_layers', type=int, default=4)

    # Subgraph extraction arguments
    parser.add_argument("--num_hops", type=int, default=1, help='enclosing subgraph hop number')
    parser.add_argument('--use_features', action='store_true',
                        help="whether to use raw node features as additional GNN input")
    parser.add_argument('--max-nodes-per-hop', default=10000, 
                    help='if > 0, upper bound the # nodes per hop by another subsampling')

    # Edge dropout settings
    parser.add_argument('--adj-dropout', type=float, default=0.2,
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')

    # Optimization settings
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

    # Other
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--model', type=str, default='igmc', choices=['igmc'])

    args = parser.parse_args()
    print(args)

    run_pipeline(args)
