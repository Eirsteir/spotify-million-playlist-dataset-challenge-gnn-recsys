import glob
import os
import os.path as osp

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from src.trainer import LightningIGMC
from src.datamodule import IGMCDatamodule, SpotifyDataModule
from src.dataset import IGMCDataset
from src.models import IGMC
from src.preprocess import train_val_split, load_processed_data
from src.contants import RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST



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
        debug=args.debug
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    model_params = dict(
        num_features=datamodule.train_dataset.num_features,
        num_relations=datamodule.train_dataset.num_relations, 
        latent_dim=[32, 32, 32, 32],
        num_layers=args.num_layers,
        num_bases=4,
        adj_dropout=args.adj_dropout,
        force_undirected=args.force_undirected,
        use_features=args.use_features,
        n_side_features=datamodule.n_features
    )
    model = LightningIGMC(
        lr=args.lr,
        lr_decay_step_size=args.lr_decay_step_size,
        lr_decay_factor=args.lr_decay_factor,
        ARR=args.ARR,
        **model_params
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
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]

        print(f'|Evaluating model saved in {logdir}...')

        datamodule.setup(stage="test")

        trainer = Trainer(
            accelerator="auto",
            devices=args.devices, 
            resume_from_checkpoint=ckpt
        )
        # checkpoint_callback.best_model_path
        model = LightningIGMC.load_from_checkpoint(
            ckpt, 
            hparams_file=f'{logdir}/hparams.yaml',
            # lr=args.lr,
            # lr_decay_step_size=args.lr_decay_step_size,
            # lr_decay_factor=args.lr_decay_factor
        )

        trainer.test(model=model, datamodule=datamodule)
        
        loader = datamodule.predict_dataloader() # challenge data

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

