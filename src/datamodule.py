import os.path as osp
from typing import Optional 

from pytorch_lightning import (LightningDataModule)
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from src.prepare_data import prepare_data as prepare_spotify_data
from src.preprocess import train_val_split, load_processed_data, load_challenge_data
from src.dataset import IGMCDataset
from src.contants import DATA_DIR_TRAIN, DATA_DIR_TEST


def create_dataset(
        adj,
        labels,
        u_indices,
        v_indices,
        debug=False,
        **ds_kwargs
    ):
    if debug:  # use a small number of data to debug
        num_data = 1000
        u_indices, v_indices = u_indices[:num_data], v_indices[:num_data]
        
    indices = (u_indices, v_indices)

    # print(f"|Link split - train: {len(train_u_indices)}, val: {len(val_u_indices)}")

    # Dynamically extract enclosing subgraphs
    root = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data')

    return IGMCDataset(
        root,
        adj,
        indices,
        labels,
        **ds_kwargs
    )


class SpotifyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        use_features: bool = False,
        num_hops: int = 1, 
        max_nodes_per_hop: int = 10000,
        platform: str = "local",
        debug: bool = False,
        **kwargs
    ):        
        super().__init__(**kwargs)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_features = use_features
        self.num_hops = num_hops
        self.max_nodes_per_hop = max_nodes_per_hop
        self.n_features = 0
        
        self.platform = platform 
        self.debug = debug

    def prepare_data(self):
        # Download etc
        prepare_spotify_data(
            platform=self.platform, 
            evaluate=True
        )
        

    def setup(self, stage: Optional[str] = None):
        if stage and stage == "fit":
            # Split, transform, etc
            if self.use_features:
                filename = 'mpd-withfeatures.pickle'
            else:
                filename = 'mpd-split.pickle'

            data = load_processed_data(data_dir=DATA_DIR_TRAIN, filename=filename)
            
            (
                u_features, v_features, adj_train, 
                train_labels, train_u_indices, train_v_indices,
                val_labels, val_u_indices, val_v_indices, 
                class_values
            ) = train_val_split(*data)

            if self.use_features:
                u_features, v_features = u_features.toarray(), v_features.toarray()
                self.n_features = u_features.shape[1] + v_features.shape[1]
                print(f'|Train data: Number of playlist features {u_features.shape[1]}, '
                    f'track features {v_features.shape[1]}, '
                    f'total features {n_features}')
            else:
                u_features, v_features = None, None
                self.n_features = 0

            self.train_dataset = create_dataset(
                adj=adj_train,
                labels=train_labels,
                u_indices=train_u_indices,
                v_indices=train_v_indices,
                debug=self.debug,
                num_hops=self.num_hops,
                max_nodes_per_hop=self.max_nodes_per_hop,
                u_features=u_features,
                v_features=v_features,
                class_values=class_values
            )

            self.val_dataset = create_dataset(
                adj=adj_train,
                labels=val_labels,
                u_indices=val_u_indices,
                v_indices=val_v_indices,
                debug=self.debug,
                num_hops=self.num_hops,
                max_nodes_per_hop=self.max_nodes_per_hop,
                u_features=u_features,
                v_features=v_features,
                class_values=class_values
            )
            
            print(f'|Using {len(self.train_dataset)} train graphs and {len(self.val_dataset)} val graphs')
        
        # TODO: challenge dataloader 
        if stage and stage == "test":
            if self.use_features:
                filename = 'mpd-withfeatures.pickle'
            else:
                filename = 'mpd-nofeatures.pickle'

            (
                u_features, v_features, adj_test, 
                test_labels, test_u_indices, test_v_indices,
                class_values
            ) = load_challenge_data(data_dir=DATA_DIR_TEST, filename=filename)
        
            self.test_dataset = create_dataset(
                adj=adj_test,
                labels=test_labels,
                u_indices=test_u_indices,
                v_indices=test_v_indices,
                debug=self.debug,
                num_hops=self.num_hops,
                max_nodes_per_hop=self.max_nodes_per_hop,
                u_features=u_features,
                v_features=v_features,
                class_values=class_values
            )
            
            print(f"Using {len(self.test_dataset)} test graphs")
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)



class IGMCDatamodule(LightningDataModule):
    def __init__(self,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 test_dataset: Dataset,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)