import os.path as osp
from typing import Optional 

from pytorch_lightning import (LightningDataModule)
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from prepare_data import prepare_data as prepare_spotify_data
from preprocess import train_val_split, load_processed_data
from dataset import IGMCDataset


class SpotifyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        use_features: bool = False,
        num_hops: int = 1, 
        max_nodes_per_hop: int = 10000,
        platform: str = "local",
        use_test_data: bool = False,
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
        self.use_test_data = use_test_data,
        self.debug = debug

    def prepare_data(self):
        # Download etc
        prepare_spotify_data(
            platform=self.platform, 
            evaluate=self.use_test_data
        )
        

    def setup(self, stage: Optional[str] = None):
        # Split, transform, etc
        if self.use_features:
            datasplit_path = 'data/train/mpd-withfeatures.pickle'
        else:
            datasplit_path = 'data/train/mpd-split.pickle'

        data = load_processed_data(datasplit_path=datasplit_path)
        
        (
            u_features, v_features, adj_train, 
            train_labels, train_u_indices, train_v_indices,
            val_labels, val_u_indices, val_v_indices, 
            class_values
        ) = train_val_split(*data)

        if self.use_features:
            u_features, v_features = u_features.toarray(), v_features.toarray()
            self.n_features = u_features.shape[1] + v_features.shape[1]
            print(f'Number of playlist features {u_features.shape[1]}, '
                f'track features {v_features.shape[1]}, '
                f'total features {n_features}')
        else:
            u_features, v_features = None, None
            self.n_features = 0

        if self.debug:  # use a small number of data to debug
            num_data = 1000
            train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
            val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]

        train_indices = (train_u_indices, train_v_indices)
        val_indices = (val_u_indices, val_v_indices)
        print(f"#train: {len(train_u_indices)}, #val: {len(val_u_indices)}")

        # Dynamically extract enclosing subgraphs
        root = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data')

        self.train_dataset = IGMCDataset(
            root,
            adj_train,
            train_indices,
            train_labels,
            self.num_hops,
            self.max_nodes_per_hop,
            u_features,
            v_features,
            class_values
        )

        self.val_dataset = IGMCDataset(
            root,
            adj_train,
            val_indices,
            val_labels,
            self.num_hops,
            self.max_nodes_per_hop,
            u_features,
            v_features,
            class_values
        )
        
        # TODO: load challenge data
        if self.use_test_data:
            self.test_dataset = self.val_dataset

        print(f'Using #train graphs: {len(self.train_dataset)}, #test graphs: {len(self.test_dataset)}')
        

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