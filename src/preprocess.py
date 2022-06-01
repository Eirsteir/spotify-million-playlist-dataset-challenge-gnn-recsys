from __future__ import division
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import scipy.sparse as sp

from data_utils import load_data
from contants import DATA_DIR_TRAIN

def load_challenge_data(datasplit_path=DATA_DIR_TEST):
    num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_processed_data(datasplit_path)

    u_idx, v_idx = np.vstack([u_nodes, v_nodes])

    labels = np.array(ratings, dtype=np.int32)    
    class_values = np.sort(np.unique(ratings))
    
    data = labels.astype(np.float32)

    R_train = sp.csr_matrix((data, [u_idx, v_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)

    return u_features, v_features, R_train, labels, u_idx, v_idx, class_values


def load_processed_data(
        path=DATA_DIR_TRAIN,
        datasplit_path=None,
        seed=1234
):
    if os.path.isfile(datasplit_path):
        print('Reading processed dataset from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)
    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = load_data(path=path, seed=seed)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    print('Number of users = %d' % num_users)
    print('Number of items = %d' % num_items)
    print('Number of links = %d' % ratings.shape[0])
    print('Fraction of positive links = %.10f' % (float(ratings.shape[0]) / (num_users * num_items),))
        
    return num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features


def train_val_split(
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features
):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """
 
    print("Using random dataset split ...")
    num_val = int(np.ceil(ratings.shape[0] * 0.1))
    num_train = ratings.shape[0] - num_val

    pairs_nonzero = np.vstack([u_nodes, v_nodes]).transpose()

    train_pairs_idx = pairs_nonzero[:num_train]
    val_pairs_idx = pairs_nonzero[num_train:]

    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    all_labels = np.array(ratings, dtype=np.int32)    
    train_labels = all_labels[:num_train]
    val_labels = all_labels[num_train:]
    class_values = np.sort(np.unique(ratings))
    
    data = train_labels.astype(np.float32)

    R_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)

    return u_features, v_features, R_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, class_values

