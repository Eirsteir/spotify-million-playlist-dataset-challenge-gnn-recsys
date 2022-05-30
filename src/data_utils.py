import random

import numpy as np
import pandas as pd
from contants import DATA_PARQUET, U_NODES_PARQUET, V_NODES_PARQUET

DATA_DIR = "./data/raw/"
files = [DATA_DIR + DATA_PARQUET, DATA_DIR + U_NODES_PARQUET, DATA_DIR + V_NODES_PARQUET]


def load_data(seed=1234, verbose=True):
    """ Loads dataset and creates adjacency matrix and feature matrix

    Parameters
    ----------
    fname : str, dataset
    seed: int, dataset shuffling seed
    verbose: to print out statements or not
    Returns
    -------
    num_users : int
        Number of playlists and items respectively
    num_items : int
    u_nodes : np.int32 arrays
        Playlist indices
    v_nodes : np.int32 array
        item (track) indices
    ratings : np.float32 array
        Playlist/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
        item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
        not necessarily all u_nodes[k] or all v_nodes[k] separately.
    u_features: np.float32 array, or None
        If present in dataset, contains the features of the users.
    v_features: np.float32 array, or None
        If present in dataset, contains the features of the users.
    seed: int,
        For datashuffling seed with pythons own random.shuffle, as in CF-NADE.
    """
    
    u_features = None
    v_features = None

    data = pd.read_parquet(files[0])

    # shuffle here like cf-nade paper with python's own random class
    # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
    data_array = data.values.tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)

    u_nodes_data = data_array[:, 0].astype(np.int32)
    v_nodes_data = data_array[:, 1].astype(np.int32)
    r_edges = np.ones(len(data)).astype(np.float64)

    playlist_df = pd.read_parquet(files[1])

    track_df = pd.read_parquet(files[2])

    num_playlists = len(playlist_df)
    num_tracks = len(track_df)

    if verbose:
        print('Number of playlists = %d' % num_playlists)
        print('Number of tracks = %d' % num_tracks)
        print('Number of links = %d' % r_edges.shape[0])
        print('Fraction of positive links = %.4f' % (float(r_edges.shape[0]) / (num_playlists * num_tracks),))

    return num_playlists, num_tracks, u_nodes_data, v_nodes_data, r_edges, u_features, v_features







