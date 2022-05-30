import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from contants import RAW_DATA_DIR, DATA_PARQUET, V_NODES_PARQUET, U_NODES_PARQUET


def fullpaths_generator(path):
    filenames = os.listdir(path)
    fullpaths = []
    for filename in filenames:
        fullpath = os.sep.join((path, filename))
        fullpaths.append(fullpath)
    return fullpaths


def get_tracks_from_playlist(playlist):
    tracks = []
    for track in playlist['tracks']:
        track_uri = track['track_uri'].split(':')[-1]
        tracks.append(track_uri)

    return tracks


def process_data(full_paths):
    playlists, track_uris, u_nodes, v_nodes = [], [], [], []

    for fullpath in tqdm(full_paths):
        track_uris, u_nodes, v_nodes = read_slice(fullpath, playlists, track_uris, u_nodes, v_nodes)

    store_playlists(playlists)
    del playlists

    uri2id = dict(zip(track_uris, range(len(track_uris))))  # TODO: valid?
    store_tracks(track_uris, uri2id)
    del track_uris

    store_edges(u_nodes, uri2id, v_nodes)


def store_edges(u_nodes, uri2id, v_nodes):
    v_nodes = np.vectorize(uri2id.get)(v_nodes)
    assert len(u_nodes) == len(v_nodes)
    mapping_df = pd.DataFrame.from_dict({
        "u_nodes": u_nodes,
        "v_nodes": v_nodes
    })
    mapping_df.to_parquet(RAW_DATA_DIR + DATA_PARQUET)
    del mapping_df


def store_tracks(track_uris, uri2id):
    track_ids = np.vectorize(uri2id.get)(track_uris)
    tracks_df = pd.DataFrame.from_dict({
        "track_id": track_ids,
        "track_uri": track_uris
    })
    tracks_df.to_parquet(RAW_DATA_DIR + V_NODES_PARQUET)
    del tracks_df


def store_playlists(playlists):
    playlists_df = pd.DataFrame.from_dict({
        "pid": playlists
    })
    playlists_df.to_parquet(RAW_DATA_DIR + U_NODES_PARQUET)
    del playlists_df


def read_slice(path, playlists, track_uris, u_nodes, v_nodes):
    f = open(path)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)
    for playlist in mpd_slice['playlists']:
        pid = playlist["pid"]
        playlists.append(pid)

        playlist_tracks = get_tracks_from_playlist(playlist)
        track_uris += playlist_tracks

        num_tracks = len(playlist_tracks)
        u_nodes += [pid] * num_tracks
        v_nodes += playlist_tracks
    return track_uris, u_nodes, v_nodes


if __name__ == '__main__':
    print("|Preprocessing data...")
    args = argparse.ArgumentParser(description="args")
    args.add_argument('--data_dir', type=str, default='./data/raw/slices', help="directory where the outputs are stored")
    args.add_argument('--mpd_tr', type=str, default='./mpd_train', help="train mpd path")
    args.add_argument('--mpd_te', type=str, default='./mpd_test', help="test mpd path")

    args = args.parse_args()

    train_fullpaths = fullpaths_generator(args.data_dir)

    # num_playlists = 1000000
    # num_tracks = 66346428
    # num_unique_tracks = 2262292

    process_data(train_fullpaths)

    """
    Vi ha en P x S matrise A. 
    Element på rad i er en spilleliste og element på kol j er en sang.
    A_i_j = 1 hvis sang j er i spilleliste i. 
    
    1 - lagre alle spilleliste-id'er i .parquet med pid
    2 - lagre alle unike tracks i .parquet: med pid og trackid
    
    3 - laste inn filene 
    4 - lag A
    
    - har ingen features - kan legge til støtte senere men da blir filene gigantiske igjen (gjør det enkelt først)
    - trenger train/val split
    """