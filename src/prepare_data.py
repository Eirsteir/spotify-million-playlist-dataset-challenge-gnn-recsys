import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm

from contants import RAW_DATA_DIR, DATA_DIR_TRAIN, DATA_DIR_TEST, DATA_PARQUET, V_NODES_PARQUET, U_NODES_PARQUET, RAW_DATA_DIR_TEST, RAW_DATA_DIR_TRAIN


def fullpaths_generator(path):
    filenames = os.listdir(path)
    fullpaths = []
    for filename in filenames:
        if filename.endswith('.json'):
            fullpath = os.sep.join((path, filename))
            fullpaths.append(fullpath)
    return fullpaths


def get_tracks_from_playlist(playlist):
    tracks = []
    for track in playlist['tracks']:
        track_uri = track['track_uri'].split(':')[-1]
        tracks.append(track_uri)

    return tracks


def process_data(full_paths, save_to):
    playlists, track_uris, u_nodes, v_nodes = [], [], [], []

    for fullpath in tqdm(full_paths):
        track_uris, u_nodes, v_nodes = read_slice(fullpath, playlists, track_uris, u_nodes, v_nodes)

    store_playlists(playlists, save_to)
    del playlists

    uri2id = dict(zip(track_uris, range(len(track_uris))))  # TODO: valid ids?
    store_tracks(track_uris, uri2id, save_to)
    del track_uris

    store_edges(u_nodes, uri2id, v_nodes, save_to)


def store_edges(u_nodes, uri2id, v_nodes, save_to):
    v_nodes = np.vectorize(uri2id.get)(v_nodes)
    assert len(u_nodes) == len(v_nodes)
    mapping_df = pd.DataFrame.from_dict({
        "u_nodes": u_nodes,
        "v_nodes": v_nodes
    })
    mapping_df.to_parquet(save_to + DATA_PARQUET)
    del mapping_df


def store_tracks(track_uris, uri2id, save_to):
    track_ids = np.vectorize(uri2id.get)(track_uris)
    tracks_df = pd.DataFrame.from_dict({
        "track_id": track_ids,
        "track_uri": track_uris
    })
    tracks_df.to_parquet(save_to + V_NODES_PARQUET)
    del tracks_df


def store_playlists(playlists, save_to):
    playlists_df = pd.DataFrame.from_dict({
        "pid": playlists
    })
    playlists_df.to_parquet(save_to + U_NODES_PARQUET)
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


def download_dataset(name, target_path):
    # azureml-core of version 1.0.72 or higher is required
    from azureml.core import Workspace, Dataset
    from decouple import config

    # TODO: keys
    subscription_id = config('subscription_id', default='')
    resource_group = 'spotify-mpd-challenge-v2'
    workspace_name = 'spotify-mpd-challenge-v2'

    workspace = Workspace(subscription_id, resource_group, workspace_name)
    dataset = Dataset.get_by_name(workspace, name=name)
    dataset.download(target_path=target_path, overwrite=True)


def prepare_data(
        platform="local", 
        evaluate=False,
        replace=False  # TODO: True
    ):
    print("|Preparing data...")

    mpd_train_dir = RAW_DATA_DIR_TRAIN
    mpd_test_dir = RAW_DATA_DIR_TEST

    if not replace and has_already_prepared_data(DATA_DIR_TRAIN, DATA_DIR_TEST):
        return

    if platform == "azure":
        print("|Downloading datasets...")
        download_dataset('spotify-mpd', target_path=mpd_train_dir)
        download_dataset('spotify-mpd-test', target_path=mpd_test_dir)
    
    if not evaluate:
        print("|Loading training dataset...")
        train_full_paths = fullpaths_generator(mpd_train_dir + "slices/")
        process_data(train_full_paths, save_to=DATA_DIR_TRAIN)
    
    print("|Loading test dataset...")
    test_full_paths = fullpaths_generator(mpd_test_dir)
    process_data(test_full_paths, save_to=DATA_DIR_TEST)


def has_already_prepared_data(train_dir, test_dir):
    return len(os.listdir(train_dir)) and len(os.listdir(test_dir))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args")
    
    args.add_argument("--platform", type=str, choices=["azure", "local"], default="local", help="Platform the mpd files are stored")
    args.add_argument('--mpd_train_dir', type=str, default=RAW_DATA_DIR_TRAIN, help="train mpd path")
    args.add_argument('--mpd_test_dir', type=str, default=RAW_DATA_DIR_TEST, help="test mpd path")
    args.add_argument("--evaluate", action="store_true", help="Only prepare test data.")

    args = args.parse_args()

    print(args)
    prepare_data(**args)

