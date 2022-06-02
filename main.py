import argparse

from src.run_pipeline import run_pipeline
from src.contants import RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IGMC')

    # General settings
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument('--evaluate', action='store_true', default=False,
                    help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
    parser.add_argument("--debug", action="store_true")
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