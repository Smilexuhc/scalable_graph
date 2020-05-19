import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to log dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gcn_type', type=str, choices=['sage', 'gat', 'egnn'], default='sage')
    parser.add_argument('--sampler', type=str, choices=['ns'], default='ns')
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--num_timesteps_input', type=int, default=12,
                        help='Num of input timesteps')
    parser.add_argument('--num_timesteps_output', type=int, default=3,
                        help='Num of output timesteps for forecasting')
    parser.add_argument('--early_stop_rounds', type=int, default=30,
                        help='Earlystop rounds when validation loss does not decrease')
    parser.add_argument('--num_neighbors',type=str, default='5,5')
    parser.add_argument('--use_gpu',type=bool,default=True)
    args = parser.parse_args()
    args.num_neighbors = [int(num) for num in args.num_neighbors.split(',')]

    if args.use_gpu:
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')
    else:
        args.device = torch.device('cpu')

    print(args)
    return args

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
#     parser.add_argument('--log-name', type=str, default='default',
#                         help='Experiment name to log')
#     parser.add_argument('--log-dir', type=str, default='./logs',
#                         help='Path to log dir')
#     parser.add_argument('--gpus', type=int, default=1,
#                         help='Number of GPUs to use')
#     parser.add_argument('-m', "--model", choices=['tgcn', 'stgcn', 'gwnet'],
#                         help='Choose Spatial-Temporal model', default='stgcn')
#     parser.add_argument('-d', "--dataset", choices=["metr", "nyc-bike"],
#                         help='Choose dataset', default='metr')
#     parser.add_argument('-t', "--gcn-type", choices=['sage', 'gated', 'gat'],
#                         help='Choose GCN Conv Type', default='graph')
#     parser.add_argument('-part', "--gcn-partition", choices=['cluster', 'sample'],
#                         help='Choose GCN partition method',
#                         default=None)
#     parser.add_argument('-batchsize', type=int, default=32,
#                         help='Training batch size')
#     parser.add_argument('-epochs', type=int, default=1000,
#                         help='Training epochs')
#     parser.add_argument('-l', '--loss-criterion', choices=['mse', 'mae'],
#                         help='Choose loss criterion', default='mse')
#     parser.add_argument('-num-timesteps-input', type=int, default=12,
#                         help='Num of input timesteps')
#     parser.add_argument('-num-timesteps-output', type=int, default=3,
#                         help='Num of output timesteps for forecasting')
#     parser.add_argument('-early-stop-rounds', type=int, default=30,
#                         help='Earlystop rounds when validation loss does not decrease')
#
#     args = parser.parse_args()
#     return args