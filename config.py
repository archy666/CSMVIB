import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Hyper-parameter settings for deterministic CSMVIB model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 2)')
    parser.add_argument('--beta', type=list, default=[0.01], help='Beta parameter for IB (default: [0.0001])')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate (default: 0.1)')
    parser.add_argument('--step_size', type=float, default=100, help='Step size for scheduler (default: 100)')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--loss_type', type=str, default='CSMVIB', help='Loss type (options: "loss_ce", "CSIB"; default: "CSIB")')
    parser.add_argument('--optimizer_name', type=str, default='SGD', help='Optimizer name (options: "Adam", "SGD"; default: "SGD")')
    parser.add_argument('--nsplits', type=int, default=5, help='Number of splits for StratifiedKFold (default: 5)')
    parser.add_argument('--scheduler', action='store_true', help='Use scheduler (default: False)')
    
    return parser.parse_args()
