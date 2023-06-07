import torch
import warnings
import argparse
from utils import build_env, process_data, gen_csv_prediction
from train import train

warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train GNN for link prediction on academic network.')
    parser.add_argument('--num_epochs', default=50, type=int, help='maximum training epochs')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=4e-4, type=float, help='weight decay')
    parser.add_argument('--lr_period', default=10, type=int, help='period for lr_scheduler')
    parser.add_argument('--lr_decay', default=0.78, type=float, help='gamma decay factor for lr')
    parser.add_argument('--in_dim', default=512, type=int, help='dimensionality of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='dimensionality of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='dimensionality of output embedding')
    parser.add_argument('--cuda', default=0, type=int, help='gpu index')
    parser.add_argument('--k', default=4, type=int, help='times for negative sampling')
    parser.add_argument('--seed', default=1063, type=int, help='global random seed for reproduction')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size for graph sampler')
    parser.add_argument('--num_layers', default=4, type=int, help='Number of Conv layers in the GNN')
    args = parser.parse_args()
    return args


def main(args):
    device = f'cuda:{args.cuda}' if args.cuda >= 0 and torch.cuda.is_available() else 'cpu'
    train_refs, test_refs, refs_to_pred, cite_edges, coauthor_edges, paper_feature = process_data()
    hetero_graph, rel_list = build_env(train_refs, cite_edges, coauthor_edges, paper_feature, device)
    best_embed, best_thresh, best_f1 = train(args, hetero_graph, test_refs, rel_list, device)
    gen_csv_prediction(best_embed, refs_to_pred, best_thresh, best_f1, args.seed)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)