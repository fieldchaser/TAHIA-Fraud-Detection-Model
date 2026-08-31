import time
import os
import random
import argparse
import atexit
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils import *
from model import *
from layers import *
from graphsage import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Tee:
	def __init__(self, *streams):
		self.streams = streams

	def write(self, data):
		for stream in self.streams:
			stream.write(data)
			stream.flush()

	def flush(self):
		for stream in self.streams:
			stream.flush()


def setup_logging(log_file):
	log_path = Path(log_file)
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_handle = log_path.open('w', buffering=1)
	old_stdout = sys.stdout
	old_stderr = sys.stderr
	sys.stdout = Tee(sys.__stdout__, log_handle)
	sys.stderr = Tee(sys.__stderr__, log_handle)
	def cleanup():
		sys.stdout = old_stdout
		sys.stderr = old_stderr
		log_handle.close()
	atexit.register(cleanup)
	print(f'Logging to: {log_path}')
	print(f'Start time: {datetime.now().isoformat(timespec="seconds")}')

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='dblp', help='The dataset name. [yelp, amazon, elliptic]')
parser.add_argument('--model', type=str, default='CARE', help='The model name. [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN', help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size 1024 for yelp, 256 for amazon.')

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument('--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).')
parser.add_argument('--emb-size', type=int, default=64, help='Node embedding size at the last layer.')
parser.add_argument('--num-epochs', type=int, default=91, help='Number of epochs.')
parser.add_argument('--test-epochs', type=int, default=3, help='Epoch interval to run test set.')
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')

# other args
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--log-file', type=str, default=None, help='Path to write the training log.')


args = parser.parse_args()
default_log_file = Path(__file__).resolve().parent / 'results' / f'{args.data}.log'
setup_logging(args.log_file or default_log_file)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(f'run on {args.data}')

# load graph, feature, and label
[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)

# train_test split
np.random.seed(args.seed)
random.seed(args.seed)
split_path = Path(__file__).resolve().parent / 'data' / args.data / 'splits.npz'
if split_path.exists():
	splits = np.load(split_path)
	idx_train = splits['train_idx'].astype(np.int64).tolist()
	idx_test = splits['test_idx'].astype(np.int64).tolist()
	y_train = labels[idx_train]
	y_test = labels[idx_test]
elif args.data == 'yelpchi':
	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.60,
															random_state=2, shuffle=True)
elif args.data == 'amazon':  # amazon
	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
															test_size=0.60, random_state=2, shuffle=True)
elif args.data == 'elliptic':
	index = list(range(len(labels)))
	idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=0.1,
															random_state=2, shuffle=True)
	idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
															test_size=0.67,
															random_state=2, shuffle=True)
	idx_test = idx_rest
	y_test = y_rest
elif args.data == 'tfinance':
	index = list(range(len(labels)))
	idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=0.1,
															random_state=2, shuffle=True)
	idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
															test_size=0.67,
															random_state=2, shuffle=True)
	idx_test = idx_rest
	y_test = y_rest
elif args.data == 'acm' or args.data == 'dblp' or args.data == 'yelp' or args.data == 'amazon' or args.data == 'yelpchi' or args.data == 'zhuanzhang' or args.data == 'xinyongka' or args.data == 'credit':
	index = list(range(len(labels)))
	idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=0.1,
															random_state=2, shuffle=True)
	idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
															test_size=0.67,
															random_state=2, shuffle=True)
	idx_test = idx_rest
	y_test = y_rest

idx_val = idx_train
y_val = y_train
print(f'Split: train={len(idx_train)}, test={len(idx_test)}, val=train-copy')

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train)

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
	features.cuda()

# set input graph
if args.model == 'SAGE':
	adj_lists = homo
else:
	adj_lists = [homo, relation2, relation3]

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# build one-layer models
if args.model == 'CARE':
	intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
	inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, adj_lists, [intra1, intra2, intra3], inter=args.inter,
					  step_size=args.step_size, cuda=args.cuda)
elif args.model == 'SAGE':
	agg1 = MeanAggregator(features, cuda=args.cuda)
	enc1 = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg1, gcn=True, cuda=args.cuda)

if args.model == 'CARE':
	gnn_model = OneLayerCARE(2, inter1, args.lambda_1)
elif args.model == 'SAGE':
	# the vanilla GraphSAGE model as baseline
	enc1.num_samples = 5
	gnn_model = GraphSage(2, enc1)

if args.cuda:
	gnn_model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
times = []
performance_log = []
best_score = -1
best_metrics = None
best_epoch = 0
final_metrics = None
final_epoch = 0

# train the model
for epoch in range(args.num_epochs):
	# randomly under-sampling negative nodes for each epoch
	sampled_idx_train = undersample(train_pos, train_neg, scale=1)
	rd.shuffle(sampled_idx_train)

	# send number of batches to model to let the RLModule know the training progress
	num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
	if args.model == 'CARE':
		inter1.batch_num = num_batches

	loss = 0.0
	epoch_time = 0

	# mini-batch training
	for batch in range(num_batches):
		start_time = time.time()
		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
		batch_nodes = sampled_idx_train[i_start:i_end]
		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()
		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		epoch_time += end_time - start_time
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

	# testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:
		gnn_model.eval()
		with torch.no_grad():
			if args.model == 'SAGE':
				val_metrics = evaluate_sage(idx_val, y_val, gnn_model, args.batch_size)
				test_metrics = evaluate_sage(idx_test, y_test, gnn_model, args.batch_size)
			else:
				val_metrics = evaluate_care(idx_val, y_val, gnn_model, args.batch_size)
				test_metrics = evaluate_care(idx_test, y_test, gnn_model, args.batch_size)
		gnn_model.train()
		final_metrics = test_metrics
		final_epoch = epoch
		score = val_metrics['macro_f1']
		print(
			f"Epoch {epoch}: val_f1={val_metrics['macro_f1']:.4f} "
			f"test_auc={test_metrics['auroc']:.4f} test_precision={test_metrics['precision']:.4f} "
			f"test_recall={test_metrics['recall']:.4f} test_f1={test_metrics['macro_f1']:.4f}"
		)
		if score >= best_score:
			best_score = score
			best_metrics = test_metrics
			best_epoch = epoch

print('[test best by train-copy val_f1]')
print(f'best_epoch: {best_epoch}')
for key in ['precision', 'recall', 'macro_f1', 'auroc']:
	print(f'{key}: {best_metrics[key]:.4f}')
print('[test final]')
print(f'final_epoch: {final_epoch}')
for key in ['precision', 'recall', 'macro_f1', 'auroc']:
	print(f'{key}: {final_metrics[key]:.4f}')
