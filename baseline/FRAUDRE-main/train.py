import time
import os
import random
import argparse
import atexit
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score
from collections import defaultdict
import math
from model import MODEL
from layers import *
from utlis import *
import torch
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"]="0"


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
   Paper: FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance
   Source: https://github.com/FraudDetection/FRAUDRE
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument('--data', type=str, default='acm', help='The dataset name. [Amazon_demo, Yelp_demo, amazon,yelp]')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size 100 for credit dblp, 32 for acm.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate. [0.1 for credit dblp and 0.001 for acm]')
parser.add_argument('--lambda_1', type=float, default=0, help='lambad_1 = 0, ignore it')
parser.add_argument('--lambda_2', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
parser.add_argument('--embed_dim', type=int, default=512, help='Node embedding size at the first layer.')
parser.add_argument('--num_epochs', type=int, default=201, help='Number of epochs.')
parser.add_argument('--test_epochs', type=int, default=10, help='Epoch interval to run test set.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--log-file', type=str, default=None, help='Path to write the training log.')
parser.add_argument('--max-neighbors', type=int, default=10,
					help='Maximum number of sampled neighbors per node. Set <=0 to use full adjacency.')

args = parser.parse_args()
default_log_file = Path(__file__).resolve().parent / 'results' / f'{args.data}.log'
setup_logging(args.log_file or default_log_file)
args.cuda = not args.no_cuda and torch.cuda.is_available()

if(torch.cuda.is_available()):
	print("cuda is available")

if(args.cuda):
	print("runing with GPU")

print(f'run on {args.data}')

# load topology, feature, and label
homo, relation1, relation2, relation3, feat_data, labels = load_data(
	args.data,
	max_neighbors=args.max_neighbors
)

# set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)


# train_test split
split_path = Path(__file__).resolve().parent / 'data' / args.data / 'splits.npz'
if split_path.exists():
	splits = np.load(split_path)
	idx_train = splits['train_idx'].astype(np.int64).tolist()
	idx_test = splits['test_idx'].astype(np.int64).tolist()
	y_train = labels[idx_train]
	y_test = labels[idx_test]
	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])
elif args.data == 'yelpchi':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.90,
															random_state = 2, shuffle = True)

	# set prior
	num_1= len(np.where(y_train==1)[0])
	num_2= len(np.where(y_train==0)[0])
	p0 = (num_1/(num_1+num_2))
	p1 = 1- p0
	prior = np.array([p1, p0])

	# prior = np.array([0.85, 0.15])

elif args.data == 'amazon':

	# 0-3304 are unlabeled nodes
	index = list(range(3305, len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify = labels[3305:],
															test_size = 0.90, random_state = 2, shuffle = True)
	prior = np.array([0.9, 0.1])

elif args.data == 'Amazon_demo':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
															test_size = 0.60, random_state = 2, shuffle = True)
	prior = np.array([0.9, 0.1])

elif args.data == 'elliptic':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
															test_size=0.9, random_state=2, shuffle=True)
	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])
elif args.data == 'tfinance':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
															test_size=0.9, random_state=2, shuffle=True)
	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])

elif args.data == 'Yelp_demo':

	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify = labels, test_size = 0.80,
															random_state = 2, shuffle = True)

	num_1 = len(np.where(y_train==1)[0])
	num_2 = len(np.where(y_train==0)[0])
	p0 = (num_1/(num_1+num_2))
	p1 = 1- p0
	prior = np.array([p1, p0])
elif args.data == 'acm' or args.data == 'dblp' or args.data == 'yelp' or args.data == 'amazon' or args.data == 'yelpchi' or args.data == 'zhuanzhang' or args.data == 'xinyongka' or args.data == 'credit':
	index = list(range(len(labels)))
	idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=0.90,
															random_state=2, shuffle=True)

	num_1 = len(np.where(y_train == 1)[0])
	num_2 = len(np.where(y_train == 0)[0])
	p0 = (num_1 / (num_1 + num_2))
	p1 = 1 - p0
	prior = np.array([p1, p0])

idx_val = idx_train
y_val = y_train
print(f'Split: train={len(idx_train)}, test={len(idx_test)}, val=train-copy')

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data) 
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad = False)
if args.cuda:
	features.cuda()

# set input graph topology
# adj_lists = [relation1, relation2, relation3]
adj_lists = [homo]



# build model

# the first neural network layer (ego-feature embedding module)
mlp = MLP_(features, feat_data.shape[1], args.embed_dim, cuda = args.cuda)

#first convolution layer
intra1_1 = IntraAgg(cuda = args.cuda)
# intra1_2 = IntraAgg(cuda = args.cuda)
# intra1_3 = IntraAgg(cuda = args.cuda)
agg1 = InterAgg(lambda nodes: mlp(nodes), args.embed_dim, adj_lists, [intra1_1], cuda = args.cuda)


#second convolution layer
intra2_1 = IntraAgg(cuda = args.cuda)
# intra2_2 = IntraAgg(cuda = args.cuda)
# intra2_3 = IntraAgg(cuda = args.cuda)

#def __init__(self, features, embed_dim, adj_lists, intraggs, cuda = False):
agg2 = InterAgg(lambda nodes: agg1(nodes), args.embed_dim*2, adj_lists, [intra2_1], cuda = args.cuda)
gnn_model = MODEL(2, 2, args.embed_dim, agg2, args.lambda_1, prior, cuda = args.cuda)

# gnn_model in one convolution layer
#gnn_model = MODEL(1, 2, args.embed_dim, agg1, args.lambda_1, prior, cuda = args.cuda)


if args.cuda:
	gnn_model.to("cuda:0")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.lambda_2)
performance_log = []
best_score = -1
best_epoch = 0
best_metrics = None
final_epoch = 0
final_metrics = None

	# train the model

overall_time = 0
for epoch in range(args.num_epochs):

	# gnn_model.train()
	# shuffle
	random.shuffle(idx_train)
	num_batches = int(len(idx_train) / args.batch_size) +1

	loss = 0.0
	epoch_time = 0

	#mini-batch training
	for batch in range(num_batches):

		i_start = batch * args.batch_size
		i_end = min((batch + 1) * args.batch_size, len(idx_train))

		batch_nodes = idx_train[i_start:i_end]

		batch_label = labels[np.array(batch_nodes)]
		optimizer.zero_grad()

		start_time = time.time()

		if args.cuda:
			loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
		else:
			loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))

		end_time = time.time()

		epoch_time += end_time - start_time

		loss.backward()
		optimizer.step()
		loss += loss.item()

	print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')
	overall_time += epoch_time

	#testing the model for every $test_epoch$ epoch
	if epoch % args.test_epochs == 0:

		val_auc, val_precision, val_a_p, val_recall, val_f1 = test_model(idx_val, y_val, gnn_model)
		auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, gnn_model)
		performance_log.append([auc, precision, a_p, recall, f1])
		final_epoch = epoch
		final_metrics = {
			"auroc": auc,
			"precision": precision,
			"average_precision": a_p,
			"recall": recall,
			"macro_f1": f1,
		}
		print(
			f"Epoch {epoch}: val_f1={val_f1:.4f} "
			f"test_auc={auc:.4f} test_precision={precision:.4f} "
			f"test_recall={recall:.4f} test_f1={f1:.4f}"
		)
		score = val_f1
		if score >= best_score:
			best_score = score
			best_epoch = epoch
			best_metrics = final_metrics.copy()

print("The training time per epoch")
print(overall_time/args.num_epochs)
if best_metrics is not None:
	print("[test best by train-copy val_f1]")
	print(f"best_epoch: {best_epoch}")
	for key in ["precision", "recall", "macro_f1", "auroc", "average_precision"]:
		print(f"{key}: {best_metrics[key]:.4f}")
	print("[test final]")
	print(f"final_epoch: {final_epoch}")
	for key in ["precision", "recall", "macro_f1", "auroc", "average_precision"]:
		print(f"{key}: {final_metrics[key]:.4f}")
