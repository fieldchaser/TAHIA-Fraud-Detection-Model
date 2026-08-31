import time
import os
import random
import argparse
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics import (
	f1_score,
	accuracy_score,
	recall_score,
	roc_auc_score,
	average_precision_score,
	precision_score,
	classification_report,
)
from collections import defaultdict
from pathlib import Path
import math

try:
	import pickle5 as pickle
except ImportError:
	import pickle


def install_scipy_sparse_pickle_aliases():
	try:
		import scipy.sparse._csr as csr_module
	except ImportError:
		import scipy.sparse.csr as csr_module
	sys.modules.setdefault("scipy.sparse.csr", csr_module)
	sys.modules.setdefault("scipy.sparse._csr", csr_module)


def sample_adjlist(adj_lists, max_neighbors=10):
	"""
	Limit the number of neighbors for each node.

	This makes pkl adjacency loading consistent with sparse_to_adjlist(),
	where each node keeps at most 10 neighbors by default.

	Args:
		adj_lists: dict[node_id] -> iterable of neighbor ids
		max_neighbors: max sampled neighbors per node.
					   If max_neighbors <= 0 or None, keep full adjacency.

	Returns:
		dict[node_id] -> set(neighbor ids)
	"""
	if max_neighbors is None or max_neighbors <= 0:
		return {k: set(v) for k, v in adj_lists.items()}

	sampled = {}
	for k, v in adj_lists.items():
		neighbors = list(v)
		if len(neighbors) > max_neighbors:
			neighbors = random.sample(neighbors, max_neighbors)
		sampled[k] = set(neighbors)

	return sampled


def load_pkl_data(prefix, data, max_neighbors=10):
	path = Path(prefix) / data

	with (path / 'node_features.pkl').open('rb') as f:
		install_scipy_sparse_pickle_aliases()
		feat_data = pickle.load(f)

	with (path / 'labels.pkl').open('rb') as f:
		label_file = pickle.load(f)

	with (path / f'{data}_adjlists.pickle').open('rb') as file:
		homo = pickle.load(file)

	# neighbor sampling for pkl adjacency list
	homo = sample_adjlist(homo, max_neighbors=max_neighbors)

	labels = []
	for part in label_file:
		labels.extend(np.asarray(part).tolist())

	labels = sorted(labels, key=lambda t: t[0])
	labels = np.array(labels)[:, 1].astype('int')
	labels[np.where(labels == 2)] = 0
	labels[np.where(labels == 3)] = 0

	if sp.issparse(feat_data):
		feat_data = torch.from_numpy(feat_data.todense())
	else:
		feat_data = torch.from_numpy(np.asarray(feat_data))

	return homo, feat_data, labels


def sparse_matrix_to_adjlist(sp_matrix, max_neighbors=10):
	sp_matrix = sp_matrix.tocsr()
	sp_matrix = sp_matrix.maximum(sp_matrix.T)
	sp_matrix = sp_matrix + sp.eye(sp_matrix.shape[0], dtype=np.float32)
	adj_lists = defaultdict(set)
	rows, cols = sp_matrix.nonzero()
	for src, dst in zip(rows, cols):
		adj_lists[int(src)].add(int(dst))
	return sample_adjlist(adj_lists, max_neighbors=max_neighbors)


def load_tahia_credit(max_neighbors=10):
	path = Path(__file__).resolve().parents[2] / 'mycode' / 'TAHIA' / 'data' / 'credit'
	with (path / 'node_features.pkl').open('rb') as f:
		feat_data = pickle.load(f)
	with (path / 'labels.pkl').open('rb') as f:
		label_file = pickle.load(f)
	with (path / 'edges.pkl').open('rb') as f:
		edges = pickle.load(f)

	feat_data = np.asarray(feat_data, dtype=np.float32)
	labels = np.asarray(label_file[0], dtype=np.int64)
	labels = (labels == 1).astype(np.int64)
	homo = sparse_matrix_to_adjlist(sum(edges.values()).astype(np.float32), max_neighbors=max_neighbors)
	return homo, torch.from_numpy(feat_data), labels


def sparse_to_adjlist(sp_matrix):
	"""Transfer sparse matrix to adjacency list"""

	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])

	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()

	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)

	adj_lists = {
		keya: random.sample(adj_lists[keya], 10) if len(adj_lists[keya]) >= 10 else adj_lists[keya]
		for i, keya in enumerate(adj_lists)
	}

	return adj_lists


def load_data(data, max_neighbors=10):
	"""load data"""
	prefix = Path(__file__).resolve().parent / 'data'

	if data == 'credit':
		homo, feat_data, labels = load_tahia_credit(max_neighbors=max_neighbors)
		relation1 = None
		relation2 = None
		relation3 = None

	elif (Path(prefix) / data / 'node_features.pkl').exists() and data not in {'acm', 'dblp', 'yelp'}:
		homo, feat_data, labels = load_pkl_data(prefix, data, max_neighbors=max_neighbors)
		relation1 = None
		relation2 = None
		relation3 = None

	elif data == 'yelpchi':

		yelp = loadmat('data/YelpChi.mat')

		# 当前 train.py 只使用 adj_lists = [homo]
		# 所以这里不要加载 net_rur / net_rtr / net_rsr，避免 WSL 内存爆掉
		homo = sparse_to_adjlist(yelp['homo'])
		relation1 = None
		relation2 = None
		relation3 = None

		feat_data = yelp['features'].toarray()
		labels = yelp['label'].flatten()

	elif data == 'amazon':

		amz = loadmat('data/Amazon.mat')
		homo = sparse_to_adjlist(amz['homo'])
		relation1 = sparse_to_adjlist(amz['net_upu'])
		relation2 = sparse_to_adjlist(amz['net_usu'])
		relation3 = sparse_to_adjlist(amz['net_uvu'])
		feat_data = amz['features'].toarray()
		labels = amz['label'].flatten()

	elif data == "Amazon_demo":
		amz = loadmat('data/Amazon_demo.mat')
		homo = sparse_to_adjlist(amz['homo'])
		relation1 = sparse_to_adjlist(amz['net_upu'])
		relation2 = sparse_to_adjlist(amz['net_usu'])
		relation3 = sparse_to_adjlist(amz['net_uvu'])
		feat_data = amz['features'].toarray()
		labels = amz['label'].flatten()

	elif data == "Yelp_demo":
		yelp = loadmat('data/Yelp_demo.mat')
		homo = sparse_to_adjlist(yelp['homo'])
		relation1 = sparse_to_adjlist(yelp['net_rur'])
		relation2 = sparse_to_adjlist(yelp['net_rtr'])
		relation3 = sparse_to_adjlist(yelp['net_rsr'])
		feat_data = yelp['features'].toarray()
		labels = yelp['label'].flatten()

	elif data == 'zhuanzhang':
		data_file = pickle.load(open(prefix + 'zhuanzhang/zhuanzhang.pkl', 'rb'))
		labels = np.array(data_file['labels'], dtype='int')
		feat_data = data_file['features']
		homo = data_file['neighbor']

		for k, v in homo.items():
			homo[k] = set(v)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		relation1 = None
		relation2 = None
		relation3 = None

	elif data == 'xinyongka':
		data_file = pickle.load(open(prefix + 'xinyongka/credit.pkl', 'rb'))
		labels = np.array(data_file['labels'], dtype='int')
		feat_data = data_file['features']
		homo = data_file['neighbor']

		for k, v in homo.items():
			homo[k] = set(v)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		relation1 = None
		relation2 = None
		relation3 = None

	elif data == 'elliptic':
		with open('data/elliptic.dat', 'rb') as file:
			data_file = pickle.load(file)

		labels = np.array(data_file.y, dtype='int')
		feat_data = data_file.x

		with open('data/elliptic.pickle', 'rb') as file:
			homo = pickle.load(file)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		relation1 = None
		relation2 = None
		relation3 = None

	elif data == 'tfinance':
		data_file = loadmat('data/tfinance.mat')
		labels = np.array(data_file['Label'].flatten(), dtype='int')
		feat_data = data_file['Attributes']

		with open('data/tfinance.pickle', 'rb') as file:
			homo = pickle.load(file)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		relation1 = None
		relation2 = None
		relation3 = None

	elif data == 'acm':
		with open(prefix + 'acm/node_features.pkl', 'rb') as f:
			feat_data = pickle.load(f)

		with open(prefix + 'acm/labels.pkl', 'rb') as f:
			label_file = pickle.load(f)

		with open(prefix + 'acm/acm_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		labels = []
		labels.extend(label_file[0].tolist())
		labels.extend(label_file[1].tolist())
		labels.extend(label_file[2].tolist())

		labels = sorted(labels, key=lambda t: t[0])
		labels = np.array(labels)[:, 1]
		labels[np.where(labels == 2)] = 0

		relation1 = None
		relation2 = None
		relation3 = None
		feat_data = torch.from_numpy(feat_data.todense())

	elif data == 'dblp':
		with open(prefix + 'dblp/node_features.pkl', 'rb') as f:
			feat_data = pickle.load(f)

		with open(prefix + 'dblp/labels.pkl', 'rb') as f:
			label_file = pickle.load(f)

		with open(prefix + 'dblp/dblp_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		labels = []
		labels.extend(label_file[0])
		labels.extend(label_file[1])
		labels.extend(label_file[2])

		labels = sorted(labels, key=lambda t: t[0])
		labels = np.array(labels)[:, 1]
		labels[np.where(labels == 2)] = 0
		labels[np.where(labels == 3)] = 0

		relation1 = None
		relation2 = None
		relation3 = None
		feat_data = torch.from_numpy(feat_data.todense())

	elif data == 'yelp':
		with open(prefix + 'yelp/node_features.pkl', 'rb') as f:
			feat_data = pickle.load(f)

		with open(prefix + 'yelp/labels.pkl', 'rb') as f:
			label_file = pickle.load(f)

		with open(prefix + 'yelp/yelp_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)

		homo = sample_adjlist(homo, max_neighbors=max_neighbors)

		labels = []
		labels.extend(label_file[0].tolist())
		labels.extend(label_file[1].tolist())
		labels.extend(label_file[2].tolist())

		labels = sorted(labels, key=lambda t: t[0])
		labels = np.array(labels)[:, 1]
		labels[np.where(labels == 2)] = 0
		labels[np.where(labels == 3)] = 0

		relation1 = None
		relation2 = None
		relation3 = None
		feat_data = torch.from_numpy(feat_data)

	else:
		raise ValueError(f"Unknown dataset: {data}")

	return homo, relation1, relation2, relation3, feat_data, labels


def normalize(mx):
	"""Row-normalize sparse matrix"""

	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.0
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)

	return mx


def prob2pred(y_prob, thres=0.5):
	"""
	Convert probability to predicted results according to given threshold

	:param y_prob: numpy array of probability in [0, 1]
	:param thres: binary classification threshold, default 0.5
	:returns: the predicted result with the same shape as y_prob
	"""
	y_pred = np.zeros_like(y_prob, dtype=np.int32)
	y_pred[y_prob >= thres] = 1
	y_pred[y_prob < thres] = 0

	return y_pred


def test_model(test_cases, labels, model, batch_size=4096):
	"""
	Test the performance of model with mini-batch evaluation.

	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: evaluation batch size
	"""
	model.eval()

	prob_list = []

	with torch.no_grad():
		for i in range(0, len(test_cases), batch_size):
			batch_nodes = test_cases[i:i + batch_size]
			batch_prob = model.to_prob(batch_nodes, train_flag=False)
			prob_list.append(batch_prob.data.cpu())

	gnn_prob = torch.cat(prob_list, dim=0).numpy()

	auc_gnn = roc_auc_score(labels, gnn_prob[:, 1].tolist())
	precision_gnn = precision_score(labels, gnn_prob.argmax(axis=1), average="macro", zero_division=0)
	a_p = average_precision_score(labels, gnn_prob[:, 1].tolist())
	recall_gnn = recall_score(labels, gnn_prob.argmax(axis=1), average="macro", zero_division=0)
	f1 = f1_score(labels, gnn_prob.argmax(axis=1), average="macro", zero_division=0)

	print(f"AUC: {auc_gnn:.4f}")
	print(f"Precision: {precision_gnn:.4f}, Recall: {recall_gnn:.4f}, F1: {f1:.4f}")

	return auc_gnn, precision_gnn, a_p, recall_gnn, f1
