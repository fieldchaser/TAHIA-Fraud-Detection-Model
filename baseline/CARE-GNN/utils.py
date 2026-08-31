import random as rd
import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score, \
	classification_report
from collections import defaultdict
from pathlib import Path
import sys

try:
	import pickle5 as pickle
except ImportError:
	import pickle


"""
	Utility functions to handle data and evaluate model.
"""


def install_scipy_sparse_pickle_aliases():
	try:
		import scipy.sparse._csr as csr_module
	except ImportError:
		import scipy.sparse.csr as csr_module
	sys.modules.setdefault("scipy.sparse.csr", csr_module)
	sys.modules.setdefault("scipy.sparse._csr", csr_module)


def load_pkl_data(prefix, data):
	path = Path(prefix) / data
	with (path / 'node_features.pkl').open('rb') as f:
		install_scipy_sparse_pickle_aliases()
		feat_data = pickle.load(f)
	with (path / 'labels.pkl').open('rb') as f:
		label_file = pickle.load(f)
	with (path / f'{data}_adjlists.pickle').open('rb') as file:
		homo = pickle.load(file)
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


def sparse_matrix_to_adjlist(sp_matrix):
	sp_matrix = sp_matrix.tocsr()
	sp_matrix = sp_matrix.maximum(sp_matrix.T)
	sp_matrix = sp_matrix + sp.eye(sp_matrix.shape[0], dtype=np.float32)
	adj_lists = defaultdict(set)
	rows, cols = sp_matrix.nonzero()
	for src, dst in zip(rows, cols):
		adj_lists[int(src)].add(int(dst))
	for node in range(sp_matrix.shape[0]):
		adj_lists[node].add(node)
	return adj_lists


def load_tahia_credit():
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
	homo = sparse_matrix_to_adjlist(sum(edges.values()).astype(np.float32))
	return homo, torch.from_numpy(feat_data), labels


def load_data(data):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""

	prefix = Path(__file__).resolve().parent / 'data'
	if data == 'credit':
		homo, feat_data, labels = load_tahia_credit()
		relation1 = None
		relation2 = None
		relation3 = None
	elif (Path(prefix) / data / 'node_features.pkl').exists() and data not in {'acm', 'dblp', 'yelp'}:
		homo, feat_data, labels = load_pkl_data(prefix, data)
		relation1 = None
		relation2 = None
		relation3 = None
	elif data == 'yelpchi':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
	elif data == 'amazon':
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
	elif data == 'zhuanzhang':
		data_file = pickle.load(open(prefix + 'zhuanzhang/zhuanzhang.pkl', 'rb'))
		labels = np.array(data_file['labels'], dtype='int')
		feat_data = data_file['features']
		homo = data_file['neighbor']
		# load the preprocessed adj_lists
		for k, v in homo.items():
			homo[k] = set(v)
		relation1 = None
		relation2 = None
		relation3 = None
	elif data == 'xinyongka':
		data_file = pickle.load(open(prefix + 'xinyongka/credit.pkl', 'rb'))
		labels = np.array(data_file['labels'], dtype='int')
		feat_data = data_file['features']
		homo = data_file['neighbor']
		# load the preprocessed adj_lists
		for k, v in homo.items():
			homo[k] = set(v)
		relation1 = None
		relation2 = None
		relation3 = None
	elif data == 'elliptic':
		with open(prefix + 'elliptic.dat', 'rb') as file:
			data_file = pickle.load(file)
		labels = np.array(data_file.y, dtype='int')
		feat_data = data_file.x
		file.close()
		with open(prefix + 'elliptic.pickle', 'rb') as file:
			homo = pickle.load(file)
		relation1 = None
		relation2 = None
		relation3 = None
	elif data == 'tfinance':
		data_file = loadmat(prefix + 'tfinance.mat')
		labels = np.array(data_file['Label'].flatten(), dtype='int')
		feat_data = data_file['Attributes']
		# load the preprocessed adj_lists
		with open(prefix + 'tfinance.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
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
	return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename):
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()


def pos_neg_split(nodes, labels):
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
	"""
	Under-sample the negative nodes
	:param pos_nodes: a list of positive nodes
	:param neg_nodes: a list negative nodes
	:param scale: the under-sampling scale
	:return: a list of under-sampled batch nodes
	"""

	aux_nodes = cp.deepcopy(neg_nodes)
	aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
	batch_nodes = pos_nodes + aux_nodes

	return batch_nodes


def test_sage(test_cases, labels, model, batch_size):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	gnn_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)
		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_list))
	print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
	print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
	print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
	print(f"GNN auc: {auc_gnn:.4f}")
	print(f"GNN ap: {ap_gnn:.4f}")

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
def test_care(test_cases, labels, model, batch_size):
	"""
	Test the performance of CARE-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	f1_label1 = 0.0
	acc_label1 = 0.00
	recall_label1 = 0.0
	gnn_list = []
	label_list1 = []

	predict_all = np.array([], dtype=int)
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)
		prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
		pred = prob2pred(prob_arr)
		predict_all = np.append(predict_all, np.array(pred))
		label_list1.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
		# f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		# acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
		# recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
		#
		# f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		# acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
		# recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		#
		# gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
		# label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())
	auc_label1 = roc_auc_score(labels, np.array(label_list1))
	# auc_gnn = roc_auc_score(labels, np.array(gnn_list))
	# ap_gnn = average_precision_score(labels, np.array(gnn_list))
	# auc_label1 = roc_auc_score(labels, np.array(label_list1))
	# ap_label1 = average_precision_score(labels, np.array(label_list1))
	# print(f"GNN F1: {f1_gnn / test_batch_num:.4f}")
	# print(f"GNN Accuracy: {acc_gnn / test_batch_num:.4f}")
	# print(f"GNN Recall: {recall_gnn / test_batch_num:.4f}")
	# print(f"GNN auc: {auc_gnn:.4f}")
	# print(f"GNN ap: {ap_gnn:.4f}")
	# print(f"Label1 F1: {f1_label1 / test_batch_num:.4f}")
	# print(f"Label1 Accuracy: {acc_label1 / test_batch_num:.4f}")
	# print(f"Label1 Recall: {recall_label1 / test_batch_num:.4f}")
	# print(f"Label1 auc: {auc_label1:.4f}")
	# print(f"Label1 ap: {ap_label1:.4f}")
	print(f"AUC: {auc_label1:.4f}")
	print(classification_report(labels, predict_all, digits=4))
	return None, auc_label1, recall_gnn, recall_label1


def evaluate_care(test_cases, labels, model, batch_size):
	test_batch_num = int(len(test_cases) / batch_size) + 1
	probs = []
	preds = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		if i_start >= i_end:
			continue
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob, _ = model.to_prob(batch_nodes, batch_label, train_flag=False)
		prob_arr = gnn_prob.data.cpu().numpy()
		probs.extend(prob_arr[:, 1].tolist())
		preds.extend(prob_arr.argmax(axis=1).tolist())
	probs = np.asarray(probs)
	preds = np.asarray(preds)
	return {
		'precision': precision_score(labels, preds, average='macro', zero_division=0),
		'recall': recall_score(labels, preds, average='macro', zero_division=0),
		'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
		'auroc': roc_auc_score(labels, probs),
	}


def evaluate_sage(test_cases, labels, model, batch_size):
	test_batch_num = int(len(test_cases) / batch_size) + 1
	probs = []
	preds = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		if i_start >= i_end:
			continue
		batch_nodes = test_cases[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)
		prob_arr = gnn_prob.data.cpu().numpy()
		probs.extend(prob_arr[:, 1].tolist())
		preds.extend(prob_arr.argmax(axis=1).tolist())
	probs = np.asarray(probs)
	preds = np.asarray(preds)
	return {
		'precision': precision_score(labels, preds, average='macro', zero_division=0),
		'recall': recall_score(labels, preds, average='macro', zero_division=0),
		'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
		'auroc': roc_auc_score(labels, probs),
	}
