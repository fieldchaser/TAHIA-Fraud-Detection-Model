import random
import numpy as np
import torch
import os
from dgl.data.utils import load_graphs
import dgl
from sklearn.metrics import f1_score, roc_auc_score

from name import *

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

import dgl.function as fn
def propogate(h, graph):
    graph.ndata['h'] = h
    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
    return graph.ndata.pop('h')

def load_data(data, layer_num):
    datadir = os.path.join(DATADIR, data)
    datapath = os.path.join(datadir, data)
    if data in [WEIBO, REDDIT, TOLOKERS, AMAZON, TFINANCE, YELP, QUESTIONS, DGRAPHFIN, TSOCIAL]:
        graph, label_dict = load_graphs(datapath)
        graph = dgl.to_homogeneous(graph[0], ndata=['feature', 'label'])
        graph = graph.add_self_loop()
    else:
        print('no such dataset')
        exit(1)
    graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature_0'] = graph.ndata['feature'].float()
    if data == TFINANCE:
        graph.ndata['label'] = graph.ndata['label'].argmax(dim=1)

    h = graph.ndata['feature_0']
    for i in range(1, layer_num):
        key = 'feature' + '_{}'.format(i)
        h = propogate(h, graph)
        graph.ndata[key] = h

    graph = graph.long()

    train_path = os.path.join(datadir, data + TRAIN)
    train_index = np.loadtxt(train_path, dtype=np.int64)

    val_path = os.path.join(datadir, data + VAL)
    val_index = np.loadtxt(val_path, dtype=np.int64)

    test_path = os.path.join(datadir, data + TEST)
    test_index = np.loadtxt(test_path, dtype=np.int64)

    return graph, torch.LongTensor(train_index), torch.LongTensor(val_index), torch.LongTensor(test_index)

def get_best_auc_f1(probs, targets):
    best_total, best_thres = -1, -1
    thres_arr = np.linspace(0.05, 0.95, 19)
    for thres in thres_arr:
        preds = np.zeros_like(targets)
        preds[probs > thres] = 1
        auc = roc_auc_score(targets, probs)
        f1 = f1_score(targets, preds, average='macro')
        if auc + f1 >= best_total:
            best_total = auc + f1
            best_thres = thres

    return best_thres
