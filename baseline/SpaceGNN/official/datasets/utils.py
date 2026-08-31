import os
import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs
import time
from sklearn.model_selection import train_test_split
import random
import pygod.utils as pygodutils

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

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed


def load_data(data):
    datapath = os.path.join(os.path.join(DATADIR, data), data)
    if data in [WEIBO, REDDIT, TOLOKERS, AMAZON, TFINANCE, YELP, QUESTIONS, DGRAPHFIN, TSOCIAL]:
        graph, label_dict = load_graphs(datapath)
        graph = graph[0]
    else:
        print('no such dataset')
        exit(1)
    graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature'] = graph.ndata['feature'].float()
    

    if data == TFINANCE:
        graph.ndata['label'] = graph.ndata['label'].argmax(dim=1)
    y = graph.ndata['label']
    return y


def split_train_val_test(data, y, trainsz, testsz):
    normalinds = []
    abnormalinds = []
    wronglabels = []
    for i, label in enumerate(y):
        
        if int(label) == 0:
            normalinds.append(i)
        elif int(label) == 1:
            abnormalinds.append(i)
        else:
            wronglabels.append(label)
        
    if wronglabels:
        print("Exist wrong label: {}".format(torch.unique(torch.LongTensor(wronglabels), return_counts=True)))

    random.shuffle(normalinds)
    random.shuffle(abnormalinds)
    
    
    trainnum = 50
    trainnormalratio = len(normalinds) / (len(normalinds) + len(abnormalinds))
    trainnormalsz = int(trainnum * trainnormalratio)
    trainabnormalsz = int(trainnum * (1 - trainnormalratio)) + 1
   
    valnum = 50
    valnormalratio = len(normalinds) / (len(normalinds) + len(abnormalinds))
    valnormalsz = int(valnum * valnormalratio)
    valabnormalsz = int(valnum * (1 - valnormalratio)) + 1

    
    
    train_normal = np.array(normalinds[: trainnormalsz])
    val_normal = np.array(normalinds[trainnormalsz: trainnormalsz + valnormalsz])#* 2])
    test_normal = np.array(normalinds[trainnormalsz + valnormalsz: ])#* 2: ])

    train_abnormal = np.array(abnormalinds[: trainabnormalsz])
    val_abnormal = np.array(abnormalinds[trainabnormalsz: trainabnormalsz +valabnormalsz])#* 2])
    test_abnormal = np.array(abnormalinds[trainabnormalsz + valabnormalsz: ])#* 2:])
    
    train_index = np.concatenate((train_normal, train_abnormal))
    val_index = np.concatenate((val_normal, val_abnormal))
    test_index = np.concatenate((test_normal, test_abnormal))
    

    random.shuffle(train_index)
    random.shuffle(val_index)
    random.shuffle(test_index)


    print("Train size: {}, normal size: {}, abnormal size: {}".format(len(train_index), len(train_normal), len(train_abnormal)))
    print("Val size: {}, normal size: {}, abnormal size: {}".format(len(val_index), len(val_normal), len(val_abnormal)))
    print("Test size: {}, normal size: {}, abnormal size: {}".format(len(test_index), len(test_normal), len(test_abnormal)))

    print("Total size: {}, generate size: {}".format(len(y), len(train_index) + len(val_index) + len(test_index)))


    train_path = os.path.join(data, data + TRAIN)
    val_path = os.path.join(data, data + VAL)
    test_path = os.path.join(data, data + TEST)

    np.savetxt(train_path, train_index, fmt='%d')
    np.savetxt(val_path, val_index, fmt='%d')
    np.savetxt(test_path, test_index, fmt='%d')

