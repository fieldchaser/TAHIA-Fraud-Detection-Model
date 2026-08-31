import json
import argparse
import time
from torch.utils.data import DataLoader
import torch 
import dgl
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from name import *
import utils
import model

parser = argparse.ArgumentParser()
parser.add_argument('--data', default=AMAZON, help='Dataset used')
parser.add_argument('--seed', type=int, default=1019, help='Random seed')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--nepoch', type=int, default=25, help='Num of epoches')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dim')
parser.add_argument('--layer_num', type=int, default=3, help='Num of layers')
parser.add_argument('--drop_rate', type=float, default=0.05, help='Droup rate')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--test_epoch', type=int, default=25, help='Test epochs')
parser.add_argument('--alpha', type=float, default=0.5, help='Coef1')
parser.add_argument('--beta', type=float, default=0.5, help='Coef2')
args = parser.parse_args()

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

data = args.data
seed = args.seed
lr = args.lr
nepoch = args.nepoch
hid_dim = args.hid_dim
layer_num = args.layer_num
drop_rate = args.drop_rate
batch_size = args.batch_size
test_epoch = args.test_epoch
alpha = args.alpha
beta = args.beta

utils.set_seed(seed)

if data in PARAMETERS:
   lr, layer_num, drop_rate, alpha, beta, stdneg, stdpos  = set_paras(data)  

cneg = torch.FloatTensor(layer_num).normal_(-0.1, stdneg)
cpos = torch.FloatTensor(layer_num).normal_(0.1, stdpos)
utils.set_seed(seed)

print("Start loading data: {}".format(data))
s = time.time()
graph, train_index, val_index, test_index = utils.load_data(data, layer_num)
e = time.time()
print("Loading successfully, time cost: {:.2f}".format(e - s))

labels = graph.ndata['label']
train_mask = torch.zeros_like(labels).bool()
val_mask = torch.zeros_like(labels).bool()
test_mask = torch.zeros_like(labels).bool()
train_mask[train_index] = 1
val_mask[val_index] = 1
test_mask[test_index] = 1
graph.ndata['train_mask'] = train_mask
graph.ndata['val_mask'] = val_mask
graph.ndata['test_mask'] = test_mask

train_loader = DataLoader(train_index, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_index, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_index, batch_size=100000, shuffle=False, drop_last=False)

spacegnn = model.SpaceGNN(graph.ndata['feature'].shape[1], hid_dim, num_class, layer_num, drop_rate, cneg, cpos)
optimizer = torch.optim.Adam(spacegnn.parameters(), lr=lr)
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

best_auc, best_f1 = 0, 0
total = 0
best_epoch = 0

def train(spacegnn, optimizer, graph, sampler, data_loader, alpha, beta):
    spacegnn.train()
    for index in data_loader:
        input_nodes, output_nodes, subgraphs = sampler.sample_blocks(graph, index)
        
        probs1, probs2, probs3 = spacegnn(subgraphs)
        target = subgraphs[-1].dstdata['label']
        probs = (1 - beta) * ((1 - alpha) * probs1 + alpha * probs2) + beta * probs3
        loss = F.nll_loss(probs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluation(spacegnn, graph, sampler, data_loader, alpha, beta, best_thres=-1):
    spacegnn.eval()
    probs1_list = []
    probs2_list = []
    probs3_list = []
    targets_list = []

    with torch.no_grad():
        for index in data_loader:
            _, _, blocks = sampler.sample_blocks(graph, index)
            targets = blocks[-1].dstdata['label']
            probs1, probs2, probs3 = spacegnn(blocks)

            targets_list.append(targets)
            probs1_list.append(probs1.detach())
            probs2_list.append(probs2.detach())
            probs3_list.append(probs3.detach())
        
        targets = torch.cat(targets_list, dim=0)
        probs1 = torch.cat(probs1_list, dim=0)
        probs2 = torch.cat(probs2_list, dim=0)
        probs3 = torch.cat(probs3_list, dim=0)
        
        probs1 = probs1.exp()[:, 1]
        probs2 = probs2.exp()[:, 1]
        probs3 = probs3.exp()[:, 1]

    targets = targets.cpu().detach().numpy()
    probs1 = probs1.cpu().detach().numpy()
    probs2 = probs2.cpu().detach().numpy()
    probs3 = probs3.cpu().detach().numpy()

    probs = (1 - beta) * ((1 - alpha) * probs1 + alpha * probs2) + beta * probs3
    auc = roc_auc_score(targets, probs)

    if best_thres == -1:
        best_thres = utils.get_best_auc_f1(probs, targets)

    preds = (probs > best_thres).astype(int)
    f1 = f1_score(targets, preds, average='macro')
    return auc, f1, best_thres

for epoch in range(nepoch):
    if epoch % test_epoch == 0:
        s = time.time()
    train(spacegnn, optimizer, graph, sampler, train_loader, alpha, beta)
    if not ((epoch + 1) % test_epoch == 0) :
        continue
    e = time.time()
    print("Total train epoch: {}, time cost: {:.2f}".format(epoch + 1, e - s))

    s = time.time()
    val_auc, val_f1, best_thres = evaluation(spacegnn, graph, sampler, val_loader, beta, alpha)
    e = time.time()
    print("Best threshold: {:.2f}".format(best_thres))
    print("Epoch: {}".format(epoch))
    print("Val AUC: {:.4f}, F1: {:.4f}, time cost: {:.2f}".format(val_auc, val_f1, e - s))


    s = time.time()
    test_auc, test_f1, _ = evaluation(spacegnn, graph, sampler, test_loader, alpha, beta, best_thres)
    e = time.time()
    print("Test AUC: {:.4f}, F1: {:.4f}, time cost: {:.2f}".format(test_auc, test_f1, e - s))

    if total <= val_auc + val_f1:
        total = val_auc + val_f1
        best_epoch = epoch + 1
        best_auc = test_auc
        best_f1 = test_f1
    
print("Best epoch: {}, test AUC: {:.4f}, F1: {:.4f}".format(best_epoch, best_auc, best_f1))
