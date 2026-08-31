import numpy
import torch
from utils import load_data, set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random
import argparse
import atexit
import sys
from pathlib import Path


warnings.filterwarnings('ignore')
args = set_params()


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


def get_log_file():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log-file', type=str, default=None)
    log_args, _ = parser.parse_known_args()
    if log_args.log_file:
        return Path(log_args.log_file)
    return Path(__file__).resolve().parents[1] / 'results' / f'{args.dataset}.log'


def setup_logging(log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open('w', buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    atexit.register(log_handle.close)
    print(f'Logging to: {log_path}')
    print(f'Start time: {datetime.datetime.now().isoformat(timespec="seconds")}')


setup_logging(get_log_file())
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

loss_list = []
def train():
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)

    # 【新增代码】强制补全 Amazon 和 YelpChi 的采样率
    if args.dataset in ['amazon', 'yelpchi']:
        args.sample_rate = [2, 2, 2]

    args.nei_num = len(nei_index)
    args.sample_rate = args.sample_rate[:args.nei_num]
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("The number of meta-paths: ", P)
    
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)
        loss_list.append(loss.data.cpu().item())
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HeCo_'+own_str+'.pkl'))
    model.eval()
    os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(feats, mps)
    eval_idx = min(1, len(idx_train) - 1)
    evaluate_ratio = args.ratio[eval_idx] if isinstance(args.ratio, list) else args.ratio
    for i in range(1):
        evaluate(embeds, evaluate_ratio, idx_train[eval_idx], idx_val[eval_idx], idx_test[eval_idx], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    print(loss_list)
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
