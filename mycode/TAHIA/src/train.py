import os
import sys
import copy
import numpy as np
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)
from early_stopper import *
from hin_loader import HIN
from evaluation import *
import util_funcs as uf
from config import TAHIAConfig
from TAHIA import TAHIA
from llm_feedback import extract_next_run, run_llm_feedback
import warnings
import time
import torch
import argparse

warnings.filterwarnings('ignore')
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]


def train_tahia(args, gpu_id=0, log_on=True):
    uf.seed_init(args.seed)
    uf.shell_init(gpu_id=gpu_id)
    cf = TAHIAConfig(args.dataset)

    # ! Modify config
    overrides = {k: v for k, v in args.__dict__.items() if v is not None}
    cf.update(overrides)
    dap_defaults = {
        'use_dap_encoder': True,
        'dap_bins': 8,
        'dap_temp': 0.2,
        'dap_dropout': 0.0,
        'dap_gate_init': -3.0,
    }
    for k, v in dap_defaults.items():
        if not hasattr(cf, k):
            setattr(cf, k, v)
        if k not in cf.model_conf_list:
            cf.model_conf_list.append(k)
    #在 Python 中，对象的属性可以在运行时动态添加，即使这些属性没有在类的 __init__ 方法中声明。
    cf.dev = torch.device("cuda:0" if gpu_id >= 0 else "cpu")
    # cf.dev = torch.device("cpu")

    # ! Load Graph
    g = HIN(cf.dataset).load_mp_embedding(cf)
    data_profile = {
        'dataset': g.dataset,
        'feature_shape': g.features.shape,
        'edge_shapes': {k: v.shape for k, v in g.edges.items()},
        'edge_nnz': {k: int(v.nnz) for k, v in g.edges.items()},
        'label_groups': [len(x) for x in g.labels],
        't_info': g.t_info,
        'mp_emb_shapes': {k: v.shape for k, v in g.mp_emb_dict.items()},
    }
    print(data_profile)
    #
    #未出现的属性在meta_data.pkl中，g.t_info,g.r_info,g.rel2id,g.relations,g.undirected_relations
    # 't_info': {'c': {'ind': range(0, 13056), 'cnt': 13056}, 's': {'ind': range(13056, 13351), 'cnt': 295}},
    # 'rel2id': {'c-s': 0, 's-c': 1}, 'id2rel': {0: 'c-s', 1: 's-c'}, 'n_feat': 32,
    # 'relations': ['c-s', 's-c'], 'types': ['c', 's'],
    # 'undirected_relations': {'c-s'},
    # 'r_info': {'c-s': (0, 13056, 13056, 13351), 's-c': (13056, 13351, 0, 13056)},
    #g.t_info: 表示节点类型及其相对应的序号范围
    #g.r_info: 为四元组，元组前两位表示起始节点的序号范围，元组后两位表示终止节点的序号范围
    print(f'Dataset: {cf.dataset}, {g.t_info}')
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch(cf)

    # ! Train Init
    if not log_on: uf.block_logs()
    print(f'{cf}\nStart training..')
    default_loss_weights = {
        'acm': (0.3, 0.7),
        'dblp': (0.2, 0.8),
        'amazon': (0.07, 0.93),
        'credit': (0.06, 0.94),
    }
    default_w0, default_w1 = default_loss_weights[cf.dataset]
    loss_weight = [
        getattr(cf, 'loss_w0', default_w0),
        getattr(cf, 'loss_w1', default_w1),
    ]
    cla_loss = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(loss_weight, dtype=torch.float32, device=cf.dev)
    )

    model = TAHIA(cf, g, args.dataset)
    model.to(cf.dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    #设置早停机制，防止过拟合，patience，早停机制的耐心参数，表示模型性能没有改善的情况下，继续训练的次数
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file)

    dur = []
    train_loss_curve = []
    val_loss_curve = []
    val_metric_curve = []
    epoch_time_curve = []
    w_list = []
    pre_ = []
    rec_ = []
    f1_ = []
    epoch_ = []
    best_f1 = 0
    best_rec = 0
    best_pre = 0
    result = []
    for epoch in range(cf.epochs):
        # ! Train
        t0 = time.time()
        model.train()
        logits, adj_new = model(features, adj, mp_emb, epoch)
        t1 = time.time()
        # train_f1, train_mif1 = eval_logits(logits, train_x, train_y)
        w_list.append(uf.print_weights(model))
        # cla_loss.to('cpu')
        l_pred = cla_loss(logits[train_x].float(), train_y)
        t2 = time.time()
        loss = l_pred
        train_loss_curve.append(float(loss.detach().cpu()))
        optimizer.zero_grad()
        loss.backward()
        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        #DEBUG
        print(
            f"forward={t1-t0:.2f}s, "
            f"loss={t2-t1:.2f}s, "
            f"backward={t3-t2:.2f}s, "
            f"step={t4-t3:.2f}s"
        )
        # ! Valid
        model.eval()
        with torch.no_grad():
            # logits = model.GCN(features, adj_new)
            auc, trec, tpre, acc, val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
            val_loss = cla_loss(logits[val_x].float(), val_y)
            val_loss_curve.append(float(val_loss.detach().cpu()))
            val_metric_curve.append({
                'epoch': epoch,
                'auc': float(auc),
                'precision_macro': float(tpre),
                'recall_macro': float(trec),
                'f1_macro': float(val_f1),
                'accuracy': float(acc),
            })
            epoch_.append('epoch' + str(epoch))
            pre_.append(round(tpre * 100, 2))
            rec_.append(round(trec * 100, 2))
            f1_.append(round(val_f1 * 100, 2))
            print('precision:{}, recall:{}, f1:{}'.format(tpre, trec, val_f1))
            result.append('Epoch' + str(epoch) + '---->' + 'precision:{}, recall:{}, f1:{}'.format(tpre, trec, val_f1))
            if best_f1 < val_f1:
                best_f1 = val_f1
                best_rec = trec
                best_pre = tpre

        dur.append(time.time() - t0)
        epoch_time_curve.append(float(dur[-1]))
        uf.print_train_log(epoch, dur, loss, 0, 0)

        if cf.early_stop > 0:
            if stopper.step(val_f1, model, epoch, auc, trec, tpre, acc):
                print(f'Early stopped, loading model from epoch-{stopper.best_epoch}')
                break

    if cf.early_stop > 0:
        model.load_state_dict(torch.load(cf.checkpoint_file))
    logits, _ = model(features, adj, mp_emb, 301)
    cf.update(w_list[stopper.best_epoch])
    runtime_profile = {
        'epoch_times': epoch_time_curve,
        'mean_epoch_time': float(np.mean(epoch_time_curve)) if len(epoch_time_curve) > 0 else 0.0,
        'total_train_time': float(np.sum(epoch_time_curve)) if len(epoch_time_curve) > 0 else 0.0,
        'cuda_max_memory_mb': (
            float(torch.cuda.max_memory_allocated(cf.dev) / 1024 / 1024)
            if cf.dev.type == 'cuda' else 0.0
        ),
    }
    training_trace = {
        'train_loss': train_loss_curve,
        'val_loss': val_loss_curve,
        'val_metrics': val_metric_curve,
    }
    eval_and_save(cf, logits, test_x, test_y, val_x, val_y, stopper)
    llm_feedback_result = run_llm_feedback(
        cf, model, logits, test_x, test_y, val_x, val_y, stopper,
        w_list[stopper.best_epoch], loss_weight, args,
        data_profile=data_profile,
        training_trace=training_trace,
        runtime_profile=runtime_profile
    )
    cf.llm_feedback_result = llm_feedback_result
    if not log_on: uf.enable_logs()
    print('####################### Epoch ##########################')
    print(epoch_)
    print('####################### Recall ##########################')
    print(rec_)
    print('####################### Precision ##########################')
    print(pre_)
    print('####################### F1 ##########################')
    print(f1_)
    print('####################### Result ##########################')
    print(result)
    print('precision:{}, recall:{}, f1:{}'.format(best_pre, best_rec, best_f1))
    
    return cf


def train_tahia_two_stage(args, gpu_id=0):
    print("####################### LLM Feedback Stage 1 ##########################")
    stage1_args = copy.copy(args)
    stage1_args.llm_feedback = True
    stage1_cf = train_tahia(stage1_args, gpu_id=gpu_id)

    base_dap_gate = getattr(stage1_args, "dap_gate_init", None)
    base_loss_w0 = getattr(stage1_args, "loss_w0", None)
    base_loss_w1 = getattr(stage1_args, "loss_w1", None)
    base_params = {
        "dap_gate_init": -3.0 if base_dap_gate is None else base_dap_gate,
        "loss_w0": 0.07 if base_loss_w0 is None else base_loss_w0,
        "loss_w1": 0.93 if base_loss_w1 is None else base_loss_w1,
    }
    next_run = extract_next_run(getattr(stage1_cf, 'llm_feedback_result', None), base_params=base_params)
    if len(next_run) == 0:
        print("LLM feedback did not provide applicable next-run parameters. Stop after stage 1.")
        return stage1_cf
    next_run = {
        k: v for k, v in next_run.items()
        if getattr(stage1_args, k, None) != v
    }
    if len(next_run) == 0:
        print("LLM feedback did not change any applicable parameter. Stop after stage 1.")
        return stage1_cf

    print("####################### LLM Feedback Applied ##########################")
    print(next_run)
    stage2_args = copy.copy(args)
    stage2_args.llm_two_stage = False
    stage2_args.llm_feedback = getattr(args, 'llm_feedback_stage2', False)
    base_exp_name = getattr(args, 'exp_name', 'debug')
    stage2_args.exp_name = getattr(args, 'llm_stage2_exp_name', None) or f"{base_exp_name}_llm_stage2"
    for k, v in next_run.items():
        setattr(stage2_args, k, v)

    print("####################### LLM Feedback Stage 2 ##########################")
    return train_tahia(stage2_args, gpu_id=gpu_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset = 'acm'
    parser.add_argument('--conv_method', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--fgd_th', type=float, default=None)
    parser.add_argument('--fgh_th', type=float, default=None)
    parser.add_argument('--sem_th', type=float, default=None)
    parser.add_argument('--emb_dim', type=int, default=None)
    parser.add_argument('--com_feat_dim', type=int, default=None)
    parser.add_argument('--loss_w0', type=float, default=None)
    parser.add_argument('--loss_w1', type=float, default=None)
    parser.add_argument('--use_dap_encoder', action='store_true', default=None)
    parser.add_argument('--disable_dap_encoder', action='store_false', dest='use_dap_encoder')
    parser.add_argument('--dap_bins', type=int, default=None)
    parser.add_argument('--dap_temp', type=float, default=None)
    parser.add_argument('--dap_dropout', type=float, default=None)
    parser.add_argument('--dap_gate_init', type=float, default=None)
    parser.add_argument('--llm_feedback', action='store_true')
    parser.add_argument('--llm_feedback_dry_run', action='store_true')
    parser.add_argument('--llm_provider', type=str, default='qwen')
    parser.add_argument('--llm_model', type=str, default=None)
    parser.add_argument('--llm_base_url', type=str, default=None)
    parser.add_argument('--llm_api_key_env', type=str, default=None)
    parser.add_argument('--llm_feedback_dir', type=str, default='results/llm_feedback')
    parser.add_argument('--llm_two_stage', action='store_true')
    parser.add_argument('--llm_feedback_stage2', action='store_true')
    parser.add_argument('--llm_stage2_exp_name', type=str, default=None)
    args = parser.parse_args()
    if args.llm_two_stage:
        train_tahia_two_stage(args, gpu_id=args.gpu_id)
    else:
        train_tahia(args, gpu_id=args.gpu_id)
