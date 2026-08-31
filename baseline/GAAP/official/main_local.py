import argparse
import json
import sys
import time
import types
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from local_data import DEFAULT_DATA_ROOT, load_pkl_graph, set_seed


try:
    import rtdl_num_embeddings  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("rtdl_num_embeddings")
    stub.compute_bins = lambda *args, **kwargs: []
    sys.modules["rtdl_num_embeddings"] = stub

OFFICIAL_CODE_ROOT = Path(__file__).resolve().parent / "mycode"
sys.path.insert(0, str(OFFICIAL_CODE_ROOT))
from nn.gnn import SAGE  # noqa: E402


class TeeLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = path.open("w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_logging(args):
    if args.no_log:
        return None
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parents[1] / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    return TeeLogger(log_dir / f"{args.data}.log")

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ("yes", "true", "t", "1", "y"):
		return True
	if v.lower() in ("no", "false", "f", "0", "n"):
		return False
	raise argparse.ArgumentTypeError("Boolean value expected.")

def build_args():
    parser = argparse.ArgumentParser(description="GAAP local adapter for ACM/DBLP pkl datasets")
    parser.add_argument("--data", choices=["acm", "dblp", "amazon", "yelpchi", "xinyongka", "credit"], default="acm")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--test_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--gnn_n_layers", type=int, default=3)
    parser.add_argument("--gnn_dropout", type=float, default=0.1)
    parser.add_argument("--pre_dropout", type=float, default=0.1)
    parser.add_argument("--gnn_agg", default="max_pool")
    parser.add_argument("--norm_type", choices=["norm01", "row"], default="norm01")
    parser.add_argument("--use_dyple", type=str2bool, default=True)
    parser.add_argument("--use_mha", type=str2bool, default=True)
    parser.add_argument("--gnn_use_res", type=str2bool, default=True)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--no_log", action="store_true")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu on this machine.")
    return torch.device(device_name)


def build_model(in_dim, args):
    cfg = {
        "d_hidden": args.hidden,
        "preprocess": "None",
        "n_bins": args.n_bins,
        "use_dyple": args.use_dyple,
        "dy_raw_bin_width": True,
        "pre_n_layers": 0,
        "pre_use_bn": True,
        "pre_dropout": args.pre_dropout,
        "gnn_n_layers": args.gnn_n_layers,
        "gnn_use_bn": True,
        "gnn_dropout": args.gnn_dropout,
        "gnn_use_res": args.gnn_use_res,
        "gnn_agg": args.gnn_agg,
        "use_mha": args.use_mha,
        "mha_n_layers": 1,
        "mha_n_heads": 4,
    }
    return SAGE(in_dim, 2, **cfg)


def train_epoch(gaap_model, optimizer, graph, sampler, train_loader):
    gaap_model.train()
    total_loss = 0.0
    batches = 0
    for index in train_loader:
        _, _, blocks = sampler.sample_blocks(graph, index)
        logits = gaap_model(blocks, blocks[0].srcdata["feature"])
        target = blocks[-1].dstdata["label"]
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def predict(gaap_model, graph, sampler, index_loader):
    gaap_model.eval()
    probs = []
    labels = []
    node_ids = []
    for index in index_loader:
        _, _, blocks = sampler.sample_blocks(graph, index)
        logits = gaap_model(blocks, blocks[0].srcdata["feature"])
        probs.append(logits.softmax(-1).detach().cpu())
        labels.append(blocks[-1].dstdata["label"].detach().cpu())
        node_ids.append(index.detach().cpu())
    probs = torch.cat(probs)
    labels = torch.cat(labels)
    node_ids = torch.cat(node_ids)
    order = torch.argsort(node_ids)
    return probs[order].numpy(), labels[order].numpy()


def evaluate(gaap_model, graph, sampler, eval_idx, eval_batch_size):
    eval_loader = DataLoader(eval_idx, batch_size=eval_batch_size, shuffle=False, drop_last=False)
    probs_all, labels = predict(gaap_model, graph, sampler, eval_loader)
    probs = probs_all[:, 1]
    preds = probs_all.argmax(axis=1)
    return {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "auroc": roc_auc_score(labels, probs),
    }


def main():
    args = build_args()
    logger = setup_logging(args)
    old_stdout = sys.stdout
    if logger is not None:
        sys.stdout = logger
    try:
        device = resolve_device(args.device)
        set_seed(args.seed)
        print("Model info:")
        print(json.dumps(vars(args), indent=2))
        print(f"Using device: {device}")

        start = time.time()
        graph, train_idx, val_idx, test_idx = load_pkl_graph(args.data, args.data_root, args.norm_type)
        graph = graph.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        print(
            f"Loaded {args.data}: nodes={graph.num_nodes()}, edges={graph.num_edges()}, "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
            f"feature_shape={tuple(graph.ndata['feature'].shape)}"
        )

        gaap_model = build_model(graph.ndata["feature"].shape[1], args).to(device)
        optimizer = torch.optim.Adam(gaap_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.gnn_n_layers)
        train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)

        best_score = -1
        best_metrics = None
        best_epoch = 0
        final_metrics = None
        final_epoch = 0
        for epoch in range(args.epoch):
            loss = train_epoch(gaap_model, optimizer, graph, sampler, train_loader)
            if (epoch + 1) % args.test_epoch != 0 and epoch + 1 != args.epoch:
                print(f"Epoch {epoch + 1}: loss={loss:.4f}")
                continue
            val_metrics = evaluate(gaap_model, graph, sampler, val_idx, args.eval_batch_size)
            test_metrics = evaluate(gaap_model, graph, sampler, test_idx, args.eval_batch_size)
            final_metrics = test_metrics
            final_epoch = epoch + 1
            score = val_metrics["macro_f1"]
            print(
                f"Epoch {epoch + 1}: loss={loss:.4f} "
                f"val_auc={val_metrics['auroc']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
                f"test_auc={test_metrics['auroc']:.4f} test_precision={test_metrics['precision']:.4f} "
                f"test_recall={test_metrics['recall']:.4f} test_f1={test_metrics['macro_f1']:.4f}"
            )
            if score >= best_score:
                best_score = score
                best_metrics = test_metrics
                best_epoch = epoch + 1

        print("[test best by train-copy val_f1]")
        print(f"best_epoch: {best_epoch}")
        for key in ["precision", "recall", "macro_f1", "auroc"]:
            print(f"{key}: {best_metrics[key]:.4f}")
        print("[test final]")
        print(f"final_epoch: {final_epoch}")
        for key in ["precision", "recall", "macro_f1", "auroc"]:
            print(f"{key}: {final_metrics[key]:.4f}")
        print(f"Total time: {time.time() - start:.1f}s")
    finally:
        if logger is not None:
            logger.flush()
            sys.stdout = old_stdout
            logger.close()


if __name__ == "__main__":
    main()
