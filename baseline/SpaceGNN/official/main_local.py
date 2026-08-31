import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from local_data import DEFAULT_DATA_ROOT, load_pkl_graph, set_seed
import model


warnings.filterwarnings("ignore")


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


def build_args():
    parser = argparse.ArgumentParser(description="SpaceGNN local adapter for ACM/DBLP pkl datasets")
    parser.add_argument("--data", choices=["acm", "dblp", "amazon", "yelpchi", "xinyongka", "credit"], default="acm")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--seed", type=int, default=1019)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--nepoch", type=int, default=25)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--drop_rate", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--test_epoch", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--stdneg", type=float, default=0.015)
    parser.add_argument("--stdpos", type=float, default=0.015)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--no_log", action="store_true")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu on this machine.")
    return torch.device(device_name)


def train_epoch(spacegnn, optimizer, graph, sampler, data_loader, alpha, beta):
    spacegnn.train()
    total_loss = 0.0
    batches = 0
    for index in data_loader:
        sample_result = sampler.sample_blocks(graph, index)

        if isinstance(sample_result, tuple):
            _, _, blocks = sample_result
        else:
            blocks = sample_result
        probs1, probs2, probs3 = spacegnn(blocks)
        target = blocks[-1].dstdata["label"]
        probs = (1 - beta) * ((1 - alpha) * probs1 + alpha * probs2) + beta * probs3
        loss = F.nll_loss(probs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def evaluate(spacegnn, graph, sampler, data_loader, alpha, beta):
    spacegnn.eval()
    probs1_list = []
    probs2_list = []
    probs3_list = []
    targets_list = []

    with torch.no_grad():
        for index in data_loader:
            sample_result = sampler.sample_blocks(graph, index)

            if isinstance(sample_result, tuple):
                _, _, blocks = sample_result
            else:
                blocks = sample_result
            targets_list.append(blocks[-1].dstdata["label"])
            probs1, probs2, probs3 = spacegnn(blocks)
            probs1_list.append(probs1.detach())
            probs2_list.append(probs2.detach())
            probs3_list.append(probs3.detach())

    targets = torch.cat(targets_list, dim=0).cpu().numpy()
    probs1 = torch.cat(probs1_list, dim=0).exp().cpu().numpy()
    probs2 = torch.cat(probs2_list, dim=0).exp().cpu().numpy()
    probs3 = torch.cat(probs3_list, dim=0).exp().cpu().numpy()
    probs_all = (1 - beta) * ((1 - alpha) * probs1 + alpha * probs2) + beta * probs3
    probs = probs_all[:, 1]
    preds = probs_all.argmax(axis=1)

    return {
        "auroc": roc_auc_score(targets, probs),
        "precision": precision_score(targets, preds, average="macro", zero_division=0),
        "recall": recall_score(targets, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
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
        graph, train_index, val_index, test_index = load_pkl_graph(args.data, args.data_root, args.layer_num)
        graph = graph.to(device)
        train_index = train_index.to(device)
        val_index = val_index.to(device)
        test_index = test_index.to(device)
        print(
            f"Loaded {args.data}: nodes={graph.num_nodes()}, train={len(train_index)}, "
            f"val={len(val_index)}, test={len(test_index)}"
        )

        cneg = torch.FloatTensor(args.layer_num).normal_(-0.1, args.stdneg).to(device)
        cpos = torch.FloatTensor(args.layer_num).normal_(0.1, args.stdpos).to(device)
        spacegnn = model.SpaceGNN(
            graph.ndata["feature"].shape[1],
            args.hid_dim,
            2,
            args.layer_num,
            args.drop_rate,
            cneg,
            cpos,
        ).to(device)
        optimizer = torch.optim.Adam(spacegnn.parameters(), lr=args.lr)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

        train_loader = DataLoader(train_index, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_index, batch_size=min(100000, len(val_index)), shuffle=False, drop_last=False)
        test_loader = DataLoader(test_index, batch_size=min(100000, len(test_index)), shuffle=False, drop_last=False)

        best_score = -1
        best_metrics = None
        best_epoch = 0
        final_metrics = None
        final_epoch = 0
        for epoch in range(args.nepoch):
            loss = train_epoch(spacegnn, optimizer, graph, sampler, train_loader, args.alpha, args.beta)
            if (epoch + 1) % args.test_epoch != 0 and epoch + 1 != args.nepoch:
                print(f"Epoch {epoch + 1}: loss={loss:.4f}")
                continue

            val_metrics = evaluate(spacegnn, graph, sampler, val_loader, args.alpha, args.beta)
            test_metrics = evaluate(spacegnn, graph, sampler, test_loader, args.alpha, args.beta)
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

        elapsed = time.time() - start
        print("[test best by train-copy val_f1]")
        print(f"best_epoch: {best_epoch}")
        for key in ["precision", "recall", "macro_f1", "auroc"]:
            print(f"{key}: {best_metrics[key]:.4f}")
        print("[test final]")
        print(f"final_epoch: {final_epoch}")
        for key in ["precision", "recall", "macro_f1", "auroc"]:
            print(f"{key}: {final_metrics[key]:.4f}")
        print(f"Total time: {elapsed:.1f}s")
    finally:
        if logger is not None:
            logger.flush()
            sys.stdout = old_stdout
            logger.close()


if __name__ == "__main__":
    main()
