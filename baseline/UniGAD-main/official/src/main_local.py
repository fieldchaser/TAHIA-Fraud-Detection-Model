import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


try:
    import line_profiler  # noqa: F401
except ModuleNotFoundError:
    line_profiler = types.ModuleType("line_profiler")
    line_profiler.profile = lambda fn: fn
    sys.modules["line_profiler"] = line_profiler

import e2e_models
from e2e_models import UnifyMLPDetector
from local_data import DEFAULT_DATA_ROOT, load_local_dataset, set_local_seed
from pretrain_models import GraphMAE
from utils import get_current_lr


def local_pareto_fn(w_list, c_list, model, num_tasks, loss_list):
    del c_list, model, num_tasks, loss_list
    return w_list


e2e_models.pareto_fn = local_pareto_fn


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
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parents[2] / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    return TeeLogger(log_dir / f"{args.data}.log")


def build_args():
    parser = argparse.ArgumentParser(description="UniGAD local adapter for ACM/DBLP pkl node anomaly datasets")
    parser.add_argument("--data", choices=["acm", "dblp", "amazon", "yelpchi", "xinyongka", "credit"], default="acm")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch_pretrain", type=int, default=50)
    parser.add_argument("--epoch_ft", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_ft", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--l2_ft", type=float, default=0.0)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--num_layer_pretrain", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act", default="leakyrelu")
    parser.add_argument("--act_ft", default="ReLU")
    parser.add_argument("--norm", default="")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--replace_ratio", type=float, default=0.0)
    parser.add_argument("--kernels", default="gcn")
    parser.add_argument("--khop", type=int, default=0)
    parser.add_argument("--cross_mode", default="n2n")
    parser.add_argument("--stitch_mlp_layers", type=int, default=1)
    parser.add_argument("--final_mlp_layers", type=int, default=2)
    parser.add_argument("--node_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--graph_loss_weight", type=float, default=1.0)
    parser.add_argument("--metric", default="MacroF1")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--decay_step", type=int, default=100)
    parser.add_argument("--decay_rate", type=float, default=1.0)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--no_log", action="store_true")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu on this machine.")
    return torch.device(device_name)


def split_kernel(kernel):
    if "-" in kernel:
        return kernel.split("-", 1)
    if kernel == "bwgnn":
        return kernel, "gcn"
    return kernel, kernel


def pretrain(model, train_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    for epoch in tqdm(range(args.epoch_pretrain), desc="pretrain"):
        model.train()
        losses = []
        loss_dict = {}
        for batched_graph in train_dataloader:
            batched_graph = batched_graph.to(args.device)
            loss, loss_dict = model(batched_graph, batched_graph.ndata["feature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        loss_dict["lr"] = get_current_lr(optimizer)
        print(f"Pretrain epoch {epoch + 1}: loss={np.mean(losses):.4f} {loss_dict}")
    return model


@torch.no_grad()
def evaluate_node_model(detector, graph, labels, scen, device):
    detector.model.eval()
    graph = graph.to(device)
    empty_sp = None
    logits = detector.model(graph, graph.ndata["feature"], empty_sp, scen=scen)["n"]
    probs = logits.softmax(1)[:, 1].detach().cpu().numpy()
    preds = logits.argmax(1).detach().cpu().numpy()
    mask = detector.model.mask_dicts["n"][scen]
    selected_labels = labels[mask].detach().cpu().numpy()
    return {
        "precision": precision_score(selected_labels, preds, average="macro", zero_division=0),
        "recall": recall_score(selected_labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(selected_labels, preds, average="macro", zero_division=0),
        "auroc": roc_auc_score(selected_labels, probs),
    }


def main():
    args = build_args()
    logger = setup_logging(args)
    old_stdout = sys.stdout
    if logger is not None:
        sys.stdout = logger
    try:
        args.device = resolve_device(args.device)
        set_local_seed(args.seed)
        print("Model info:")
        print(json.dumps({k: str(v) if k == "device" else v for k, v in vars(args).items()}, indent=2))

        start = time.time()
        dataset = load_local_dataset(args.data, args.data_root)
        print(
            f"Loaded {args.data}: nodes={dataset.graph.num_nodes()}, "
            f"train={len(dataset.train_mask_node_cur)}, val={len(dataset.val_mask_node_cur)}, "
            f"test={len(dataset.test_mask_node_cur)}"
        )

        encoder_type, decoder_type = split_kernel(args.kernels.split(",")[0])
        pretrain_model = GraphMAE(
            in_dim=dataset.in_dim,
            hid_dim=args.hid_dim,
            num_layer=args.num_layer_pretrain,
            drop_ratio=args.dropout,
            act=args.act,
            norm=args.norm,
            residual=args.residual,
            mask_ratio=args.mask_ratio,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            replace_ratio=args.replace_ratio,
        ).to(args.device)
        pretrain_model = pretrain(pretrain_model, dataset.get_pretrain_dataloaders(args.batch_size), args)

        train_dataloader, val_dataloader, test_dataloader = dataset.get_graph_and_sp_dataloaders(args.batch_size)
        detector = UnifyMLPDetector(
            pretrain_model,
            dataset,
            (train_dataloader, val_dataloader, test_dataloader),
            cross_mode=args.cross_mode,
            args=args,
        )
        score_test = detector.train()
        val_metrics = evaluate_node_model(detector, dataset.graph, dataset.graph.ndata["node_label"], "val", args.device)
        test_metrics = evaluate_node_model(
            detector,
            dataset.graph,
            dataset.graph.ndata["node_label"],
            "test",
            args.device,
        )

        print("Official score dictionary:")
        print(score_test)
        print("[test best by train-copy val_f1]")
        for key in ["precision", "recall", "macro_f1", "auroc"]:
            print(f"{key}: {test_metrics[key]:.4f}")
        print(f"Total time: {time.time() - start:.1f}s")
    finally:
        if logger is not None:
            logger.flush()
            sys.stdout = old_stdout
            logger.close()


if __name__ == "__main__":
    main()
