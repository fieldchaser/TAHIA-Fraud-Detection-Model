import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

try:
    import pickle5 as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "UniGAD-main" / "data" / "pkl"
DEFAULT_CREDIT_PATH = PROJECT_ROOT / "CARE-GNN" / "data" / "xinyongka" / "credit.pkl"
DEFAULT_TAHIA_DATA_ROOT = PROJECT_ROOT.parent / "mycode" / "TAHIA" / "data"


def install_scipy_sparse_pickle_aliases():
    try:
        import scipy.sparse._csr as csr_module
    except ImportError:
        import scipy.sparse.csr as csr_module
    sys.modules.setdefault("scipy.sparse.csr", csr_module)
    sys.modules.setdefault("scipy.sparse._csr", csr_module)


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
    log_dir = Path(args.log_dir) if args.log_dir else Path(__file__).resolve().parent / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = args.data
    return TeeLogger(log_dir / f"{name}.log")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def row_normalize(features):
    features = sp.csr_matrix(features, dtype=np.float32)
    rowsum = np.asarray(features.sum(1)).flatten().astype(np.float32)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    inv[rowsum != 0] = np.power(rowsum[rowsum != 0], -1)
    return sp.diags(inv).dot(features).astype(np.float32)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj, dtype=np.float32)
    rowsum = np.asarray(adj.sum(1)).flatten().astype(np.float32)
    inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    inv_sqrt[rowsum != 0] = np.power(rowsum[rowsum != 0], -0.5)
    d_mat = sp.diags(inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).tocoo().astype(np.float32)


def sparse_to_torch(matrix, device):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def labels_from_parts(label_parts, num_nodes):
    labels = np.zeros(num_nodes, dtype=np.int64)
    labeled_ids = []
    labeled_y = []
    for part in label_parts:
        part = np.asarray(part, dtype=np.int64)
        ids = part[:, 0]
        y = part[:, 1].copy()
        y[(y == 2) | (y == 3)] = 0
        labels[ids] = y
        labeled_ids.extend(ids.tolist())
        labeled_y.extend(y.tolist())
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    labeled_y = np.asarray(labeled_y, dtype=np.int64)
    order = np.argsort(labeled_ids)
    return labels, labeled_ids[order], labeled_y[order]


def adjlists_to_adj(adjlists, num_nodes):
    rows = []
    cols = []
    for src, neighbors in adjlists.items():
        src = int(src)
        for dst in neighbors:
            dst = int(dst)
            if src == dst:
                continue
            rows.extend([src, dst])
            cols.extend([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(num_nodes, dtype=np.float32)
    return normalize_adj(adj)


def care_fraudre_split(labeled_ids, labeled_y):
    train_idx, test_idx = train_test_split(
        labeled_ids,
        stratify=labeled_y,
        train_size=0.1,
        random_state=2,
        shuffle=True,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def load_split_file(path, fallback_ids, fallback_y):
    split_path = Path(path) / "splits.npz"
    if not split_path.exists():
        return care_fraudre_split(fallback_ids, fallback_y)
    splits = np.load(split_path)
    return np.asarray(splits["train_idx"], dtype=np.int64), np.asarray(splits["test_idx"], dtype=np.int64)


def load_credit(path):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    features = data["features"]
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = row_normalize(features.astype(np.float32)).toarray()
    labels = np.asarray(data["labels"], dtype=np.int64)
    labels[(labels == 2) | (labels == 3)] = 0
    labeled_ids = np.arange(len(labels), dtype=np.int64)
    train_idx, test_idx = care_fraudre_split(labeled_ids, labels)
    adj = adjlists_to_adj(data["neighbor"], features.shape[0])
    return features, labels, train_idx, test_idx, adj


def load_tahia_credit(data_root):
    path = Path(data_root) / "credit"
    with (path / "node_features.pkl").open("rb") as handle:
        features = pickle.load(handle)
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = row_normalize(features.astype(np.float32)).toarray()
    num_nodes = features.shape[0]

    with (path / "labels.pkl").open("rb") as handle:
        label_parts = pickle.load(handle)
    if len(label_parts) == 1:
        labeled_y = np.asarray(label_parts[0], dtype=np.int64)
        labeled_ids = np.arange(len(labeled_y), dtype=np.int64)
    else:
        _, labeled_ids, labeled_y = labels_from_parts(label_parts, num_nodes)
    labeled_y = (labeled_y == 1).astype(np.int64)

    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[labeled_ids] = labeled_y
    train_idx, test_idx = care_fraudre_split(labeled_ids, labeled_y)

    with (path / "edges.pkl").open("rb") as handle:
        edges = pickle.load(handle)
    adj = sum(edges.values()).astype(np.float32)
    adj = adj.maximum(adj.T)
    adj = adj + sp.eye(num_nodes, dtype=np.float32)
    adj = normalize_adj(adj)
    return features, labels, train_idx, test_idx, adj


def load_pkl_dataset(dataset, data_root):
    if dataset == "xinyongka":
        return load_credit(DEFAULT_CREDIT_PATH)
    if dataset == "credit":
        return load_tahia_credit(DEFAULT_TAHIA_DATA_ROOT)

    path = Path(data_root) / dataset
    with (path / "node_features.pkl").open("rb") as handle:
        install_scipy_sparse_pickle_aliases()
        features = pickle.load(handle)
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = row_normalize(features.astype(np.float32)).toarray()
    num_nodes = features.shape[0]

    with (path / "labels.pkl").open("rb") as handle:
        label_parts = pickle.load(handle)
    labels, labeled_ids, labeled_y = labels_from_parts(label_parts, num_nodes)
    train_idx, test_idx = load_split_file(path, labeled_ids, labeled_y)

    with (path / f"{dataset}_adjlists.pickle").open("rb") as handle:
        adjlists = pickle.load(handle)
    adj = adjlists_to_adj(adjlists, num_nodes)
    return features, labels, train_idx, test_idx, adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = x @ self.weight
        return torch.sparse.mm(adj, support) + self.bias


class GCN(nn.Module):
    def __init__(self, in_features, hidden, num_classes, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden)
        self.gc2 = GraphConvolution(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


def best_threshold(scores, labels):
    best_f1 = -1
    best_thres = 0.5
    for thres in np.linspace(0.05, 0.95, 19):
        preds = (scores > thres).astype(np.int64)
        score = f1_score(labels, preds, average="macro", zero_division=0)
        if score >= best_f1:
            best_f1 = score
            best_thres = thres
    return best_thres


def calc_metrics(logits, labels, idx):
    probs = logits.softmax(1)[:, 1].detach().cpu().numpy()
    preds = logits.argmax(1).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    idx = np.asarray(idx, dtype=np.int64)
    return {
        "precision": precision_score(y[idx], preds[idx], average="macro", zero_division=0),
        "recall": recall_score(y[idx], preds[idx], average="macro", zero_division=0),
        "macro_f1": f1_score(y[idx], preds[idx], average="macro", zero_division=0),
        "auroc": roc_auc_score(y[idx], probs[idx]),
    }


@torch.no_grad()
def evaluate(model, features, adj, labels, train_idx, test_idx):
    model.eval()
    logits = model(features, adj)
    return {
        "val": calc_metrics(logits, labels, train_idx),
        "test": calc_metrics(logits, labels, test_idx),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Kipf-Welling two-layer GCN baseline for local anomaly datasets.")
    parser.add_argument("--data", choices=["acm", "dblp", "amazon", "yelpchi", "xinyongka", "credit"], default="acm")
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--no_log", action="store_true")
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu on this machine.")
    return torch.device(device_name)


def main():
    args = parse_args()
    logger = setup_logging(args)
    old_stdout = sys.stdout
    if logger is not None:
        sys.stdout = logger
    try:
        set_seed(args.seed)
        device = resolve_device(args.device)
        print("Model info:")
        print(json.dumps(vars(args), indent=2))
        print(f"Using device: {device}")

        start = time.time()
        features_np, labels_np, train_idx, test_idx, adj_sp = load_pkl_dataset(args.data, args.data_root)
        features = torch.from_numpy(features_np).float().to(device)
        labels = torch.from_numpy(labels_np).long().to(device)
        adj = sparse_to_torch(adj_sp, device)
        train_idx_t = torch.from_numpy(train_idx).long().to(device)

        non_self_edges = int((adj_sp.nnz - features_np.shape[0]))
        print(
            f"Loaded {args.data}: nodes={features_np.shape[0]}, adj_entries={adj_sp.nnz}, "
            f"non_self_entries={non_self_edges}, train={len(train_idx)}, test={len(test_idx)}, "
            f"feature_shape={features_np.shape}"
        )

        model = GCN(features_np.shape[1], args.hidden, 2, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_score = -1
        best_metrics = None
        best_epoch = 0
        final_metrics = None
        final_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(features, adj)
            loss = F.cross_entropy(logits[train_idx_t], labels[train_idx_t])
            loss.backward()
            optimizer.step()

            if epoch % args.test_epoch != 0 and epoch != args.epochs:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")
                continue

            metrics = evaluate(model, features, adj, labels, train_idx, test_idx)
            final_metrics = metrics["test"]
            final_epoch = epoch
            score = metrics["val"]["macro_f1"]
            print(
                f"Epoch {epoch}: loss={loss.item():.4f} "
                f"val_f1={metrics['val']['macro_f1']:.4f} "
                f"test_auc={metrics['test']['auroc']:.4f} test_precision={metrics['test']['precision']:.4f} "
                f"test_recall={metrics['test']['recall']:.4f} test_f1={metrics['test']['macro_f1']:.4f}"
            )
            if score >= best_score:
                best_score = score
                best_metrics = metrics["test"]
                best_epoch = epoch

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
