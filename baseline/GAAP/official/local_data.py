import pickle
import random
import sys
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "UniGAD-main" / "data" / "pkl"
DEFAULT_CREDIT_PATH = PROJECT_ROOT / "CARE-GNN" / "data" / "xinyongka" / "credit.pkl"
CREDIT_ALIASES = {"xinyongka"}
TAHIA_CREDIT_ALIASES = {"credit"}


def install_scipy_sparse_pickle_aliases():
    try:
        import scipy.sparse._csr as csr_module
    except ImportError:
        import scipy.sparse.csr as csr_module
    sys.modules.setdefault("scipy.sparse.csr", csr_module)
    sys.modules.setdefault("scipy.sparse._csr", csr_module)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def row_normalize(features):
    rowsum = np.asarray(features.sum(1)).flatten().astype(np.float32)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    inv[rowsum != 0] = np.power(rowsum[rowsum != 0], -1)
    return sp.diags(inv).dot(features).astype(np.float32)


def minmax_normalize(features):
    min_v = features.min(axis=0, keepdims=True)
    max_v = features.max(axis=0, keepdims=True)
    denom = np.maximum(max_v - min_v, 1e-12)
    return (features - min_v) / denom


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


def care_fraudre_split(labeled_ids, labeled_y):
    train_idx, test_idx = train_test_split(
        labeled_ids,
        stratify=labeled_y,
        train_size=0.1,
        random_state=2,
        shuffle=True,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return train_idx, train_idx.copy(), test_idx


def load_split_file(path, fallback_ids, fallback_y):
    split_path = Path(path) / "splits.npz"
    if not split_path.exists():
        return care_fraudre_split(fallback_ids, fallback_y)
    splits = np.load(split_path)
    train_idx = np.asarray(splits["train_idx"], dtype=np.int64)
    val_idx = np.asarray(splits["val_idx"], dtype=np.int64)
    test_idx = np.asarray(splits["test_idx"], dtype=np.int64)
    return train_idx, val_idx, test_idx


def resolve_credit_path(data_root):
    root = Path(data_root)
    if root.is_file():
        return root
    if (root / "credit.pkl").exists():
        return root / "credit.pkl"
    return DEFAULT_CREDIT_PATH


def load_credit_parts(data_root):
    with resolve_credit_path(data_root).open("rb") as handle:
        data = pickle.load(handle)
    features = data["features"]
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = features.astype(np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    labels[(labels == 2) | (labels == 3)] = 0
    labeled_ids = np.arange(len(labels), dtype=np.int64)
    train_idx, val_idx, test_idx = care_fraudre_split(labeled_ids, labels)
    return features, labels, data["neighbor"], train_idx, val_idx, test_idx


def load_tahia_credit_parts():
    path = REPO_ROOT / "mycode" / "TAHIA" / "data" / "credit"
    with (path / "node_features.pkl").open("rb") as handle:
        install_scipy_sparse_pickle_aliases()
        features = pickle.load(handle)
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = features.astype(np.float32)
    num_nodes = features.shape[0]

    with (path / "labels.pkl").open("rb") as handle:
        label_parts = pickle.load(handle)
    labeled_y = np.asarray(label_parts[0], dtype=np.int64)
    labeled_y = (labeled_y == 1).astype(np.int64)
    labeled_ids = np.arange(len(labeled_y), dtype=np.int64)
    labels = np.zeros(num_nodes, dtype=np.int64)
    labels[labeled_ids] = labeled_y
    train_idx, val_idx, test_idx = care_fraudre_split(labeled_ids, labeled_y)

    with (path / "edges.pkl").open("rb") as handle:
        edges = pickle.load(handle)
    adj = sum(edges.values()).tocsr()
    adj = adj.maximum(adj.T)
    rows, cols = adj.nonzero()
    adjlists = {}
    for src, dst in zip(rows, cols):
        adjlists.setdefault(int(src), set()).add(int(dst))
    return features, labels, adjlists, train_idx, val_idx, test_idx


def adjlists_to_edges(adjlists):
    rows = []
    cols = []
    for src, neighbors in adjlists.items():
        src = int(src)
        for dst in neighbors:
            rows.append(src)
            cols.append(int(dst))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def index_to_mask(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = True
    return mask


def load_pkl_graph(dataset, data_root=DEFAULT_DATA_ROOT, norm_type="norm01", to_bidirected=True):
    if dataset in CREDIT_ALIASES:
        features, labels, adjlists, train_idx, val_idx, test_idx = load_credit_parts(data_root)
    elif dataset in TAHIA_CREDIT_ALIASES:
        features, labels, adjlists, train_idx, val_idx, test_idx = load_tahia_credit_parts()
    else:
        path = Path(data_root) / dataset
        with (path / "node_features.pkl").open("rb") as handle:
            install_scipy_sparse_pickle_aliases()
            features = pickle.load(handle)
        if not sp.issparse(features):
            features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
        features = features.astype(np.float32)
        with (path / "labels.pkl").open("rb") as handle:
            label_parts = pickle.load(handle)
        labels, labeled_ids, labeled_y = labels_from_parts(label_parts, features.shape[0])
        train_idx, val_idx, test_idx = load_split_file(path, labeled_ids, labeled_y)

    if norm_type == "row":
        features = row_normalize(features).toarray()
    else:
        features = minmax_normalize(features.toarray())
    num_nodes = features.shape[0]

    if dataset not in CREDIT_ALIASES and dataset not in TAHIA_CREDIT_ALIASES:
        with (path / f"{dataset}_adjlists.pickle").open("rb") as handle:
            adjlists = pickle.load(handle)
    src, dst = adjlists_to_edges(adjlists)
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    graph = graph.remove_self_loop().add_self_loop()
    graph.ndata["feature"] = torch.from_numpy(features).float().contiguous()
    graph.ndata["label"] = torch.from_numpy(labels).long().contiguous()
    graph.ndata["train_mask"] = index_to_mask(train_idx, num_nodes)
    graph.ndata["val_mask"] = index_to_mask(val_idx, num_nodes)
    graph.ndata["test_mask"] = index_to_mask(test_idx, num_nodes)
    graph.ndata["train_masks"] = graph.ndata["train_mask"].unsqueeze(1)
    graph.ndata["val_masks"] = graph.ndata["val_mask"].unsqueeze(1)
    graph.ndata["test_masks"] = graph.ndata["test_mask"].unsqueeze(1)
    return graph.long(), torch.from_numpy(train_idx).long(), torch.from_numpy(val_idx).long(), torch.from_numpy(test_idx).long()
