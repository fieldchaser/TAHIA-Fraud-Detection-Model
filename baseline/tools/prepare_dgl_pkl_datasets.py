import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT.parent / "datasets"
DEFAULT_OUTPUT_ROOTS = [
    ROOT / "UniGAD-main" / "data" / "pkl",
    ROOT / "CARE-GNN" / "data",
    ROOT / "FRAUDRE-main" / "data",
    ROOT / "HeCo-main" / "data" / "pkl",
    ROOT / "HGMAE-master" / "data" / "pkl",
]
DEFAULT_DATASETS = ["amazon", "yelpchi"]
SOURCE_NAMES = {
    "yelpchi": "yelp",
}


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_dgl_graph(path):
    graphs, _ = load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError(f"{path} contains {len(graphs)} graphs; this converter expects one graph.")
    graph = graphs[0]
    if not graph.is_homogeneous:
        graph = dgl.to_homogeneous(graph, ndata=["feature", "label"])
    return graph


def graph_to_adjlists(graph):
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = graph.remove_self_loop().add_self_loop()
    src, dst = graph.edges()
    src = tensor_to_numpy(src).astype(np.int64)
    dst = tensor_to_numpy(dst).astype(np.int64)
    adjlists = defaultdict(set)
    for s, d in zip(src, dst):
        adjlists[int(s)].add(int(d))
    for node in range(graph.num_nodes()):
        adjlists[node].add(node)
    return dict(adjlists)


def labels_from_graph(graph):
    labels = tensor_to_numpy(graph.ndata["label"])
    if labels.ndim > 1:
        labels = labels.argmax(axis=1)
    labels = labels.reshape(-1).astype(np.int64)
    labels[(labels == 2) | (labels == 3)] = 0
    return labels


def features_from_graph(graph):
    features = tensor_to_numpy(graph.ndata["feature"]).astype(np.float32)
    return sp.csr_matrix(features)


def stratified_one_to_nine_split(labels):
    node_ids = np.arange(labels.shape[0], dtype=np.int64)
    train_idx, test_idx = train_test_split(
        node_ids,
        stratify=labels,
        train_size=0.1,
        random_state=2,
        shuffle=True,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return {
        "train_idx": train_idx,
        "val_idx": train_idx.copy(),
        "test_idx": test_idx,
    }


def convert_one(name, source_root, output_roots):
    source_name = SOURCE_NAMES.get(name, name)
    graph = load_dgl_graph(source_root / source_name)
    features = features_from_graph(graph)
    labels = labels_from_graph(graph)
    adjlists = graph_to_adjlists(graph)
    label_parts = [np.column_stack([np.arange(graph.num_nodes(), dtype=np.int64), labels])]
    splits = stratified_one_to_nine_split(labels)

    for root in output_roots:
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "node_features.pkl").open("wb") as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (out_dir / "labels.pkl").open("wb") as handle:
            pickle.dump(label_parts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (out_dir / f"{name}_adjlists.pickle").open("wb") as handle:
            pickle.dump(adjlists, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if all(value is not None for value in splits.values()):
            np.savez(out_dir / "splits.npz", **splits)

    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(
        f"{name}: source={source_name}, nodes={graph.num_nodes()}, edges={graph.num_edges()}, "
        f"feature_dim={features.shape[1]}, labels={counts}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert downloaded DGL anomaly datasets into local pkl adapters.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--source_root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output_roots", nargs="+", default=[str(path) for path in DEFAULT_OUTPUT_ROOTS])
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_roots = [Path(path) for path in args.output_roots]
    for name in args.datasets:
        convert_one(name, source_root, output_roots)


if __name__ == "__main__":
    main()
