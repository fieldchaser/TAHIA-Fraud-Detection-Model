#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def to_numpy_array(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if sp is not None and sp.issparse(value):
        return value.toarray()
    return np.asarray(value)


def iter_neighbors(neighbor):
    for src, dsts in neighbor.items():
        src = int(src)
        for dst in dsts:
            yield src, int(dst)


def write_summary(path, features, labels, neighbor):
    label_values, label_counts = np.unique(labels, return_counts=True)
    raw_edges = 0
    self_loops = 0
    undirected_nonself = set()
    for src, dst in iter_neighbors(neighbor):
        raw_edges += 1
        if src == dst:
            self_loops += 1
        else:
            undirected_nonself.add((src, dst) if src < dst else (dst, src))

    rows = [
        ("num_nodes", features.shape[0]),
        ("feature_dim", features.shape[1] if features.ndim > 1 else 1),
        ("num_labels", labels.shape[0]),
        ("neighbor_dict_len", len(neighbor)),
        ("raw_directed_edges", raw_edges),
        ("raw_self_loops", self_loops),
        ("unique_undirected_nonself_edges", len(undirected_nonself)),
        ("dgl_total_edges_after_add_self_loop", len(undirected_nonself) * 2 + features.shape[0]),
    ]
    for label, count in zip(label_values, label_counts):
        rows.append((f"label_{int(label)}_count", int(count)))
        rows.append((f"label_{int(label)}_rate", float(count / labels.shape[0])))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def write_label_counts(path, labels):
    label_values, label_counts = np.unique(labels, return_counts=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count", "rate"])
        for label, count in zip(label_values, label_counts):
            writer.writerow([int(label), int(count), float(count / labels.shape[0])])


def write_nodes(path, features, labels):
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    header = ["node_id", "label"] + [f"feature_{idx}" for idx in range(features.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for node_id in range(features.shape[0]):
            writer.writerow([node_id, int(labels[node_id]), *features[node_id].tolist()])


def write_directed_edges(path, neighbor):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        for src, dst in iter_neighbors(neighbor):
            writer.writerow([src, dst])


def write_undirected_edges(path, neighbor):
    edges = set()
    for src, dst in iter_neighbors(neighbor):
        if src == dst:
            continue
        edges.add((src, dst) if src < dst else (dst, src))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target"])
        for src, dst in sorted(edges):
            writer.writerow([src, dst])


def write_neighbor_stats(path, neighbor, num_nodes):
    out_degree = np.zeros(num_nodes, dtype=np.int64)
    has_self_loop = np.zeros(num_nodes, dtype=np.int64)
    for src, dst in iter_neighbors(neighbor):
        out_degree[src] += 1
        if src == dst:
            has_self_loop[src] = 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "out_degree", "has_self_loop"])
        for node_id in range(num_nodes):
            writer.writerow([node_id, int(out_degree[node_id]), int(has_self_loop[node_id])])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CARE-GNN xinyongka credit.pkl into readable CSV files."
    )
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "data" / "xinyongka" / "credit.pkl",
        help="Path to credit.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "xinyongka" / "csv_export",
        help="Directory for exported CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.input.open("rb") as handle:
        data = pickle.load(handle)

    expected_fields = {"features", "labels", "neighbor"}
    missing = expected_fields.difference(data)
    if missing:
        raise KeyError(f"Missing expected fields: {sorted(missing)}")

    features = to_numpy_array(data["features"])
    labels = to_numpy_array(data["labels"]).astype(np.int64)
    neighbor = data["neighbor"]
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"features rows ({features.shape[0]}) != labels ({labels.shape[0]})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "summary.csv", features, labels, neighbor)
    write_label_counts(args.output_dir / "label_counts.csv", labels)
    write_nodes(args.output_dir / "nodes.csv", features, labels)
    write_directed_edges(args.output_dir / "edges_directed.csv", neighbor)
    write_undirected_edges(args.output_dir / "edges_undirected_unique.csv", neighbor)
    write_neighbor_stats(args.output_dir / "neighbor_stats.csv", neighbor, features.shape[0])

    print(f"Exported CSV files to: {args.output_dir}")


if __name__ == "__main__":
    main()
