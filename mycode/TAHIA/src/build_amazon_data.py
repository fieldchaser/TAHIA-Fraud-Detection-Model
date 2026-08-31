import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def install_scipy_sparse_pickle_aliases():
    try:
        import scipy.sparse._csr as csr_module
    except ImportError:
        import scipy.sparse.csr as csr_module
    import sys

    sys.modules.setdefault("scipy.sparse.csr", csr_module)
    sys.modules.setdefault("scipy.sparse._csr", csr_module)


def adjlists_to_sparse(adjlists, num_nodes):
    rows = []
    cols = []
    for src, neighbors in adjlists.items():
        src = int(src)
        for dst in neighbors:
            rows.append(src)
            cols.append(int(dst))
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "baseline" / "UniGAD-main" / "data" / "pkl" / "amazon"
    target = Path(__file__).resolve().parents[1] / "data" / "amazon"
    target.mkdir(parents=True, exist_ok=True)

    install_scipy_sparse_pickle_aliases()
    with (source / "node_features.pkl").open("rb") as handle:
        features = pickle.load(handle)
    if not sp.issparse(features):
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
    features = features.astype(np.float32).tocsr()
    num_nodes, feat_dim = features.shape

    with (source / "labels.pkl").open("rb") as handle:
        label_parts = pickle.load(handle)
    label_arr = np.asarray(label_parts[0], dtype=np.int64)
    labels = label_arr[:, 1].astype(np.int64)

    with (source / "amazon_adjlists.pickle").open("rb") as handle:
        adjlists = pickle.load(handle)
    adj = adjlists_to_sparse(adjlists, num_nodes)
    adj = adj.maximum(adj.T).astype(np.float32).tocsr()

    meta_data = {
        "t_info": {"u": {"ind": range(0, num_nodes), "cnt": num_nodes}},
        "rel2id": {"u-u": 0},
        "id2rel": {0: "u-u"},
        "n_class": 2,
        "n_feat": feat_dim,
        "relations": ["u-u"],
        "types": ["u"],
        "undirected_relations": {"u-u"},
        "r_info": {"u-u": (0, num_nodes, 0, num_nodes)},
    }

    # Single-node-type Amazon has no heterogeneous metapath embedding in TAHIA format.
    # Reuse the shared node feature matrix as the semantic embedding to keep the
    # existing semantic graph generation path unchanged.
    mp_emb = features.toarray().astype(np.float32)

    outputs = {
        "node_features.pkl": features,
        "edges.pkl": {"u-u": adj},
        "labels.pkl": [labels],
        "meta_data.pkl": meta_data,
        "uu_emb.pkl": mp_emb,
    }
    for name, obj in outputs.items():
        with (target / name).open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(
        {
            "target": str(target),
            "feature_shape": features.shape,
            "edge_nnz": int(adj.nnz),
            "label_counts": {int(k): int(v) for k, v in counts.items()},
            "mp_emb_shape": mp_emb.shape,
        }
    )


if __name__ == "__main__":
    main()
