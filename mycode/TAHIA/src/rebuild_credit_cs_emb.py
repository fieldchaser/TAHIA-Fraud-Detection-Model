import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "credit"


def _row_normalize(mat):
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    return sp.diags(inv).dot(mat)


def _l2_normalize(arr):
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return arr / norm


def main():
    with open(DATA_DIR / "node_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(DATA_DIR / "edges.pkl", "rb") as f:
        edges = pickle.load(f)

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D node features, got shape {features.shape}")

    adj = sum(edges.values())
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.tocsr()

    if adj.shape[0] != features.shape[0] or adj.shape[1] != features.shape[0]:
        raise ValueError(
            f"Adjacency shape {adj.shape} does not match feature rows {features.shape[0]}"
        )

    self_part = _l2_normalize(features)
    neigh_part = _l2_normalize(_row_normalize(adj).dot(features).astype(np.float32))
    emb = np.concatenate([self_part, neigh_part], axis=1).astype(np.float64)

    out_path = DATA_DIR / "cs_emb.pkl"
    if out_path.exists():
        backup_path = DATA_DIR / "cs_emb.pkl.bak"
        if not backup_path.exists():
            shutil.copy2(out_path, backup_path)

    tmp_path = DATA_DIR / "cs_emb.pkl.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)

    print(f"Wrote {out_path} with shape {emb.shape}")


if __name__ == "__main__":
    main()
