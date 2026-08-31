import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData
import dgl
data_folder = "data/"


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return features.todense()
    return features

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def _adjlists_to_sparse(adjlists, num_nodes):
    rows = []
    cols = []
    for src, neighbors in adjlists.items():
        src = int(src)
        for dst in neighbors:
            rows.append(src)
            cols.append(int(dst))
    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()


def _binary_labels_from_parts(label_parts, num_nodes):
    labels = np.zeros(num_nodes, dtype=np.int64)
    labeled_ids = []
    labeled_y = []
    for part in label_parts:
        part = np.asarray(part, dtype=np.int64)
        ids = part[:, 0]
        y = part[:, 1].copy()
        y[np.where(y == 2)] = 0
        y[np.where(y == 3)] = 0
        labels[ids] = y
        labeled_ids.extend(ids.tolist())
        labeled_y.extend(y.tolist())
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    labeled_y = np.asarray(labeled_y, dtype=np.int64)
    order = np.argsort(labeled_ids)
    return labels, labeled_ids[order], labeled_y[order]


def _care_fraudre_split(labeled_ids, labeled_y):
    idx_train, idx_test = train_test_split(
        labeled_ids,
        stratify=labeled_y,
        train_size=0.1,
        random_state=2,
        shuffle=True,
    )
    train = th.LongTensor(np.asarray(idx_train, dtype=np.int64))
    test = th.LongTensor(np.asarray(idx_test, dtype=np.int64))
    return [train], [train], [test]


def _load_split_file(path, fallback_ids, fallback_y):
    split_path = Path(path) / "splits.npz"
    if not split_path.exists():
        return _care_fraudre_split(fallback_ids, fallback_y)
    splits = np.load(split_path)
    train = th.LongTensor(np.asarray(splits["train_idx"], dtype=np.int64))
    val = th.LongTensor(np.asarray(splits["val_idx"], dtype=np.int64))
    test = th.LongTensor(np.asarray(splits["test_idx"], dtype=np.int64))
    return [train], [val], [test]


def _resolve_pkl_path(dataset):
    local_path = Path(__file__).resolve().parents[2] / "data" / "pkl" / dataset
    shared_path = Path(__file__).resolve().parents[3] / "UniGAD-main" / "data" / "pkl" / dataset
    if local_path.exists():
        return local_path
    return shared_path


def load_pkl_anomaly(dataset):
    path = _resolve_pkl_path(dataset)
    with (path / "node_features.pkl").open("rb") as f:
        feat_data = pickle.load(f)
    if not sp.issparse(feat_data):
        feat_data = sp.csr_matrix(np.asarray(feat_data, dtype=np.float32))
    feat_data = feat_data.astype(np.float32)
    num_nodes = feat_data.shape[0]

    with (path / "labels.pkl").open("rb") as f:
        label_parts = pickle.load(f)
    labels, labeled_ids, labeled_y = _binary_labels_from_parts(label_parts, num_nodes)

    with (path / f"{dataset}_adjlists.pickle").open("rb") as f:
        adjlists = pickle.load(f)
    adj = _adjlists_to_sparse(adjlists, num_nodes)
    nei = []
    for node in range(num_nodes):
        neighbors = np.asarray(sorted(adjlists.get(node, {node})), dtype=np.int64)
        if neighbors.size == 0:
            neighbors = np.asarray([node], dtype=np.int64)
        nei.append(neighbors)

    label = th.FloatTensor(encode_onehot(labels))
    feat = th.FloatTensor(preprocess_features(feat_data))
    mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    pos = sparse_mx_to_torch_sparse_tensor(adj)
    train, val, test = _load_split_file(path, labeled_ids, labeled_y)
    return [nei], [feat, feat], [mp], pos, label, train, val, test


def process_data_in_pyg(neigs):
    d = defaultdict(dict)
    metapaths = []
    for mp_i, nei1 in enumerate(neigs):
        dst_array_concat = np.concatenate(nei1)
        src_array_concat = []
        for src_id, dst_array in enumerate(nei1):
            src_array_concat.extend([src_id] * len(dst_array))
        src_array_concat = np.array(src_array_concat)
        src_name = f"target"
        dst_name = f"dst_{mp_i}"
        relation = f"relation_{mp_i}"
        d[(src_name, relation + "-->", dst_name)]["edge_index"] = th.LongTensor(np.vstack([src_array_concat, dst_array_concat]))
        metapaths.append((src_name, relation + "-->", dst_name))
        d[(dst_name, "<--" + relation, src_name)]["edge_index"] = th.LongTensor(np.vstack([dst_array_concat, src_array_concat]))
        metapaths.append((dst_name, "<--" + relation, src_name))
    g = HeteroData(d, allow_zero_in_degree=True)
    return g, metapaths


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = data_folder + "acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = data_folder + "dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_credit(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = data_folder + "credit/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = data_folder + "freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label[np.where(label == 4)] = 0
    label = encode_onehot(label)
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test

# def load_credit(ratio, type_num):
#     # The order of node types: 0 p 1 a 2 s
#     ratio = [10, 20, 30]
#     path = data_folder + "credit/"
#     # path = "../data/credit/"
#     label = np.load(path + "labels.npy").astype('int32')
#     label = encode_onehot(label)
#     nei_c = np.load(path + "nei_c.npy", allow_pickle=True)
#     nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
#     # feat_c = sp.load_npz(path + "c_feat.npz")
#     feat_c = np.load(path + "c_feat.npy")
#     feat_s = np.load(path + "s_feat.npy")
#     # feat_s = sp.eye(type_num[1])
#     # feat_s = sp.eye(type_num[2])
#     cc = sp.load_npz(path + "cc.npz")
#     cs = sp.load_npz(path + "cs.npz")
#     pos = sp.load_npz(path + "pos.npz")
#     # pos = th.tensor(pos)
#     train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
#     test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
#     val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

#     label = th.FloatTensor(label)
#     nei_c = [th.LongTensor(i) for i in nei_c]
#     nei_s = [th.LongTensor(i) for i in nei_s]
#     feat_c = th.FloatTensor(preprocess_features1(feat_c))
#     feat_s = th.FloatTensor(preprocess_features1(feat_s))
#     cc = sparse_mx_to_torch_sparse_tensor(normalize_adj(cc))
#     cs = sparse_mx_to_torch_sparse_tensor(normalize_adj(cs))
#     train = [th.LongTensor(i) for i in train]
#     val = [th.LongTensor(i) for i in val]
#     test = [th.LongTensor(i) for i in test]
#     feat = th.concat([feat_c, feat_s])
#     return [nei_c, nei_s], [feat, feat], [cc, cs], pos, label, train, val, test

def load_data(dataset, ratio, type_num):
    if _resolve_pkl_path(dataset).exists():
        data = load_pkl_anomaly(dataset)
    elif dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "credit":
        data = load_credit(ratio, type_num)
    # elif dataset == "credit":
    #     data = load_credit(ratio, type_num)
    # elif dataset == "credit":
    #     data = load_freebase(ratio, type_num)
    g, metapaths = process_data_in_pyg(data[0])
    return data, g, metapaths
