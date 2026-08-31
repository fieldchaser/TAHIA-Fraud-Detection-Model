import numpy as np
import pickle
from pathlib import Path
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def preprocess_features1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return features.todense()
    return features


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
        raw_y = part[:, 1].copy()

        # ACM / DBLP: class 1 is anomaly/fraud, all other classes are normal
        y = (raw_y == 1).astype(np.int64)

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

    id2y = dict(zip(labeled_ids.tolist(), labeled_y.tolist()))
    y_train = np.asarray([id2y[int(i)] for i in idx_train])
    y_test = np.asarray([id2y[int(i)] for i in idx_test])

    print("1:9 split check")
    print("Train:", len(idx_train), np.unique(y_train, return_counts=True))
    print("Test:", len(idx_test), np.unique(y_test, return_counts=True))

    train = th.LongTensor(np.asarray(idx_train, dtype=np.int64))
    test = th.LongTensor(np.asarray(idx_test, dtype=np.int64))

    # HeCo framework still expects val. Use train as val only for compatibility.
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

def load_credit_single_pkl(dataset="xinyongka"):
    """
    Load CARE-GNN style credit.pkl:
    {
        'features': torch.Tensor [N, F],
        'labels': np.ndarray [N],
        'neighbor': defaultdict
    }
    and convert it to HeCo input format.
    """
    path = Path(__file__).resolve().parents[2] / "data" / "pkl" / dataset
    pkl_path = path / "credit.pkl"

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    features = data["features"]
    labels = data["labels"]
    neighbor = data["neighbor"]

    if isinstance(features, th.Tensor):
        feat = features.float()
    else:
        feat = th.FloatTensor(np.asarray(features))

    label_np = np.asarray(labels).astype(np.int64)

    # binary: 1 = fraud, 0 = normal
    binary_y = (label_np == 1).astype(np.int64)

    labeled_ids = np.arange(len(binary_y), dtype=np.int64)
    labeled_y = binary_y

    # build adjacency from neighbor dict
    rows = []
    cols = []

    for src, neighs in neighbor.items():
        src = int(src)
        for dst in neighs:
            rows.append(src)
            cols.append(int(dst))

    num_nodes = feat.shape[0]
    adj = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )

    # make undirected and binarize
    adj = adj + adj.T
    adj.data[:] = 1.0

    # HeCo needs nei: neighbor indices for each node type/view.
    # For this credit dataset, we use one view.
    nei = []
    adj_csr = adj.tocsr()
    for i in range(num_nodes):
        nei.append(adj_csr[i].indices.astype(np.int64))

    # one meta-path / one graph view
    mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    pos = sparse_mx_to_torch_sparse_tensor(adj)

    label = th.FloatTensor(np.eye(2, dtype=np.float32)[binary_y])

    train, val, test = _care_fraudre_split(labeled_ids, labeled_y)

    print("Loaded CARE-GNN credit.pkl as xinyongka")
    print("Feature shape:", feat.shape)
    print("Label distribution:", np.unique(binary_y, return_counts=True))
    print("The constructed meta-paths: credit-neighbor-view")

    return [nei], [feat, feat], [mp], pos, label, train, val, test

def load_pkl_anomaly(dataset):
    path = Path(__file__).resolve().parents[2] / "data" / "pkl" / dataset
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
    # positive graph
    pos = sparse_mx_to_torch_sparse_tensor(adj)

    # meta-path graphs
    mp_list = []

    edges_path = path / "edges.pkl"

    if dataset == "acm" and edges_path.exists():
        with edges_path.open("rb") as f:
            edges = pickle.load(f)

        pap = edges["p-a"].dot(edges["a-p"]).tocsr()
        psp = edges["p-s"].dot(edges["s-p"]).tocsr()

        for m in [pap, psp]:
            m.data[:] = 1.0

        mp_list = [
            sparse_mx_to_torch_sparse_tensor(normalize_adj(pap)),
            sparse_mx_to_torch_sparse_tensor(normalize_adj(psp)),
        ]

        print("The constructed meta-paths: PAP, PSP")

    elif dataset == "dblp" and edges_path.exists():
        with edges_path.open("rb") as f:
            edges = pickle.load(f)

        # A-P-A
        apa = edges["a-p"].dot(edges["p-a"]).tocsr()

        # A-P-C-P-A
        apcpa = edges["a-p"].dot(edges["p-c"]).dot(edges["c-p"]).dot(edges["p-a"]).tocsr()

        for m in [apa, apcpa]:
            m.data[:] = 1.0

        mp_list = [
            sparse_mx_to_torch_sparse_tensor(normalize_adj(apa)),
            sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa)),
        ]

        print("The constructed meta-paths: APA, APCPA")

    else:
        mp = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
        mp_list = [mp]

    train, val, test = _load_split_file(path, labeled_ids, labeled_y)
    
    return [nei], [feat, feat], mp_list, pos, label, train, val, test


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
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
    path = "../data/dblp/"
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
    path = "../data/credit/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label[np.where(label == 4)] = 0
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
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')

    label[np.where(label == 2)] = 0
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

def load_amazon(ratio, type_num):
    if isinstance(ratio, int):
        ratio = [ratio]
        
    path = "../data/amazon/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label = encode_onehot(label)
    
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    nei_v = np.load(path + "nei_v.npy", allow_pickle=True)
    
    try:
        feat_u = sp.load_npz(path + "u_feat.npz")
        feat_u = th.FloatTensor(preprocess_features(feat_u))
    except:
        feat_u = np.load(path + "u_feat.npy")
        feat_u = th.FloatTensor(preprocess_features1(feat_u))
        
    upu = sp.load_npz(path + "upu.npz")
    usu = sp.load_npz(path + "usu.npz")
    uvu = sp.load_npz(path + "uvu.npz")
    pos = sp.load_npz(path + "pos.npz")
    
    # ==========================================
    # 核心补丁：自动对齐特征和图的维度 (Auto-Align)
    # ==========================================
    num_nodes = upu.shape[0]
    if feat_u.shape[0] < num_nodes:
        # 特征不够，用 0 补齐 (Padding)
        pad_len = num_nodes - feat_u.shape[0]
        feat_u = th.cat([feat_u, th.zeros(pad_len, feat_u.shape[1])], dim=0)
    elif feat_u.shape[0] > num_nodes:
        # 特征多了，直接截断 (Truncating)
        feat_u = feat_u[:num_nodes]
    # ==========================================
    
    # 此时维度已完美对齐，放心复用
    feat_p = feat_u
    feat_s = feat_u
    feat_v = feat_u
    
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    
    nei_p = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_p)]
    nei_s = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_s)]
    nei_v = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_v)]
    
    upu = sparse_mx_to_torch_sparse_tensor(normalize_adj(upu))
    usu = sparse_mx_to_torch_sparse_tensor(normalize_adj(usu))
    uvu = sparse_mx_to_torch_sparse_tensor(normalize_adj(uvu))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    return [nei_p, nei_s, nei_v], [feat_u, feat_p, feat_s, feat_v], [upu, usu, uvu], pos, label, train, val, test


def load_yelpchi(ratio, type_num):
    # 1. 防止命令行单整数报错
    if isinstance(ratio, int):
        ratio = [ratio]
        
    path = "../data/yelpchi/"
    label = np.load(path + "labels.npy").astype('int32')
    label[np.where(label == 2)] = 0
    label[np.where(label == 3)] = 0
    label = encode_onehot(label)
    
    # 必须先 load
    nei_u = np.load(path + "nei_u.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    nei_t = np.load(path + "nei_t.npy", allow_pickle=True)
    
    try:
        feat_r = sp.load_npz(path + "r_feat.npz")
        feat_r = th.FloatTensor(preprocess_features(feat_r))
    except:
        feat_r = np.load(path + "r_feat.npy")
        feat_r = th.FloatTensor(preprocess_features1(feat_r))
        
    # 复用 Review 特征
    feat_u = feat_r
    feat_s = feat_r
    feat_t = feat_r
    
    rur = sp.load_npz(path + "rur.npz")
    rsr = sp.load_npz(path + "rsr.npz")
    rtr = sp.load_npz(path + "rtr.npz")
    pos = sp.load_npz(path + "pos.npz")
    
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    
    # 孤立节点自环保护
    nei_u = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_u)]
    nei_s = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_s)]
    nei_t = [th.LongTensor(i) if len(i) > 0 else th.LongTensor([idx]) for idx, i in enumerate(nei_t)]
    
    rur = sparse_mx_to_torch_sparse_tensor(normalize_adj(rur))
    rsr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rsr))
    rtr = sparse_mx_to_torch_sparse_tensor(normalize_adj(rtr))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    return [nei_u, nei_s, nei_t], [feat_r, feat_u, feat_s, feat_t], [rur, rsr, rtr], pos, label, train, val, test

# def load_credit(ratio, type_num):
#     # The order of node types: 0 p 1 a 2 s
#     path = "../data/credit/"
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
    if dataset == "xinyongka":
        return load_credit_single_pkl(dataset)
    elif (Path(__file__).resolve().parents[2] / "data" / "pkl" / dataset).exists():
        data = load_pkl_anomaly(dataset)
    elif dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "credit":
        data = load_credit(ratio, type_num)
    elif dataset == "amazon":
        data = load_amazon(ratio, type_num)
    elif dataset == "yelpchi":
        data = load_yelpchi(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)

    return data
