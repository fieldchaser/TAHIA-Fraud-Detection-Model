import pickle
import numpy as np
import torch
import torch.nn.functional as F
import scipy
from sklearn.model_selection import train_test_split


class HIN(object):

    def __init__(self, dataset):
        self.dataset = dataset
        data_path = f'data/{dataset}/'
        with open(f'{data_path}node_features.pkl', 'rb') as f:
            self.features = pickle.load(f)
        with open(f'{data_path}edges.pkl', 'rb') as f:
            self.edges = pickle.load(f)
            print({k: {'shape': v.shape, 'nnz': v.nnz} for k, v in self.edges.items()})
        with open(f'{data_path}labels.pkl', 'rb') as f:
            self.labels = pickle.load(f)
            # print(len(self.labels[0]))
            # print(len(self.labels[1]))
            # print(len(self.labels[2]))
            # print(np.array(self.labels[1]))
        with open(f'{data_path}meta_data.pkl', 'rb') as f:
            self.__dict__.update(pickle.load(f))
        if scipy.sparse.issparse(self.features):
            self.features = self.features.todense()

    def to_torch(self, cf):
        '''
        Returns the torch tensor of the graph.
        Args:
            cf: The ModelConfig file.
        Returns:
            features, adj: feature and adj. matrix
            mp_emb: only available for models that uses mp_list.
            train_x, train_y, val_x, val_y, test_x, test_y: train/val/test index and labels
        '''
        features = torch.from_numpy(self.features).type(torch.FloatTensor).to(cf.dev)

        # features = torch.cat([features, add])
        # x表示对应样本的序号，y表示标签
        train_x, train_y, val_x, val_y, test_x, test_y = self.get_label(cf.dev)

        #邻接矩阵转换和归一化

        #使用 np.sum 函数对列表中的所有邻接矩阵进行逐元素相加，得到一个新的矩阵
        #self.edges.values() 都表示一个单个边的稀疏矩阵，两个矩阵相加后得到一个完整的邻接矩阵
        adj = np.sum(list(self.edges.values())).todense()
        adj = torch.from_numpy(adj).type(torch.FloatTensor).to(cf.dev)
        #使用 2 范数（欧几里得范数）进行归一化   确保每个节点在聚合其邻居的信息时，所有邻居的贡献都是等权的
        adj = F.normalize(adj, dim=1, p=2)

        #元路径嵌入转换
        mp_emb = {}
        if hasattr(cf, 'mp_list'):
            for mp in cf.mp_list:
                mp_emb[mp] = torch.from_numpy(self.mp_emb_dict[mp]).type(torch.FloatTensor).to(cf.dev)
        #特征归一化
        if hasattr(cf, 'feat_norm'):
            if cf.feat_norm > 0:
                features = F.normalize(features, dim=1, p=cf.feat_norm)
                for mp in cf.mp_list:
                    mp_emb[mp] = F.normalize(mp_emb[mp], dim=1, p=cf.feat_norm)

        #DEBUG
        print("adj shape:", adj.shape)
        print("adj type:", type(adj))
        print("adj dense memory MB:", adj.numel() * adj.element_size() / 1024 / 1024)

        return features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y

    def load_mp_embedding(self, cf):
        """
        用于加载预训练的元路径（meta-path）嵌入，并将其存储在 self.mp_emb_dict 字典中
        """
        '''Load pretrained mp_embedding'''
        self.mp_emb_dict = {}
        for mp in cf.mp_list:
            f_name = f'{cf.data_path}{mp}_emb.pkl'
            with open(f_name, 'rb') as f:
                z = pickle.load(f)

                #检查并处理零行，避免零向量存在
                zero_lines = np.nonzero(np.sum(z, 1) == 0)
                if len(zero_lines) > 0:
                    # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), mode, zero_lines))
                    z[zero_lines, :] += 1e-8

                if z.shape[0] != self.features.shape[0]:
                    raise ValueError(
                        f'{mp}_emb row count {z.shape[0]} does not match node count {self.features.shape[0]}. '
                        f'Rebuild {f_name} before training.'
                    )

                self.mp_emb_dict[mp] = z
        return self

    def get_label(self, dev):
        """
        Convert original labels into binary anomaly labels and
        perform a stratified 1:9 train-test split.
        """
        if len(self.labels) > 1:
            labels_arr = np.concatenate([np.array(x) for x in self.labels], axis=0)
            all_x = labels_arr[:, 0].astype(np.int64)
            raw_y = labels_arr[:, 1].astype(np.int64)
        else:
            raw_y = np.array(self.labels[0]).astype(np.int64)
            all_x = np.arange(len(raw_y), dtype=np.int64)

        print("Before binary:", np.unique(raw_y, return_counts=True))

        if self.dataset in ['acm', 'dblp']:
            # ACM / DBLP: class 1 is anomaly/fraud, others are normal
            all_y = (raw_y == 1).astype(np.int64)
        else:
            # Credit-like binary datasets: label 1 is fraud
            all_y = (raw_y == 1).astype(np.int64)

        train_x, test_x, train_y, test_y = train_test_split(
            all_x,
            all_y,
            stratify=all_y,
            train_size=0.1,
            random_state=2,
            shuffle=True
        )

        # For compatibility with the original framework.
        # The paper result should be reported on test_x/test_y.
        val_x = train_x.copy()
        val_y = train_y.copy()

        train_x = torch.from_numpy(train_x).type(torch.LongTensor).to(dev)
        train_y = torch.from_numpy(train_y).type(torch.LongTensor).to(dev)
        val_x = torch.from_numpy(val_x).type(torch.LongTensor).to(dev)
        val_y = torch.from_numpy(val_y).type(torch.LongTensor).to(dev)
        test_x = torch.from_numpy(test_x).type(torch.LongTensor).to(dev)
        test_y = torch.from_numpy(test_y).type(torch.LongTensor).to(dev)

        print("Train:", train_x.shape, torch.bincount(train_y.detach().cpu()))
        print("Test:", test_x.shape, torch.bincount(test_y.detach().cpu()))

        return train_x, train_y, val_x, val_y, test_x, test_y
