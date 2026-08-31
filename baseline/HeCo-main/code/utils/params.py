import argparse
import sys


argv = sys.argv
dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument('--dataset', type=str, default='acm')
dataset, _ = dataset_parser.parse_known_args()
dataset = dataset.dataset


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=15)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=38)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5) 

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def credit_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="credit")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=24)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.5)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[3, 8])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def pkl_anomaly_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--ratio', type=int, default=[10])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=30)
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.type_num = [0]
    args.nei_num = 1
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)    
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=500)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[1, 18, 2])
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


# def credit_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_emb', action="store_true")
#     parser.add_argument('--turn', type=int, default=0)
#     parser.add_argument('--dataset', type=str, default="credit")
#     parser.add_argument('--ratio', type=int, default=[10, 20, 30])
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=32)
#     parser.add_argument('--hidden_dim', type=int, default=16)
#     parser.add_argument('--nb_epochs', type=int, default=1)

#     # The parameters of evaluation
#     parser.add_argument('--eva_lr', type=float, default=0.01)
#     parser.add_argument('--eva_wd', type=float, default=0)

#     # The parameters of learning process
#     parser.add_argument('--patience', type=int, default=20)
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--l2_coef', type=float, default=0)

#     # model-specific parameters
#     parser.add_argument('--tau', type=float, default=0.5)
#     parser.add_argument('--feat_drop', type=float, default=0.1)
#     parser.add_argument('--attn_drop', type=float, default=0.3)
#     parser.add_argument('--sample_rate', nargs='+', type=int, default=[1, 18, 2])
#     parser.add_argument('--lam', type=float, default=0.5)

#     args, _ = parser.parse_known_args()
#     args.type_num = [13056, 295]  # the number of every node type
#     args.nei_num = 1  # the number of neighbors' types
#     return args


def set_params():
    if dataset in {"acm", "dblp", "amazon", "yelpchi", "xinyongka"}:
        args = pkl_anomaly_params()
    elif dataset == "credit":
        args = credit_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
