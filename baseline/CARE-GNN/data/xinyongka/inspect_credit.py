import pickle
import numpy as np
import scipy.sparse as sp

path = "credit.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print("type:", type(data))

if isinstance(data, dict):
    print("keys:", data.keys())
    for k, v in data.items():
        print("\nKEY:", k)
        print("type:", type(v))
        if hasattr(v, "shape"):
            print("shape:", v.shape)
        if sp.issparse(v):
            print("sparse nnz:", v.nnz)
        elif isinstance(v, (list, tuple)):
            print("len:", len(v))
            if len(v) > 0:
                print("first type:", type(v[0]))
                if hasattr(v[0], "shape"):
                    print("first shape:", v[0].shape)
else:
    print("len:", len(data) if hasattr(data, "__len__") else "no len")
    if isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            print("\nIDX:", i)
            print("type:", type(v))
            if hasattr(v, "shape"):
                print("shape:", v.shape)
            if sp.issparse(v):
                print("sparse nnz:", v.nnz)