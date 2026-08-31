# GCN

Two-layer Graph Convolutional Network baseline following Kipf & Welling,
"Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017).

Reference implementations:

- `tkipf/gcn`: original TensorFlow implementation.
- `tkipf/pygcn`: PyTorch proof-of-concept implementation.

This local baseline uses the same ACM/DBLP/credit loaders and
train:test = 1:9 stratified split as the other comparison models. Logs are
written to `GCN/results/{data}.log` by default.

## DGL Public Trial Datasets

From the repo root, convert the downloaded DGL graph files in `../datasets`
into the shared pkl adapter format. The converter enforces stratified
train:test = 1:9 and writes `splits.npz`.

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

Then run:

```bash
cd GCN
python train.py --data amazon --device cuda --epochs 200 --test_epoch 10 --hidden 64
python train.py --data yelpchi --device cuda --epochs 200 --test_epoch 10 --hidden 64
```

## WSL Run

```bash
cd GCN
python train.py --data acm --device cuda --epochs 200 --test_epoch 10 --hidden 64
python train.py --data dblp --device cuda --epochs 200 --test_epoch 10 --hidden 64
python train.py --data credit --device cuda --epochs 200 --test_epoch 10 --hidden 64
```
