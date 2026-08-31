# FRAUDRE
**Overview** 

This repository is PyTorch implementation of the method proposed in "FRAUDRE: Fraud Detection Dual-Resistant toGraph Inconsistency and Imbalance".

## Local DGL Trial Datasets

The local comparison setup can read converted DGL public datasets `amazon`
and `yelpchi`. From the repo root, convert the downloaded DGL graph
files in `../datasets` into the shared pkl adapter format. The converter
enforces stratified train:test = 1:9:

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

Then run FRAUDRE:

```bash
cd FRAUDRE-main
python train.py --data amazon --batch-size 100 --lr 0.1 --num_epochs 201 --test_epochs 10
python train.py --data yelpchi --batch-size 100 --lr 0.1 --num_epochs 201 --test_epochs 10
python train.py --data credit --batch-size 100 --lr 0.1 --num_epochs 201 --test_epochs 10
```

Training logs are written automatically to `results/{dataset}.log`.
