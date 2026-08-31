# TAHIA graph fraud detection — reproducibility release

This repository contains the source code and processed datasets used to reproduce the TAHIA experiments and their comparison baselines. Generated logs, checkpoints, IDE settings, caches, compressed archives, and local machine configuration are intentionally excluded.

## Repository layout

```text
.
├── baseline/
│   ├── CARE-GNN/
│   ├── FRAUDRE-main/
│   ├── GAAP/
│   ├── GCN/
│   ├── HGMAE-master/
│   ├── HeCo-main/
│   ├── SpaceGNN/
│   └── UniGAD-main/
└── mycode/
    └── TAHIA/
        ├── data/
        └── src/
```

The `credit` adapters deliberately read `mycode/TAHIA/data/credit` so every main comparison uses the same processed graph. Keep `baseline/` and `mycode/` together at the repository root.

## Included datasets

| Dataset | Nodes | Feature dimension | Packaged representation |
| --- | ---: | ---: | --- |
| ACM | 8,994 | 1,902 | homogeneous pkl copies and TAHIA heterogeneous files |
| DBLP | 7,305 | 334 | homogeneous pkl copies and TAHIA heterogeneous files |
| Amazon | 11,944 | 25 | homogeneous pkl copies, HeCo files, and TAHIA files |
| YelpChi | 45,954 | 32 | homogeneous pkl copies and HeCo files |
| Credit | 13,351 | 32 | canonical TAHIA graph; 13,056 labeled card nodes and 295 auxiliary nodes |

The canonical credit labels contain 767 positive/fraud samples among the 13,056 labeled card nodes. A legacy 34-dimensional `xinyongka/credit.pkl` is retained only for backward compatibility with old commands; use `--data credit` for the paper's unified experimental setting.

All packaged pkl/NumPy files passed shape, index-range, edge-shape, and embedding-row consistency checks before release. These binary files are serialized Python objects; load only copies obtained from a trusted source.

## Environment

Python 3.10 on Linux is recommended. The local reproduction adapters were prepared around PyTorch 2.2 and DGL 2.x. Original upstream entry points may document older environments in their own README files.

For a CPU environment:

```bash
conda create -n tahia-open python=3.10 -y
conda activate tahia-open
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the PyTorch and DGL wheels matching the server's CUDA version first, then install the remaining requirements. For example, for a CUDA 12.1 PyTorch build:

```bash
conda create -n tahia-open python=3.10 -y
conda activate tahia-open
python -m pip install --upgrade pip
python -m pip install torch==2.2.1 torchdata==0.7.1 --index-url https://download.pytorch.org/whl/cu121
# Install the DGL 2.x CUDA wheel matching PyTorch/CUDA from https://www.dgl.ai/pages/start.html
python -m pip install -r requirements.txt
```

Run every command below from the repository root unless a `cd` command is shown.

## Reproduce TAHIA

```bash
python -u mycode/TAHIA/src/train.py --dataset credit --gpu_id 0 --seed 0
python -u mycode/TAHIA/src/train.py --dataset acm --gpu_id 0 --seed 0
python -u mycode/TAHIA/src/train.py --dataset dblp --gpu_id 0 --seed 0
python -u mycode/TAHIA/src/train.py --dataset amazon --gpu_id 0 --seed 0 --loss_w0 0.07 --loss_w1 0.93
```

Use `--gpu_id -1` for CPU. TAHIA does not package a YelpChi configuration because its current dense adjacency/similarity construction is not suitable for the 45,954-node graph.

Optional LLM diagnosis is disabled by default. Enable it with `--llm_feedback`; credentials are read only from `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` environment variables and must never be committed.

## Reproduce the main comparison models

GCN:

```bash
for DATA in credit acm dblp amazon yelpchi; do
  python -u baseline/GCN/train.py --data "$DATA" --device cuda --epochs 300 --test_epoch 10
done
```

CARE-GNN:

```bash
python -u baseline/CARE-GNN/train.py --data credit --num-epochs 200 --test-epochs 10 --batch-size 256
python -u baseline/CARE-GNN/train.py --data acm --num-epochs 200 --test-epochs 10 --batch-size 256
python -u baseline/CARE-GNN/train.py --data dblp --num-epochs 200 --test-epochs 10 --batch-size 256
python -u baseline/CARE-GNN/train.py --data amazon --num-epochs 200 --test-epochs 10 --batch-size 1024
python -u baseline/CARE-GNN/train.py --data yelpchi --num-epochs 200 --test-epochs 10 --batch-size 1024
```

FRAUDRE:

```bash
python -u baseline/FRAUDRE-main/train.py --data credit --num_epochs 300 --test_epochs 10 --batch-size 100
python -u baseline/FRAUDRE-main/train.py --data acm --num_epochs 300 --test_epochs 10 --batch-size 32
python -u baseline/FRAUDRE-main/train.py --data dblp --num_epochs 300 --test_epochs 10 --batch-size 100
python -u baseline/FRAUDRE-main/train.py --data amazon --num_epochs 300 --test_epochs 10 --batch-size 1024
python -u baseline/FRAUDRE-main/train.py --data yelpchi --num_epochs 300 --test_epochs 10 --batch-size 1024
```

UniGAD:

```bash
for DATA in credit acm dblp amazon yelpchi; do
  python -u baseline/UniGAD-main/official/src/main_local.py --data "$DATA" --device cuda --epoch_pretrain 100 --epoch_ft 300
done
```

SpaceGNN:

```bash
python -u baseline/SpaceGNN/official/main_local.py --data credit --device cuda --nepoch 200 --test_epoch 20 --batch_size 256
python -u baseline/SpaceGNN/official/main_local.py --data acm --device cuda --nepoch 200 --test_epoch 20 --batch_size 50
python -u baseline/SpaceGNN/official/main_local.py --data dblp --device cuda --nepoch 200 --test_epoch 20 --batch_size 50
python -u baseline/SpaceGNN/official/main_local.py --data amazon --device cuda --nepoch 200 --test_epoch 20 --batch_size 256
python -u baseline/SpaceGNN/official/main_local.py --data yelpchi --device cuda --nepoch 200 --test_epoch 20 --batch_size 256
```

GAAP:

```bash
python -u baseline/GAAP/official/main_local.py --data credit --device cuda --epoch 300 --test_epoch 20 --batch_size 128 --eval_batch_size 4096
python -u baseline/GAAP/official/main_local.py --data acm --device cuda --epoch 300 --test_epoch 20 --batch_size 128 --eval_batch_size 4096
python -u baseline/GAAP/official/main_local.py --data dblp --device cuda --epoch 300 --test_epoch 20 --batch_size 128 --eval_batch_size 4096
python -u baseline/GAAP/official/main_local.py --data amazon --device cuda --epoch 300 --test_epoch 20 --batch_size 128 --eval_batch_size 4096
python -u baseline/GAAP/official/main_local.py --data yelpchi --device cuda --epoch 300 --test_epoch 20 --batch_size 64 --eval_batch_size 2048 --n_bins 8
```

## Additional heterogeneous baselines

HGMAE uses its packaged heterogeneous ACM/DBLP/Credit files and the shared UniGAD pkl copies for Amazon/YelpChi:

```bash
cd baseline/HGMAE-master
python -u main.py --dataset acm --use_cfg
python -u main.py --dataset dblp --use_cfg
python -u main.py --dataset credit --use_cfg
python -u main.py --dataset amazon --use_cfg
python -u main.py --dataset yelpchi --use_cfg
cd ../..
```

HeCo:

```bash
cd baseline/HeCo-main/code
python -u main.py --dataset acm
python -u main.py --dataset dblp
python -u main.py --dataset credit
python -u main.py --dataset amazon
python -u main.py --dataset yelpchi
cd ../../..
```

## Quick smoke tests

The following commands verify data loading and one training epoch without writing repository logs:

```bash
python baseline/GCN/train.py --data credit --device cpu --epochs 1 --test_epoch 1 --no_log
python baseline/CARE-GNN/train.py --data credit --num-epochs 1 --test-epochs 1 --batch-size 256 --no-cuda --log-file /tmp/care-credit.log
python baseline/FRAUDRE-main/train.py --data credit --num_epochs 1 --test_epochs 1 --batch-size 100 --no_cuda --log-file /tmp/fraudre-credit.log
```

## Evaluation protocol

The packaged local adapters follow the experiment's stratified 1:9 train:test split. For compatibility with the original experiment scripts, some entries expose the training indices as a validation copy. Report the fixed-epoch final test metrics rather than choosing an epoch using test performance.

## License and attribution

TAHIA and repository-authored adapters are released under the Apache License 2.0. Third-party model code and datasets retain their original rights. Read `THIRD_PARTY_NOTICES.md` before redistribution. Several upstream research repositories did not publish a license file, so author permission must be confirmed before making a combined public release.
