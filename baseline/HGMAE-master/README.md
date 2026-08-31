# HGMAE

This is the official repo for the AAAI'23 paper "[Heterogeneous Graph Masked Autoencoders](https://arxiv.org/abs/2208.09957)".


## How to Run the code

```
python main.py --dataset dblp --task classification --use_cfg 
```

Supported datasets: "dblp", "freebase", "acm", "aminer".

Supported tasks: "classification", "clustering". 

## Local DGL Trial Datasets

The local comparison setup can read converted DGL public datasets `amazon`
and `yelpchi`. From the repo root, convert the downloaded DGL graph
files in `../datasets` into the shared pkl adapter format. The converter
enforces stratified train:test = 1:9:

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

Then run HGMAE:

```bash
cd HGMAE-master
python main.py --dataset amazon --use_cfg
python main.py --dataset yelpchi --use_cfg
python main.py --dataset credit --use_cfg
```

Training logs are written automatically to `results/{dataset}.log`.


## Citing HGMAE

If you find HGMAE useful, please cite our paper.
```
@inproceedings{HGMAE,
  title={Heterogeneous Graph Masked Autoencoders},
  author={Tian, Yijun and Dong, Kaiwen and Zhang, Chunhui and Zhang, Chuxu and Chawla, Nitesh V},
  booktitle = {AAAI},
  year={2023}
}
```
