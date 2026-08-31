# UniGAD

`official/` 里是从官方仓库 `lllyyq1121/UniGAD` 拉下来的源码。为了避免把官方仓库里的大数据文件也下载进来，这里只保留源码、README、requirements 和空数据目录标记。

本地适配只新增两个文件：

- `official/src/local_data.py`: 读取 `data/pkl/{acm,dblp,...}`，把 pkl 转成 UniGAD 官方代码使用的单图 node-label 输入格式，并按 CARE/FRAUDRE 的 1:9 有标签节点划分。
- `official/src/main_local.py`: 调用官方 GraphMAE 和 UnifyMLPDetector，在本地 ACM/DBLP 上训练，并输出 test best 的 `precision`、`recall`、`macro_f1`、`auroc`。

## WSL 运行

先跑 smoke test 确认环境和数据路径：

```bash
cd UniGAD-main/official/src
python main_local.py --data acm --device cuda --epoch_pretrain 1 --epoch_ft 1 --hid_dim 16
python main_local.py --data dblp --device cuda --epoch_pretrain 1 --epoch_ft 1 --hid_dim 16
```

推荐论文表格 run：

```bash
python main_local.py --data acm --device cuda --epoch_pretrain 50 --epoch_ft 200 --hid_dim 32
python main_local.py --data dblp --device cuda --epoch_pretrain 50 --epoch_ft 200 --hid_dim 32
```

运行日志默认写入 `UniGAD-main/results/{data}.log`；如需关闭写日志可追加 `--no_log`。

公开 DGL 试点数据集。先在仓库根目录把 `../datasets` 下的 DGL graph 转成共享 pkl 格式；转换脚本会强制 stratified train:test = 1:9：

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

然后运行：

```bash
cd UniGAD-main/official/src
python main_local.py --data amazon --device cuda --epoch_pretrain 50 --epoch_ft 200 --hid_dim 32
python main_local.py --data yelpchi --device cuda --epoch_pretrain 50 --epoch_ft 200 --hid_dim 32
```

可选更强 run。这个组合更接近之前 WSL 文档里建议的长训练设置，但耗时和显存都会增加：

```bash
python main_local.py --data acm --device cuda --epoch_pretrain 200 --epoch_ft 500 --hid_dim 256
python main_local.py --data dblp --device cuda --epoch_pretrain 200 --epoch_ft 500 --hid_dim 256
```

公开信用卡数据集 run。`credit` 入口统一读取仓库根目录下的 `mycode/TAHIA/data/credit`，仍按 train:test = 1:9 stratified split：

```bash
python main_local.py --data credit --device cuda --epoch_pretrain 1 --epoch_ft 1 --hid_dim 16
python main_local.py --data credit --device cuda --epoch_pretrain 50 --epoch_ft 200 --hid_dim 32
python main_local.py --data credit --device cuda --epoch_pretrain 200 --epoch_ft 500 --hid_dim 256
```

本机没有 CUDA 时可以改成 `--device cpu` 先跑通流程。CPU 上 ACM `50/200, hid_dim=32` 约 74 秒。
