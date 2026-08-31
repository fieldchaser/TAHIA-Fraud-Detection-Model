# GAAP

`official/` 里是从官方仓库 `AtwoodDuan/GAAP` 直接拉下来的源码。

本地适配只新增两个文件：

- `official/local_data.py`: 读取 `../UniGAD-main/data/pkl/{acm,dblp,...}`，按 CARE/FRAUDRE 的 1:9 有标签节点划分构图。
- `official/main_local.py`: 基于官方 `mycode/nn/gnn.py` 里的 GAAP/GraphSAGE 模型训练，并输出 test best 的 `precision`、`recall`、`macro_f1`、`auroc`。

## WSL 运行

先跑 smoke test 确认环境和数据路径：

```bash
cd GAAP/official
python main_local.py --data acm --device cuda --epoch 1 --test_epoch 1 --hidden 32 --batch_size 256 --eval_batch_size 4096
python main_local.py --data dblp --device cuda --epoch 1 --test_epoch 1 --hidden 32 --batch_size 256 --eval_batch_size 4096
```

推荐论文表格 run：

```bash
python main_local.py --data acm --device cuda --epoch 300 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
python main_local.py --data dblp --device cuda --epoch 300 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
```

`main_local.py` 默认启用完整 GAAP 组件：DyPLE、MHA、GNN residual。
如果需要做消融，可显式追加 `--no-use_dyple`、`--no-use_mha` 或 `--no-gnn_use_res`。
运行日志默认写入 `GAAP/results/{data}.log`；如需关闭写日志可追加 `--no_log`。

当前论文实验口径固定为 train:test = 1:9，没有额外独立 validation split。`main_local.py` 会同时打印 `[test best]` 和 `[test final]`；论文表格建议使用固定 epoch 的 `[test final]`，避免用测试集分数挑 epoch。

公开 DGL 试点数据集。先在仓库根目录把 `../datasets` 下的 DGL graph 转成共享 pkl 格式；转换脚本会强制 stratified train:test = 1:9：

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

然后运行：

```bash
cd GAAP/official
python main_local.py --data amazon --device cuda --epoch 300 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
python main_local.py --data yelpchi --device cuda --epoch 300 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
```

可选更长 run。ACM CPU 结果在 300 epoch 内 best 出现在 240 epoch，500 epoch 未必继续提升，但可以用来确认 DBLP 是否还没收敛：

```bash
python main_local.py --data acm --device cuda --epoch 500 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
python main_local.py --data dblp --device cuda --epoch 500 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
```

公开信用卡数据集 run。`credit` 入口统一读取仓库根目录下的 `mycode/TAHIA/data/credit`，仍按 train:test = 1:9 stratified split：

```bash
python main_local.py --data credit --device cuda --epoch 1 --test_epoch 1 --hidden 32 --batch_size 256 --eval_batch_size 4096
python main_local.py --data credit --device cuda --epoch 300 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
python main_local.py --data credit --device cuda --epoch 500 --test_epoch 20 --hidden 64 --batch_size 256 --eval_batch_size 4096
```

本机没有 CUDA 时可以改成 `--device cpu` 先跑通流程。CPU 上 ACM 300 epoch 约 319 秒。
