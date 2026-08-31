# SpaceGNN

`official/` 里是从官方仓库 `xydong127/SpaceGNN` 直接拉下来的代码。

本地适配只新增两个文件：

- `official/local_data.py`: 读取 `../UniGAD-main/data/pkl/{acm,dblp,...}`，按 CARE/FRAUDRE 的 1:9 有标签节点划分构图。
- `official/main_local.py`: 在官方 `model.py` 基础上训练，并输出 test best 的 `precision`、`recall`、`macro_f1`、`auroc`。

## WSL 运行

先跑 smoke test 确认环境和数据路径：

```bash
cd SpaceGNN/official
python main_local.py --data acm --device cuda --nepoch 1 --test_epoch 1 --hid_dim 32 --batch_size 128
python main_local.py --data dblp --device cuda --nepoch 1 --test_epoch 1 --hid_dim 32 --batch_size 128
```

推荐论文表格 run：

```bash
python main_local.py --data acm --device cuda --nepoch 200 --test_epoch 20 --hid_dim 128 --batch_size 128
python main_local.py --data dblp --device cuda --nepoch 200 --test_epoch 20 --hid_dim 128 --batch_size 128
```

当前论文实验口径固定为 train:test = 1:9，没有额外独立 validation split。`main_local.py` 会同时打印 `[test best]` 和 `[test final]`；论文表格建议使用固定 epoch 的 `[test final]`，避免训练集副本验证集过早满分导致 best epoch 选择失真。
运行日志默认写入 `SpaceGNN/results/{data}.log`；如需关闭写日志可追加 `--no_log`。

公开 DGL 试点数据集。先在仓库根目录把 `../datasets` 下的 DGL graph 转成共享 pkl 格式；转换脚本会强制 stratified train:test = 1:9：

```bash
cd ..
python tools/prepare_dgl_pkl_datasets.py --datasets amazon yelpchi
```

然后运行：

```bash
cd SpaceGNN/official
python main_local.py --data amazon --device cuda --nepoch 200 --test_epoch 20 --hid_dim 128 --batch_size 128
python main_local.py --data yelpchi --device cuda --nepoch 200 --test_epoch 20 --hid_dim 128 --batch_size 128
```

可选更长 run。ACM 上 200 epoch 时验证集很早满分，继续到 500 不一定更好，但可以作为补充尝试：

```bash
python main_local.py --data acm --device cuda --nepoch 500 --test_epoch 20 --hid_dim 128 --batch_size 128
python main_local.py --data dblp --device cuda --nepoch 500 --test_epoch 20 --hid_dim 128 --batch_size 128
```

公开信用卡数据集 run。`credit` 入口统一读取仓库根目录下的 `mycode/TAHIA/data/credit`，仍按 train:test = 1:9 stratified split：

```bash
python main_local.py --data credit --device cuda --nepoch 1 --test_epoch 1 --hid_dim 32 --batch_size 128
python main_local.py --data credit --device cuda --nepoch 200 --test_epoch 20 --hid_dim 128 --batch_size 128
python main_local.py --data credit --device cuda --nepoch 500 --test_epoch 20 --hid_dim 128 --batch_size 128
```

本机没有 CUDA 时可以改成 `--device cpu` 跑通流程。CPU 上 ACM 200 epoch 约 215 秒。
