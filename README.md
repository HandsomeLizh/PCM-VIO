# PCM VIO

基于 Visual-Selective-VIO 的 Predictive Common Motion v3 研究交接版。当前唯一预设为 `PCM3`，用 IMU 与历史状态预测共同运动，再用门控视觉新息和视觉补充特征修正估计，并学习按需调用视觉编码器。

截至 2026 年 11 月 13 日，地面数据上的 v3 第一阶段训练与消融已完成；第二阶段策略训练及 v3 KITTI 结果尚未完成。全视觉精度和停车漂移改善，但低调用率、冻结画面门控与不确定性仍有明显问题。

- [架构与交接说明](docs/HANDOFF.md)：模型、训练、checkpoint、数据、接手步骤和待办。
- [固定 benchmark 与测评流程](docs/BENCHMARK.md)：划分、单位、种子、命令、结果判定。
- [当前进度与结果](docs/PROGRESS.md)：最新结果、已知问题与历史参考。
- [机器可读协议](benchmarks/protocol_v1.json)与[原始指标证据](benchmarks/evidence)：固定本次交接的比较口径。

## 环境

已验证环境：Python 3.10、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、RTX 3090。系统 ROS 的 `PYTHONPATH` 会干扰依赖，运行前清除。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

只使用已有 checkpoint 测试不需要 FlowNet 初始化权重。重新训练需要交接包中的 `pretrain_models/flownets_bn_EPE2.459.pth.tar`。CPU-only 环境可安装相同版本的 CPU 构建作功能测试；正式数值复现使用上述 CUDA 环境。原始 bag 转换另需 `rosbags==0.11.5`，固定 benchmark 直接使用已经转换的地面数据。

## 复现最新结果

共享包位于 `/remote-home/share/ums_lzj/PCM-VIO_handoff_20261113`。代码仓库不包含 checkpoint、数据、图像缓存或完整实验日志。

```bash
export BUNDLE=/remote-home/share/ums_lzj/PCM-VIO_handoff_20261113
export GROUND_DATA="$PWD/data/data_ground"
mkdir -p data
tar -xf "$BUNDLE/data_ground.tar" -C data  # 首次运行解包；保留转换版本
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
python tools/benchmark.py --dataset ground \
  --ckpt "$BUNDLE/checkpoints/pcm3_g/stage1_last.pth" \
  --data-dir "$GROUND_DATA" --gpu 0 --name pcm3_reproduce
```

该命令顺序执行全视觉、随机 0/10/25/50% 视觉、blank 扰动六项测试，并保存 checkpoint SHA256、代码版本、命令、原始指标及正确汇总的 `results/pcm3_reproduce/summary.json`。加 `--dry-run` 可只查看命令。输出目录已存在时拒绝覆盖。

## 验证代码

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests -q -p no:cacheprovider
```

CPU 测试包括两阶段合成数据训练和断点续训。B0 的 CUDA 专项会跳过；可在有空闲 GPU 且备好 FlowNet 权重时单独运行。真实 GPU 训练续训受 cuDNN LSTM dropout 状态影响，不承诺逐位相同。

## 代码范围与来源

`crvio/` 是当前 PCM v3；`train_crvio.py` / `test_crvio.py` 是主入口。`model.py` 保留原始编码骨干和 B0，`train.py` / `test.py` 仅用于历史 baseline。旧 B1–B7、BM、PCM v1/v2 的实现不混入当前预设，独立快照放在共享包 `legacy/`。

本项目派生自 [mingyuyng/Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO)，基线提交 `27b06b3`；保留[上游说明与论文引用](docs/UPSTREAM_README.md)。本交接没有为上游代码新增或推定许可证；代码来源与使用条件仍应按上游核实。
