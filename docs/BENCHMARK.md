# PCM v3 固定 benchmark 与测评流程

协议标识为 `pcm-v3-handoff-v1`，机器可读定义位于 `benchmarks/protocol_v1.json`。本协议固定当前数据和计算口径，防止接手后因划分、统计或调用率改变而产生虚假进步。改动数据转换、时间同步、尺度、序列列表、测试随机性或指标实现时，应另起协议版本，不能覆盖 v1 结果。

## 数据划分

| 数据集 | 训练 | 验证与阈值标定 | 测试 |
| --- | --- | --- | --- |
| 地面机器人 | m2_bridge2、m2_street1、m2_switch、m3_dynamic01、m3_shaturn01、m3_wheelfloat01 | m2_tree | m2_building2、m2_street2、m3_varyingillu01、m3_occlusion01 |
| KITTI | 00、01、02、04、06、08 | 09 | 05、07、10 |

地面主表是前三条正常序列的等权平均；`m3_occlusion01` 单独列为压力测试，也同时保留四条总平均。原 `test_crvio.py` 打印的 `mean` 包含四条序列，**不能直接复制为正常序列主表**。KITTI 主表是 05/07/10 等权平均。

09 参加训练的历史 KITTI 实验单独归档，不进入此 benchmark。阈值仅在验证序列标定，目标 25% 是验证集目标，必须同时报告测试集实际视觉使用率，不能把它写成测试集恰好 25%。

地面 benchmark 绑定共享包中的已转换数据。原 bag 转换通过真值与陀螺信号估计时间/旋转关系，并做航向修正；它不是无需真值的在线标定成果。保留 `data_ground/meta/` 中每条序列的来源、时间偏移、旋转、质量检查和“未建模杆臂”说明。修改转换应重建整个比较基线。

`benchmarks/data/*.json` 固定每条序列的帧数、完整排序文件名、首尾图像字节、pose 和 IMU 的 SHA256。测评启动前自动检查所有训练/验证/测试序列，发现不符即拒绝运行。快速检查不覆盖每张内部图像的字节，地面数据全部字节由共享包 `data_ground.tar` 的 SHA256 保护；KITTI 图像是外部共享数据。

## 指标与单位

| 指标 | 定义与单位 |
| --- | --- |
| t_rel | 分段累计轨迹平移漂移 / 段长 ×100，% |
| r_rel | 分段累计轨迹旋转漂移 / 段长，度/100 米，**不是百分比** |
| t_rmse | 每区间三维相对平移误差的欧氏 RMSE，米；不是轨迹 ATE |
| r_rmse | 每区间欧拉角向量差的 RMSE，度；不是 SO(3) 测地误差 |
| usage | 实际视觉编码调用比例，%，包含首区间强制调用 |

实现位于 `utils/kitti_eval.py` 与 `utils/utils.py`。轨迹由相对位姿积分，无尺度或 Sim(3) 后对齐。KITTI 段长固定 100、200、…、800 米；地面固定 5、10、15、20 米，分段起点步长均为 10 帧。地面 r_rel 仍统一报度/100 米。

沿用的速度图使用 baseline 的 10 Hz 换算，因此地面 15 Hz 的速度着色不作定量结论。此问题不改变 t_rel/r_rel；修图时不要顺带改动历史误差口径。

## 权重与推理设置

- 只报告预先确定训练长度的最后一个 epoch：地面阶段一 40、KITTI 阶段一 60、两者阶段二 40。`best` 和中间 epoch 可作诊断，不能替换最终结果。
- 评测加载 checkpoint 的 `model`，即 EMA 权重；不使用 `model_raw`。
- 图像 256×512，窗口 11 帧；地面 dt=0.0666667 秒，KITTI dt=0.1 秒。
- 每条序列只在开始初始化状态，分块之间连续传递 `(h,c,tau)`。首区间强制用视觉；所谓随机 0% 是其后不再调用，实际使用率略大于零。
- 关闭 BN 重校准、`max_skip` 兜底和额外同步错位。保留 checkpoint 的 bf16 视觉编码设置。
- 完整评测不得使用截短序列、`max_chunks` 或仅部分帧。GPU 非确定性可能造成小数差异，跨硬件不承诺逐位复现。

## 固定测试矩阵

每个估计器 checkpoint 执行以下六项，不用未训练的第一阶段策略作为结果。

| 测试 | 规则 | 决策种子 |
| --- | --- | --- |
| full | 所有区间用视觉 | 0 |
| rand0 | 首区间外不调用视觉 | 0 |
| rand10 | 每步独立随机，p=0.10 | 0、1、2 |
| rand25 | 每步独立随机，p=0.25 | 0、1、2 |
| rand50 | 每步独立随机，p=0.50 | 0、1、2 |
| blank | 全视觉调用，视觉施加 blank 扰动 | 0 |

blank 是 `uniform/dark/frozen/cover` 的混合，不是全程纯黑图；按每个分块施加，固定 scale=1、p_severe=0.3。`tools/pcm_diag.py` 会另测各子类型，不可混为同一个测试。

第二阶段增加 learned 采样策略（种子 0–4）、验证集阈值标定到 25%（确定性）和每四个区间调用一次的 interval4。与 random 25% 比较时报告实际使用率，不以“用了策略”直接归因于不确定性；归因还需要 `--policy_signal zero` 的同配方训练消融。

三次随机决策仅反映同一个模型的决策波动，不是三个训练种子。新性能结论固定训练 seed 0、1、2，分别完整评测后再报告跨训练种子的均值和标准差。

## 执行与保存

```bash
export BUNDLE=/remote-home/share/ums_lzj/PCM-VIO_handoff_20261113
export GROUND_DATA="$PWD/data/data_ground"
unset PYTHONPATH
# 首次先按 README 解包 data_ground.tar 到 data/。
python tools/benchmark.py --dataset ground --name ground_v3_seed0 \
  --ckpt "$BUNDLE/checkpoints/pcm3_g/stage1_last.pth" \
  --data-dir "$GROUND_DATA" --gpu 0

# 对已经训练好的 KITTI 第二阶段 checkpoint 执行九项测试。
python tools/benchmark.py --dataset kitti --name kitti_v3_s2_seed0 \
  --ckpt results/kitti_v3/checkpoints/stage2_last.pth \
  --data-dir "$BUNDLE/data_kitti" --gpu 0 --include-policy
```

工具先校验版本、数据划分、图像尺寸、窗口、dt 和最终 epoch，再执行测试；错误即停止，不留下“已完成”的假成功标记。首次运行可加 `--dry-run` 查看全部命令。

输出包括 `results/<name>/manifest.json`、每项日志、`summary.json`，以及 `results/<name>_<case>/files/metrics.json` 中每次随机运行的逐序列指标。主表先在每次运行内对主序列等权平均，再对各次运行算均值及总体标准差；不能平均各序列的标准差。轨迹文件仅对应最后一次决策运行，所有运行的指标保存在 metrics.json。

复现历史 checkpoint 应与 `benchmarks/evidence/ground_summary.json` 对照。不要因测试结果不理想而调整阈值、筛选 checkpoint、改变随机种子或从表中删掉失败序列。

## 比较对象与结论门槛

当前主线比较为完整 PCM3 与只去掉 `L_x` 的 PCM3 nox。`bmr_g` 是旧直接融合的历史参照；它和旧 PCM+ 使用自己的冻结快照与已有指标，不受当前 v3 加载器支持。已有地面对照均为小样本种子结果，新模型正式结论需补齐训练种子。

继续沿用项目的实际差异门槛：小于 10% 的差异按实际打平处理，**这不是统计显著性检验**。报告所有六项、旋转指标和遮挡序列，不能仅凭全视觉 t_rel 改善宣称整体更鲁棒或更高效。

机制目标沿用此前标准：无纹理视觉平移增益比 ≤0.2，视觉标准差比 ≥3；冻结视觉 pi_trans ≤0.5；验证集方差—误差 Spearman ≥0.4、控制调用步数后 ≥0.2；替换 IMU 后先验前向 R² 下降 ≥0.2；全部拒绝视觉后逐帧 t_rmse 上升 ≥20%；building2 停车净漂移 ≤0.9 米。它们是诊断门槛，不能替代主表精度。

## 效率报告

视觉调用率不能替代整模型耗时。必须在同一 GPU、相同输入与 batch=1 下计入 IMU、先验、策略、融合和 LSTM，分别报告全视觉与实际跳过视觉的序列耗时；数据读取开销是否计入必须写清楚。

`tools/profile_cost.py` 是合成输入的初步核算工具。它统计 Conv/Linear/LSTM MAC 和强制调用/跳过的耗时，**强制路径没有执行策略网络**，也未统计所有逐元素操作、真实数据读取；因此结果只作组件诊断，不应直接作为端到端效率结论。当前交接没有给出完整效率胜出的结论。
