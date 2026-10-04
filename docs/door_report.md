# Panda Door 第二阶段报告索引

- [环境与 GT 接口](door_environment_report.md)
- [1000 条物理仿真专家轨迹](door_dataset_report.md)
- [A/B/C/D 多种子在线 RL 结果](door_rl_report.md)
- [未来 LWD/DIVL 接口准备](lwd_preparation.md)

实验数据与图位于 `results/door/`，Door checkpoint 位于 `checkpoints/door/`，TensorBoard 与运行日志位于 `logs/door/`。`configs/door_experiment.json` 固定主实验参数；`experiments/door/launch_full.sh` 是 8 GPU 调度入口。所有 Door 输出与已完成的 Drawer 目录分开。

复现入口：在项目根目录先 `source configs/runtime_env.sh`，用 `door_env/author_asset.py` 生成自定义 USD，`door_dataset/collect.py` 采集示教，`door_dataset/validate.py` 审计，再运行 `bash experiments/door/launch_full.sh`。全部退出码为 0 后执行 `experiments/door/analyze.py`，最后用 `experiments/door/launch_heldout.sh` 和 `experiments/door/analyze_heldout.py` 做独立复测。运行参数与 GPU 分配以脚本和配置清单为准。
