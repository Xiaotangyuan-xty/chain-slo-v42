# chain-slo: v44 frozen snapshot

私有仓库 `Xiaotangyuan-xty/chain-slo-v42` 的 `v44` 分支及 `v44-frozen-2026-11-01` 标签。
全部代码与轻量证据从 Mac 本地整理、提交和 push，没有在实验服务器上上传 GitHub。

v44 根据整链按时完成收益与剩余工作量排序，使用独立训练图估计、短期 KV 预留及运行保护。
一次完整 200 链公开固定长度回放中，原生 A/GPU0 达标 7 条，v44 C/GPU1 达标 45 条；平均 JCT 增加 14.0%。
该 trace 已用于开发分析，未证明独立泛化；上一轮有效 A 为 16 条，本轮 A 为 7 条，跨轮波动尚待研究。
完整输出序列仅 386/1204 相同，不能推论语义等价。两臂全部请求完成，验收和独立重算通过。

- 冻结定义：`FREEZE_V44.json`
- 全部本轮源码、训练图模型资产、协议、CPU/GPU验收与报告：`stage14_chain_completion_v44_gpu01_20261101_v1/`
- 当前机制：该目录下 `programs/native_chain.py`、`chain_forecast.py`、`chain_admission.py`
- 结果分析：该目录下 `结果分析与独立核验_2026-11-01.md`

执行 `python3 verify_v44_freeze.py` 可在 CPU 核验全部快照与验收源码 SHA。
模型权重、冻结运行环境、QRF pickle、三份登记 SHA 的本地编译 .so、大型原始轨迹是外部资产，未上传。
本快照是完整代码归档，非独立安装包；历史绝对路径与源码保留。
源码内历史实验的运行授权属于当时实验，复现应新建 attempt、明确设备并重新验收，不能覆盖旧结果。
v42/v43 标签保留；旧版 verifier 请在相应旧标签下使用。本分支 README 已更新。
