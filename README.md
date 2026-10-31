# chain-slo: v43 frozen snapshot

当前分支 `v43`，冻结标签 `v43-frozen-2026-10-31`。代码、配置和轻量证据从 Mac 本地归档并 push 到本私有仓库，未在实验服务器上上传 GitHub。

v43 使用原生 vLLM 后端，增加整链 deadline 和当前阶段 KV 准入。在物理 GPU1 的六链 DEV 对照中，原生 A 达标 **4/6**，C v43 达标 **5/6**；平均 JCT 为 158.4 / 206.4 秒。全部 47 个请求完成。新增链仅有 0.766 秒余量，完整 token 序列哈希只有 19/47 一致；这不是稳定泛化优势或逐 token 等价的证明。

- [冻结定义](FREEZE_V43.json)
- [实验报告](stage9_native_ac_gpu1_20261031_v1/原生AC轨迹诊断与v43改进报告_2026-10-31.md)
- [统一比较器回执](stage9_native_ac_gpu1_20261031_v1/receipts/AC_COMPARISON.json)
- [最终源码、设备和实际准入执行审计](stage9_native_ac_gpu1_20261031_v1/receipts/FINAL_AUDIT.json)
- 当前机制入口：`stage9_native_ac_gpu1_20261031_v1/programs/native_chain.py`、`deadline_reservation.py`、`c_server.py`。

验证本次冻结，不启动 GPU：

```bash
python3 verify_v43_freeze.py
```

归档包含 v43 全部 14 个验收文件（12 个 Python 源文件、2 个契约），以及 1369 个匹配基础文件：v42 70 个、官方 1297 个、原生 FlashAttention Python 包装 2 个。三份已登记 `.so` 为外部运行依赖，SHA 保存在冻结定义中。模型、CUDA/Python 环境、QRF pickle、大型原始 trace 没有打包为 Git 文件；因此这是完整代码快照，不是独立运行环境安装包。原绝对路径和验收源码未改写。

根目录的历史 `FREEZE.json` / `PREPARE.json` 含旧 GPU 元数据，不能据此运行旧 controller。本轮仅允许物理 GPU1，授权 UUID 和资源上限见 stage9 契约。复现实验应使用新 attempt 名称、重新 CPU 验收，并执行原 controller 的设备空闲和进程身份检查，不覆盖已有回执。

旧 v42 标签保持在原提交 `b2b86c50003b0e1c867bef10f68a6c63876342f4`。若要核验旧 v42 的完整历史快照，请 checkout `v42-frozen-2026-10-30` 后运行 `python3 verify_freeze.py`。旧 verifier 覆盖的 README 在本 v43 分支已更新，不能把本分支当成旧 v42 标签运行它。

原始 stage9 报告中“未上传新版本”是实验完成当时的记录；本次上传状态以冻结标签和 Mac 上传回执为准。历史失败的 CPU 回执留存供审计，不是当前通过回执。

后续 token 差异分析单独记录，不修改或移动本冻结标签。第三方许可证保留在 `repo/`。
