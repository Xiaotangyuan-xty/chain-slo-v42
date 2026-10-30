# chain-slo-v42

v42 链级 SLO 调度研究的私有冻结代码快照。代码全部从 Mac 本地归档并推送，未从服务器上传。

冻结标签：`v42-frozen-2026-10-30`。冻结定义见 [FREEZE_V42.json](FREEZE_V42.json)。

## 结果与证据

完整官方 JITServe 对照：官方缩放 SLO 达标链数 **B=3/6、C=4/6**，新增链 1，保住链 2、3、5；25 项独立检查通过，47 个请求全部完成。

C 平均 JCT 310.53 秒，B 为 267.70 秒，C 高约 16%。这是冻结六链 DEV 单次结果，不是统计显著性或普遍优势。

- [研究报告](docs/研究改进与GPU0实验报告_2026-10-30.md)
- [CPU V44 验收](stage7_dynamic/receipts/CPU_QUALIFICATION_V44.json)
- [第 42 组有效比较](stage7_dynamic/receipts/FULL_DYNAMIC_COMPARISON_V3_FULL_B_GPU0_42_V3_FULL_C_GPU0_42.json)
- [最终 GPU0 与源码审计](evidence/FINAL_GPU0_AND_SOURCE_AUDIT.json)

## 目录

- `stage7_dynamic/`：v42 全部机制源码、共享 backend、预测与 planner、CPU/runtime 检查、配置和资产。70 个验收文件逐字节保留。
- `programs/`、其他 `stage*/`：Mac 上现有的运行支持及历史阶段源码；当前入口以 `stage7_dynamic` 为准。
- `repo/`：官方 JITServe commit `84584a119b892cf90c65e2b3f5bac67250d39d27`，含 vendored vLLM、原始许可证，以及本实验使用的两份公开 DeepResearch 输入。官方 1297 个源码文件 SHA 全部保留。
- `manifests/`：官方源码 SHA 清单。
- `evidence/`、`docs/`：轻量收据、阶段分析和研究记录。

## 冻结验证

```bash
python3 verify_freeze.py
```

该命令只核验归档文件 SHA、70 个验收文件及官方 1297 个源码文件，不启动 GPU。

## 运行环境边界

归档保留了测试时源码中的服务器绝对路径，不对已验收代码进行路径重写。它是完整代码快照，不是包含模型和 CUDA 环境的独立安装包。

模型权重、QRF pickle、外部 PERFORMANCE/PREDICTOR 资产、Python/CUDA/native binary 环境及大型原始运行 trace 仍是外部依赖，见 `FREEZE_V42.json`。新环境运行前需按原冻结契约配置依赖、`repo` 路径和 runtime 布局，随后重新 CPU 验收。

根目录 `FREEZE.json` 与 `PREPARE.json` 是原部署历史文件，含早期阶段/GPU2 元数据；当前版本定义为 `FREEZE_V42.json` 与 `stage7_dynamic/FULL_FREEZE.json`。实际 v42 controller 强制检查物理 GPU0 UUID、空闲状态及 CPU 资格 SHA。不得直接运行历史阶段的旧 controller。

## 版本维护

冻结标签不移动。后续修改应在新分支/新版本中进行，并重新通过 CPU 与所需实验验收。主仓库未包含 `.env`、认证文件、模型权重、编译缓存或大型运行日志。

第三方源码许可证保留在 `repo/LICENSE` 及 `repo/third_party/vllm/`；此归档未额外变更原创代码的授权条款。
