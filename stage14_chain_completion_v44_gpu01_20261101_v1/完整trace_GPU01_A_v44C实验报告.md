# GPU0/GPU1 完整公开 trace：原生 JITServe A / v44 C

状态：COMPLETED。统一离线对照有效：True。C 的达标链数严格多于 A。

| 指标 | 原生 A | 新机制 v44 C |
|---|---:|---:|
| 实际下发 SLO 达标链数 | 7/200 | 45/200 |
| 严格未缩放达标链数 | 1/200 | 14/200 |
| 平均 JCT，秒 | 2503.815 | 2855.403 |
| JCT P95，秒 | 5099.709 | 5212.755 |
| 主负载排空时长，秒 | 5616.387 | 5692.153 |
| 达标链请求 goodput，req/s | 0.008012 | 0.036542 |
| 主负载平均实际 batch | 3.576 | 3.588 |
| 实际最大 batch（含热身） | 32 | 32 |
| 主负载执行器等待墙钟，秒 | 4734.111 | 4702.787 |

完整输入为原顺序全部 200 条链、1197 个主请求，每臂另有相同7请求热身；主目标输出504695 tokens。A使用物理GPU0、C使用物理GPU1（同型号A40，顺序运行），BF16 Qwen2.5-7B-Instruct、原生vLLM执行器、GPU512/CPU1170 KV块、最大32序列、每步2048 tokens、模型长度8192、1GiB swap。v43冻结及既有stage13完整保留；v44使用独立训练图、完成密度、短期KV预留与执行量保护。A启动器与stage13逐字节一致，A调度语义不变。

到达间隔沿用seed42的1链/秒公开生成规则，两臂使用同一累计绝对计划（最后一链计划199.780秒）。仅将HTTP传输等待和控制器墙钟上限扩展至27000/28800秒，实际SLO不变。SLO以客户端收到的值乘实际阶段数，未缩放20×阶段数另列；没有为结果改链、deadline、seed、负载或资源。

达标重合：{'neither': 152, 'A_only': 3, 'both': 4, 'C_only': 41}。200条链逐条JCT和达标结果见FULL_TRACE_SUMMARY.json。执行器等待墙钟不是CUDA event时间，批次内各请求时间重叠。

完整输出token序列：386/1204个请求逐序列一致。该比例不等于逐token一致率。本次保存了服务器端真实输入/输出token数组，并在TOKEN_COMPARISON.json记录每个不一致请求的首次分叉位置与token ID。未记录logits，不据此断言差异原因或语义质量相同。

源码审计：PASS，1405个基础文件及22个本轮文件；最终KV：{'A': {'gpu_blocks': 512, 'cpu_blocks': 1170}, 'C': {'gpu_blocks': 512, 'cpu_blocks': 1170}}。两卡最终状态：{'0': {'gpu': '0, GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938, 0 MiB', 'processes': ''}, '1': {'gpu': '1, GPU-f5874940-7f3b-ad3e-b810-bc7dda423d1a, 0 MiB', 'processes': ''}}。端口释放：{'18080': True, '65433': True}。

C准入执行对齐记录见FINAL_AUDIT.json，集合外执行调用数为0。v44在CPU换出池不足时使用原生重算；集合外执行如出现仍单独报告。

这是完整公开trace的一次固定负载/seed对照，包含此前查看过的DEV前缀；不属于未接触过的独立HOLDOUT，也不证明泛化优势或真实agent语义等价。所有迟到请求均按原机制排空，不能只截取达标前缀解释结果。

原始client_calls、common/server事件、reservation事件和engine_token_ids保留在服务器：
`/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1/stage14_chain_completion_v44_gpu01_20261101_v1/attempts/`

本报告和轻量证据不会改动GitHub冻结标签或既有论文初稿。
