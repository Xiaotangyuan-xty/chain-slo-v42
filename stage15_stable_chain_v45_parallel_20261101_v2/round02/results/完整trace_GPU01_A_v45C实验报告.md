# GPU0/GPU1 完整公开 trace：原生 JITServe A / v45 C

状态：COMPLETED。统一离线对照有效：True。C 的达标链数严格多于 A。

| 指标 | 原生 A | 新机制 v45 C |
|---|---:|---:|
| 实际下发 SLO 达标链数 | 9/200 | 44/200 |
| 严格未缩放达标链数 | 1/200 | 16/200 |
| 平均 JCT，秒 | 2423.653 | 3107.009 |
| JCT P95，秒 | 5032.907 | 6135.369 |
| 主负载排空时长，秒 | 5576.333 | 6804.038 |
| 达标链请求 goodput，req/s | 0.010042 | 0.029247 |
| 主负载平均实际 batch | 3.577 | 3.533 |
| 实际最大 batch（含热身） | 32 | 32 |
| 主负载执行器等待墙钟，秒 | 4755.334 | 5827.955 |

完整输入为原顺序全部 200 条链、1197 个主请求，每臂另有相同7请求热身；主目标输出504695 tokens。A使用物理GPU1、C使用物理GPU0（同型号A40，并行运行；独立预测器和端口、各56个互不重叠CPU线程），BF16 Qwen2.5-7B-Instruct、原生vLLM执行器、GPU512/CPU1170 KV块、最大32序列、每步2048 tokens、模型长度8192、1GiB swap。v43/v44冻结及既有stage13/14完整保留；v45将运行保护计入KV和序列预算，并按可见compound输入变化刷新预测邻居。A启动器与stage13逐字节一致，A调度语义不变。

到达间隔沿用seed42的1链/秒公开生成规则，两臂使用同一累计绝对计划（最后一链计划199.780秒）。仅将HTTP传输等待和控制器墙钟上限扩展至27000/28800秒，实际SLO不变。SLO以客户端收到的值乘实际阶段数，未缩放20×阶段数另列；没有为结果改链、deadline、seed、负载或资源。

达标重合：{'neither': 151, 'A_only': 5, 'both': 4, 'C_only': 40}。200条链逐条JCT和达标结果见FULL_TRACE_SUMMARY.json。执行器等待墙钟不是CUDA event时间，批次内各请求时间重叠。

完整输出token序列：386/1204个请求逐序列一致。该比例不等于逐token一致率。本次保存了服务器端真实输入/输出token数组，并在TOKEN_COMPARISON.json记录每个不一致请求的首次分叉位置与token ID。未记录logits，不据此断言差异原因或语义质量相同。

源码审计：PASS，1427个基础文件及25个本轮文件；最终KV：{'A': {'gpu_blocks': 512, 'cpu_blocks': 1170}, 'C': {'gpu_blocks': 512, 'cpu_blocks': 1170}}。两卡最终状态：{'0': {'gpu': '0, GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938, 0 MiB', 'processes': ''}, '1': {'gpu': '1, GPU-f5874940-7f3b-ad3e-b810-bc7dda423d1a, 0 MiB', 'processes': ''}}。端口释放：{'18080': True, '18081': True, '65433': True, '65434': True}。

C准入执行对齐记录见FINAL_AUDIT.json，集合外执行调用数为0。v45在CPU换出池不足时使用原生重算；集合外执行如出现仍单独报告。

这是完整公开trace的一次固定负载/seed对照，包含此前查看过的DEV前缀；不属于未接触过的独立HOLDOUT，也不证明泛化优势或真实agent语义等价。所有迟到请求均按原机制排空，不能只截取达标前缀解释结果。

原始client_calls、common/server事件、reservation事件和engine_token_ids保留在服务器：
`/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1/stage15_stable_chain_v45_parallel_20261101_v2/round02/attempts/`

本报告和轻量证据不会改动GitHub冻结标签或既有论文初稿。
