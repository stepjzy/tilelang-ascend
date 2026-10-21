# indexer_topk_reducesum 优化点清单（tilelang-perf-optimization Step 3）

算子类型（Step 2 判定）：**混合型**（msprof Op Type: mix；AIC cube/mte1/mte2/fix + AIV vec/mte2/mte3，
MIX_AIC_1_2）。基线画像见 `indexer_topk_reducesum_baseline_iter1.json`。

## Part A 优化点清单

[#1] V 侧向量链融合（参考: optimization-guide.md §向量链精简 / flash_attn_optimize.md "批量 Softmax"
     —— 以大 tile 替代小 tile 逐元素链）：**适用** — 生成代码证实每 token 13 条向量指令 + ~12 条
     PipeBarrier_V（RAW/WAW 同 pipe 依赖逐条插 barrier），4 token 串行；融合为 [QG2*H, BN] 大 op 后
     指令数与 barrier 数 ÷4，且不改变数值路径（elementwise 作用在不相交行块上，逐位等价）。
[#2] 因果 mask 仅在 straddle 块执行（参考: flash_attn_optimize.md 加性 mask / 块级掩蔽变体；
     gqa_sink 向量化 mask 先例）：**适用** — 每任务每 token 仅 ~1 个块跨越因果对角（k_st ≤ t_abs <
     k_st+BN），其余块要么全有效（免 mask 4 op）要么全无效（score_row 保持 fill 的 -inf，免整条链）。
     mask 形态沿用已验证的 [H, BN] CompareScalar/Select（compare 要求 src 256B 对齐：H*BN*4 = H≥8
     时 ≥1024B 且整除 ✓）。
[#3] sm_scale 移位到 per-task [CAP] 行乘（参考: optimization-guide.md §2.x op 削减 / indexer_bwd
     iter3 "s 折入"）：**适用（s>0 时）** — golden 顺序为 (Σ bf16(prod)) × s（sum 后乘），当前实现是
     Σ(bf16(prod)×s)（sum 前乘，靠容差吸收）。移到 topk_src [CAP]（8KB ≥ 256B ✓）后每任务省
     NBLK×[QG2H,BN] 次 Muls 且与 golden 逐位一致。s≤0 时（定义域内合法但无测试覆盖）保留 per-block
     融合 Muls 回退路径——sm_scale 是编译期常量，Python 分支零运行时成本。
[#4] score_2d [QG2, BN] 中转 + fill(-inf) + 融合 4 行 store（参考: vector-practices/
     vector_reduce_pass_fusion.md 归约遍数融合）：**适用** — 无效 token 行保持 -inf（topk pad 语义），
     store 从 4 次 [BN] 拷贝合并为 1 次 [QG2, BN] 带 dst 行间隔拷贝（copy_ub_to_ub 支持 dstGap）。
[#5] 转置 GEMM [BN, QG*H] + 宽 burst 中转（参考: optimization-guide.md 数据布局/DMA burst；
     performance-antipatterns 窄行 DMA）：**适用（iter2）** — 现中转读写均为 64B burst（workspace
     行宽 = BN=32 bf16）；转置后 relay 读变为 32×512B burst、fix 写 32×1KB burst；H 归约变 dim=-1
     连续 AR 模式。GEMM 数学恒等（同点积同 K 序），L0A/B 槽位 2KB/32KB 均满足 common.h:1282。
[#6] relay ping-pong + AIC 领先 2 对（参考: optimization-guide.md §Double Buffer / DB-ANALYSIS；
     indexer_bwd iter2 C/V 软件流水）：**适用（iter3）** — mte2 529us 与 vec 串行；[DB-ANALYSIS]
     Q1 循环内有 MTE3（ack 写）：是。Q2 无跨迭代累加器（relay 每块整读整弃）。Q3 同步方式：
     workspace 读写由 AUTO_CV_SYNC 按序配对 flag（1:1 语句配对约束保持），ack 背压从 wait(j-1)
     放宽到 wait(j-2)（AIC 领先 2 对，3 槽 workspace 仍用 2 槽 + V 读完成先于对应 ack 的次序证明
     安全）。风险：CrossCoreWaitFlag 阻塞 PIPE_S——read-ahead 必须放在链尾（读 j+1 在链 j 之后），
     且循环尾 PIPE_ALL barrier 会排空 mte2——iter3 实测验证，失败则回滚为 iter1/2 结构。
[#7] BN=64 变体（参考: DESIGN §9.4 路径②）：备选 — 中转对数减半、score 后置 op 达 256B 门槛；
     L0C [512,64] 恰满 128KB（indexer_bwd 教训 #5：恰满可用）；UB 需重排（w_2d/prod 融合缓冲
     [256,64] 超 192KB 预算——需半块融合）。iter4 视 iter1-3 结果决定。
[#8] host 侧 _CachedLauncher 式缓存（参考: optimization-guide.md §2.12 P0 Host 侧优化；indexer_bwd
     iter1 先例 1000us→13us）：**适用（次要目标）** — full interface 4776us vs 纯 kernel 2003us，
     差值 ~2770us 为 host 检测（.item() 同步）/ prepare_group_table Python 循环 / torch.zeros /
     launch 重编组（indexer_bwd 实测 Analyzer ~550us/次）。keyed-by-tensor-identity 缓存属通用
     优化。不影响主指标判定，iter5。
[#9] pass_configs 调整：不适用 — 4 True Developer 配置为 DESIGN §4.7 锁定，Stage 3 不回退架构。
[#10] T.Pipelined / num_stages：不适用 — ascend-constraints §1 动态边界不可流水（Stage 2 已论证）。

## [ORDER-PLAN] 实施顺序

1. [#1][#2][#3][#4] V 侧链融合 + straddle mask + scale 移位 + score_2d —— 前置依赖: 无 —— 单轮
   实施（同属 V 侧一个代码区域，共享 score_2d 结构，拆开无法独立验证精度），预期 vec 1015→~400us。
2. [#5] 转置 GEMM + 宽 burst —— 前置依赖: [#1]（链结构稳定后再换布局）—— 独立变量隔离。
3. [#6] 软件流水 —— 前置依赖: [#1]（vec 时间下降后 mte2 才是剩余主项）—— 失败可独立回滚。
4. [#7] BN=64 —— 前置依赖: [#5]（转置后 L0C/UB 形态不同）—— 视余量决定是否需要。
5. [#8] host 缓存 —— 独立，最后做（不阻塞主指标）。
