# PR Description — sparse_mla_fwd（草稿：分支 feat/sparse-mla-fwd-pr @ cfa8828c，PR 未创建）

为 Ascend NPU 新增 `sparse_mla_fwd`（DSA 稀疏 MLA 前向：varlen packed 随机行 gather + 在线 softmax + QK/PV 双 GEMM 融合）算子。迁移自主仓 GPU 实现（`ref_gpu/sparse_mla_fwd.py`），交付于 `examples/dsa_sparse_finetune/`（kernel 1286 行 + test 179 行）。

- **Kernel 架构**：双 kernel（fast v2 特化 + portable 通用兜底）+ host 分派（DESIGN §11.4 分派树）。Developer 模式（用户显式指定），`T.Kernel(grid, is_npu=True) as (cid, vid)` MIX_AIC_1_2 形态，`T.symbolic` 动态维。fast v2 支持域 gate：`H=128 ∧ G=1 ∧ (dim,tail)=(512,64) ∧ topk%256==0 ∧ NI∈[1,6]`，域外任一不满足自动走 portable。
- **计算结构**：对每个 packed 行 `s_row`（batch b、行内位置 s_i）、每个 kv_group g：按 `Indices[s_row, g, j]` gather KV 行，`score[h,j] = (Q[s_row,h,:512]·KV_main[idx,:512] + Q[s_row,h,512:576]·KV[idx,512:576]) × sm_scale`；`mask[j] = (idx ≤ s_i) ∧ (idx ≠ -1)`（causal + 无效槽；重复索引不去重、重复计贡献——主仓 kernel 语义）；在线 softmax（m_i/sumexp/alpha 流式更新），`O = P @ V`，`Lse[s_row,h] = log(sumexp) + m_i`；`sm_scale` 缺省 `(dim+tail_dim)^-0.5`。
- **数据流与内存管理**：两遍法（a3）：BI=256 / NI=4 / M=64×2 / `ws_kv` DEPTH=NI+1=5 环。V 侧随机行 gather（MTE）→ GM `ws_kv` 环形暂存 → C 侧 `gemm_v0`（bf16 A/B + fp32 C，transpose_B，BufferRegion 切片）QK/PV → mask/在线 softmax 向量链（UB）→ 输出。API 全部有 ascendc/pto 双后端先例证据（DESIGN §3 映射表逐条附出处）。
- **关键优化策略**：（1）`ws_kv` 多槽环形暂存重叠 V 侧 gather 与 C 侧 GEMM；（2）gather 批量 staging（整 2D 写替代 320 微写/块/AIV，探针实测 61.14→17.11 μs/块/AIV，3.6×）；（3）C/V 跨块软件流水（ws_kv/ws_s/ws_p/ws_o ring 维，V 领先产出、C 消费上一块）；（4）实测证据审查驱动重定标（evidence audit：逐项访存量三栏推导 + 同会话 A/B + 交错轮测中位，机器态双峰 Δ18% 显式记录）。
- **数值精度处理**：bf16 GEMM + fp32 累加（L0C）；在线 softmax 全 fp32 向量链；golden 的 scatter mask 为去重语义、主仓为重复语义，无重复输入上两者一致（L0 输入 = 主仓 randperm 构造，语义有效；重复语义验证属 L1）。
- **硬件与实现约束**：`heads % kv_group == 0` 且 head_kv ∈ {16..64 的 2 的幂} ∪ {64 的倍数}（主仓 assert 原样）；dim/tail_dim 2 的幂；`seq_len ≥ topk`；fast v2 域界 gate（上述五条件）。
- **测试验证**：L0 4 用例（主/小 shape × ascendc/pto 双后端，输入构造与主仓 `test_sparse_mla_fwd` L274-286 逐行一致）；L1（重复索引语义 / -1 与越界 / 多段非整除）、L2（host 域正确拒绝）、Boundary 由分层扩展，`--level all` 全量绿在案（iter-6 回滚后）。

## 精度验证 ✅

**测试配置**：L0 主 shape B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16, offsets=[0, 2047, 4096]（主仓 `__main__` 配置复刻，block_I=128）；小 shape 1/2050/128/1/576/512/1024, offsets=[0, 1024, 2050]（段长非 topk 整除）。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0（主 shape，ascendc / pto 双后端） | 2 | 2 | ✅ PASS | 混合容差双门限通过 | — |
| L0（小 shape，ascendc / pto 双后端） | 2 | 2 | ✅ PASS | 混合容差双门限通过 | — |
| L1/L2/Boundary 分层扩展（--level all） | — | — | ✅ 全量绿 | （明细见开发档案 perf_log） | — |

- **精度阈值**：out 用主仓判据 `assert_tensors_similar(eps=1e-2)`；其余输出混合容差双门限（atol / rtol / max_abs_error_limit / required_matched_ratio，按 dtype 派生，见 test 内 `check_precision`）
- **通过条件**：双门限 AND + `raise_assert` 硬失败
- **参考实现**：主仓 PyTorch 参考（`ref_gpu/sparse_mla_fwd.py` 语义）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：主 shape B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16；fast v2 路径

| **性能指标** | **测试结果** |
| --- | --- |
| fast 纯 kernel（NPU Event，warmup 5 + 20 轮，NPU 空闲串行） | **46667 μs**（交错轮测中位；同会话复测 52321 μs，前日 A/B 55258±296 μs——机器态双峰 Δ18%，见下） |
| portable 纯 kernel（不特化） | 64000-70000 μs |
| 实测聚合有效带宽 | 0.74 TB/s（38.7 GB / 52.321 ms，可达性下界锚点） |
| Block Dim | fast GRID=20 |
| Op Type | mix（MIX_AIC_1_2） |

> 同型机器存在性能双峰态（Δ18%），以交错轮测中位为主口径并显式记录波动带；调优轨迹：Stage 2 基线 138551±205 μs → 终态（iter-5 采纳版，冻结为实现层最优参考基线）中位 46667 μs（**-66.3%**）。iter-6 曾因负收益回滚，全量精度门禁复验绿。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次 PR，fast v2，中位口径） | **46667 μs** | — |
| DESIGN §9.4 哨兵目标 | < 24288 μs | **未达标，1.92×**（如实披露） |

- 瓶颈（实测证据，见 `perf_tuning/sparse_mla_fwd_evidence_audit.md`）：V 侧微事务发射结构 + 流量地板——当前架构单次调用搬运 38.7 GB，聚合有效带宽实测 0.74 TB/s；L2 ~1 TB/s 为 DESIGN 估算口径（非实测）。
- 已推导但未实施的 rev2 方向（a+c：L0C 驻留 acc_o + fp16 UB→L1 直连）：推导流量 14.3 GB / 0.59 TB/s，两侧首次同时低于目标线，需数据通路级重构（[DESIGN_ERROR] 重定标结论），本 PR 冻结 iter-5 为实现层最优参考基线，rev2 后作对照。
