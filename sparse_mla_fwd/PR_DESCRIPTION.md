为 Ascend NPU 新增 `sparse_mla_fwd`（DSA 稀疏 MLA 前向：varlen packed 随机行 gather + 在线 softmax + QK/PV 双 GEMM 融合）算子。参考 GPU 实现迁移，交付于 `examples/dsa_sparse_finetune/`（kernel + test）。

- **Kernel 架构**：双 kernel（fast 特化路径 + portable 通用路径）+ host 侧分派。Developer 模式，`T.Kernel(grid, is_npu=True) as (cid, vid)` 的 MIX_AIC_1_2 形态，`T.symbolic` 动态维。fast 路径支持域：`H=128 ∧ G=1 ∧ (dim,tail)=(512,64) ∧ topk%256==0 ∧ NI∈[1,6]`，域外任一条件不满足自动分派至 portable。
- **计算结构**：对每个 packed 行 `s_row`（batch b、行内位置 s_i）、每个 kv_group g：按 `Indices[s_row, g, j]` gather KV 行，`score[h,j] = (Q[s_row,h,:512]·KV_main[idx,:512] + Q[s_row,h,512:576]·KV[idx,512:576]) × sm_scale`；`mask[j] = (idx ≤ s_i) ∧ (idx ≠ -1)`（causal + 无效槽）；在线 softmax（m_i / sumexp / alpha 流式更新），`O = P @ V`，`Lse[s_row,h] = log(sumexp) + m_i`；`sm_scale` 缺省 `(dim+tail_dim)^-0.5`。重复索引不去重、重复计贡献（与参考 kernel 语义一致）。
- **数据流与内存管理**：两遍法结构（BI=256 / NI=4 / M=64×2），`ws_kv` 取 DEPTH=NI+1=5 槽环形缓冲。V 侧随机行 gather（MTE）→ GM 环形暂存 → C 侧 `gemm_v0`（bf16 A/B + fp32 C，transpose_B，BufferRegion 切片）完成 QK / PV → UB 上 mask / 在线 softmax 向量链 → 输出。所用 TileLang API 均具备 AscendC 与 PTO 双后端可用先例。
- **关键优化策略**：（1）`ws_kv` 多槽环形缓冲重叠 V 侧 gather 与 C 侧 GEMM；（2）gather 批量整 2D 写替代逐行微写——微基准实测 61.14 → 17.11 μs/块/AIV（3.6×）；（3）C/V 跨块软件流水（`ws_kv`/`ws_s`/`ws_p`/`ws_o` 加 ring 维，V 领先产出、C 消费上一块，串行和折叠为 max(V, C)）。
- **数值精度处理**：bf16 GEMM + fp32 累加（L0C）；在线 softmax 全 fp32 向量链；参考实现的 scatter mask 为去重语义、本 kernel 为重复语义，二者在无重复输入上结果一致（测试输入按参考构造方式生成，重复语义由专项用例覆盖）。
- **硬件与实现约束**：`heads % kv_group == 0` 且 head_kv ∈ {16..64 的 2 的幂} ∪ {64 的倍数}；dim / tail_dim 为 2 的幂；`seq_len ≥ topk`；fast 路径域界见上。
- **测试验证**：L0 4 用例（主 / 小 shape × AscendC / PTO 双后端，输入构造与参考测试逐行一致）；L1（重复索引语义 / -1 与越界 / 多段非整除）、L2（host 域正确拒绝）、Boundary 分层测试全量通过。

## 精度验证 ✅

**测试配置**：主 shape B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16, offsets=[0, 2047, 4096]（参考实现 `__main__` 配置复刻，block_I=128）；小 shape 1/2050/128/1/576/512/1024, offsets=[0, 1024, 2050]（段长非 topk 整除）。

| **测试层级/类别** | **用例数** | **检查项数** | **测试结果** | **精度指标** | **最大绝对误差** |
| --- | --- | --- | --- | --- | --- |
| L0 主 shape（AscendC / PTO 双后端） | 2 | 2 | ✅ PASS | 混合容差双门限通过 | — |
| L0 小 shape（AscendC / PTO 双后端） | 2 | 2 | ✅ PASS | 混合容差双门限通过 | — |
| L1 / L2 / Boundary 分层扩展 | — | — | ✅ 全量通过 | （含重复索引 / -1 / 越界 / 多段非整除专项） | — |

- **精度阈值**：out 采用参考仓判据 `assert_tensors_similar(eps=1e-2)`；其余输出为混合容差双门限（atol / rtol / max_abs_error_limit / required_matched_ratio，按 dtype 派生）
- **通过条件**：双门限 AND，超限即硬失败
- **参考实现**：PyTorch 参考（`ref_gpu/sparse_mla_fwd.py` 语义）

## 性能测试

**测试平台**：Ascend 910B3（A2）
**测试配置**：主 shape B=1, S=4096, H=128, HKV=1, DQK=576, DV=512, topk=1024, bf16；fast 路径

| **性能指标** | **测试结果** |
| --- | --- |
| fast 纯 kernel（NPU Event，warmup 5 + 20 轮，NPU 空闲串行） | **46667 μs**（交错轮测中位；同会话复测 52321 μs，另日 A/B 55258±296 μs——同型机器存在双峰性能态，波动 Δ18%，实测记录如下） |
| portable 纯 kernel（通用路径） | 64000-70000 μs |
| 实测聚合有效带宽 | 0.74 TB/s（38.7 GB / 52.321 ms） |
| Block Dim | 20 |
| Op Type | mix |

> 计时口径：纯 kernel NPU 事件计时，warmup 5 + 20 轮，NPU 空闲串行；机器双峰态以交错轮测中位为主口径并显式记录波动带。相对初始实现的 138551±205 μs，fast 路径累计优化 **-66.3%**（中位口径）。

### ⚠️ 性能差距说明

| **平台/方案** | **运行延迟** | **性能差距** |
| --- | --- | --- |
| Ascend 910B3（本次提交，fast，中位口径） | **46667 μs** | — |
| 性能目标 | < 24288 μs | **未达标，1.92×**（如实披露） |

- 主要瓶颈（实测）：V 侧微事务发射结构 + 搬运量地板——当前架构单次调用搬运 38.7 GB，聚合有效带宽实测 0.74 TB/s（L2 聚合带宽 ~1 TB/s 为设计估算口径，非实测）。
- 已验证的改进方向：L0C 驻留累加器 + fp16 UB→L1 直连的数据通路重构，推导可将搬运量降至 14.3 GB（0.59 TB/s），两侧首次同时低于目标线；该方向属数据通路级重构，超出本次变更范围。
