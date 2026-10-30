# fla_chunk_o 踩坑记录（Stage 2 debug log）

> 沙箱：`examples_experiment/dsa_sparse_finetune/`。仅记录 fla_chunk_o（KDA chunked 前向输出 O），
> 禁止与其他算子共写。语义权威 `ref_gpu/fla_chunk_o.py`（只读）；计时口径 `ref_gpu/fla_chunk_o_tilelang_fwd.py`。

---

## Attempt 1 — 2026-10-22T10:41:39Z (first_impl 会话汇总)

- mode: first_impl
- classification: precision_pass
- fail_category: none（会话内 3 次迭代失败均已定位修复，见下方迭代记录）
- test_level: all（L0 → 扩展 L1/L2/Boundary → 覆盖门禁 → 全量二次校验 + perf 哨兵）
- coverage: L0:2 L1:9 L2:10 Boundary:5（Boundary 3 PASS + 2 WARN）
- boundary_warnings:
  - `bnd_inf_g`（稀疏 +inf 门控）[BOUNDARY_WARN]：golden 给 ±inf，kernel hi/lo 分裂产生
    `lo = inf - inf = NaN` → NaN/inf 结构错位。根因是精度回退方案（qg hi/lo 双 gemm）在 inf
    输入下的固有局限（语义权威 Triton 单 cast 路径无此问题）；非阻塞，已记录。
  - `bnd_g_dbound`（g=±120，2^120≈1.3e36 逼近 bf16 上限）[BOUNDARY_WARN]：ratio=0.9999
    （个别 |o|≪|o_inter| 消零元素超出逐元素容差）+ max_abs=2.6e33（超出 1e0 绝对硬帽——该量级
    本身超出精度标准可认证域，见迭代 3）。非阻塞，已记录。
- changes:
  - 新建 `fla_chunk_o.py`：Developer 模式单 kernel（`fla_chunk_o_kernel`，threads=1，
    PASS_CONFIGS 四件套，out_idx=[6] 绝对索引）+ host wrapper `fla_chunk_o`（_check_shape /
    _pick_block_dk / _pick_block_dv / Msk per-(device,chunk) 缓存）。
  - 新建 `test_fla_chunk_o.py`：golden（DESIGN §9.1 + intra einsum 轴修复）+ check_precision
    （precision-standard §4.1 原样）+ L0×2 + L1×9 + L2×10 + Boundary×5 + perf_bench 哨兵 +
    --level {l0,l1,l2,boundary,perf,all} main 分发器 + COVERAGE_CATEGORY/MANIFEST。
  - kernel 迭代：单 cast qg → hi/lo 双 gemm（DESIGN §9.3 预授权回退，见迭代 2）。
- error_summary: 会话内 3 次失败（迭代 1/2/3），全部定位到实现/测试层根因并修复；无设计层错误。
- design_error_reason: none
- rollback: no
- backup_path: history_version/fla_chunk_o_impl_s2_attempt1.py（hi/lo 修改前的单 cast 版本）
- instrumentation_cleaned: n/a（未使用 T.printf/T.dump_tensor 插桩；诊断全部通过 CPU mimic
  golden 对照完成，无内核插桩残留）
- next_hint:
  - **性能未达标交 Stage 3**：perf_bench (1,8192,64,128,128) 实测 3.263 ms（rep 稳态 ~3.20 ms），
    目标 < 1.320 ms（超 2.5×；IO 下限 797us → 带宽效率 ~25%）。调优空间：threads∈{1,2}（R9）、
    block_DV=64+L2 复用变体（§10.4-A）、num_stages 微调；关注 4D bshd 切片行间 gap 的 DMA 效率
    （q/g/A 每块 64 行 × 尾维连续小行，DataCopyPad 粒度可能是瓶颈）。
  - hi/lo 双 gemm 使 inter gemm 次数 ×2（3→5 组/block），Cube 时间仍非瓶颈（<带宽下限 21%）。
  - DESIGN §11.3 的 example_ 前缀命名（example_fla_chunk_o.py）与本次调度指令不一致；
    按调度指令采用 fla_chunk_o.py / test_fla_chunk_o.py，Stage 3/上库时注意统一。

### 迭代 1 — out_idx=[-1] 触发 IndexError（运行失败，实现层）

- 分类：runtime_fail / fail_category=other（JIT 适配器参数解析）
- 现象：`kernel(q, v, g, A, h, msk)` 抛 `IndexError: list index out of range`（两 L0 用例同报）。
- 根因：`AUTO_CV_COMBINE` 会在 kernel 签名**末尾追加**自动 GM workspace 参数；adapter 的
  `_legalize_auto_memory_idx` 按注入后的完整参数表解析负索引 → `out_idx=[-1]` 指向最后一个
  自动注入参数，O（声明序第 7 个参数）被当作输入，6 个实参不够 → IndexError。
- 受控验证：读 `tilelang/jit/adapter/base.py:22-42`（负索引 = len(params)+idx）+
  `tilelang/jit/kernel.py:249`（auto_gm_indices 来自 pass 注入）确认机制。
- 修复：`out_idx=[6]` 绝对索引（★fla_wy_fast impl_final docstring 第 7 条同款——
  "out_idx uses absolute indices (auto-injected GM workspace params append to the kernel
  signature and shift negative indices)"）。DESIGN §4.7 伪代码的 `out_idx=[-1]` 写法在这点上
  与同族交付实证冲突，按实证修正（实现层，非设计错误）。
- 结果：两 L0 用例跑通（shape/dtype 正确）。

### 迭代 2 — L0 精度 0.988 < 0.99（精度失败，实现层，DESIGN 预授权回退）

- 分类：precision_fail / fail_category=other（数值精度）
- 现象：l0_min ratio=0.9883 / l0_repr ratio=0.9866（均 < 0.99），max_abs=0.125（≤1e0 硬帽内）。
  修改前已按备份规则备份 → history_version/fla_chunk_o_impl_s2_attempt1.py。
- 定位（精度调试方法学）：
  1. 量化：两用例均为"通过率不足、硬帽未超"型——大面积轻微偏差。
  2. 受控对照（CPU mimic golden，无内核插桩）：`mimic_cast`（qg 先 cast bf16 再 einsum）精确复现
     kernel 失败（ratio=0.9883、max_abs=1.250e-01、191 mismatch，与 kernel 逐位一致）；
     `mimic_hilo`（hi/lo 分裂双 einsum）ratio=1.0000 / 0 mismatch。
  3. 根因：qg 单次 bf16 cast 的 ~2^-9 相对误差落在 o_inter 上；消零元素（|o| ≪ |o_inter|，
     mismatch 元素中位 |o|=0.069 vs 整体中位 3.47，76% 满足 |o|<|o_inter|/4）逐元素容差退化为
     atol=2^-10 → ~1.2% 元素超差。DESIGN §9.3 误差预算只算了典型元素（余量 9×），漏了消零尾部。
     顺带实证：kernel 误差与 mimic_cast 完全一致 → R1（epilogue C→V 单跳）/R4（循环后 GM→L1）/
     R5（5D h 切片）链路全部正确（否则误差会超出 mimic）。
- 修复：DESIGN §9.3 注 + 被否方案 F 预授权的回退——qg hi/lo 双 gemm（★impl_final fwd_u 同款）：
  `qg_hi=bf16(qg)`、`qg_lo=bf16(qg-hi)`，`gemm_v0(hi,h,o_frag,init=(i_k==0))` +
  `gemm_v0(lo,h,o_frag,init=False)`，相对误差 2^-9 → ~2^-18。
- 结果：L0 两用例 [PRECISION_PASS] ratio=1.0000（l0_min max_abs=1.562e-02 / l0_repr 1.250e-01）。
- 回滚判定：保留修改（精度通过）。

### 迭代 3 — L1 vrange_l 超绝对硬帽（精度失败，测试标定问题）

- 分类：precision_fail / fail_category=other（测试值域标定 vs 精度标准可认证域）
- 现象：l1_vrange_l（初版 spread=1e4/g×8，|o|~2.4e18）ratio=1.0000 但 max_abs=2.749e+11 > 1e0
  硬帽 → FAIL。逐元素相对误差 ~2.7e-4（远小于 rtol），纯粹是绝对硬帽不达标。
- 根因：bf16 精度标准的 max_abs_error_limit=1e0 硬帽隐含输出量级约束——ulp(|o|)>1e0（即 |o|≳256）
  时任何非逐位一致实现都无法通过双门限（输出 bf16 量化本身 1 ulp 就超帽）。初版值域超出了
  标准可认证域，属测试用例标定问题，非 kernel 缺陷。
- 受控验证：CPU 标定（seed 2107）：spread 1e4/g8 → max|o|=2.4e18；2/g2 → 580（边缘）；
  1.5/g1.5 → 69（ulp≤0.5，2× 余量）；自然 randn → 28.5。
- 修复：`_xfn_large` 重标定为 ×1.5/g×1.5（max|o|≈69，自然值的 2.4×），注释记录标定依据；
  超认证域行为由 Boundary DBOUND WARN（迭代外记录）如实记录。
- 结果：l1_vrange_l [PRECISION_PASS] ratio=1.0000 max_abs=6.250e-02。

### 附：golden 修复记录（DESIGN §9.1 einsum 轴 bug，实现层落地修正）

DESIGN §9.1 的 intra einsum `"bcij,bcjd->bcid"` 仅在 H==chunk_size 时可绑定、其 `Ac * M`
广播仅在 H==64 时对齐（L0 用例 H=2/4 会直接报错）。按语义权威（Triton `m_s[t,j]=t>=j` 掩码 +
`dot(where(m_s,A,0),v)`）修复为 `einsum("bthj,bjhd->bthd", Ac * M[None,:,None,:], vc)`
（同族 fla_wy_fast 交付对 DESIGN §9.1 einsum 轴 bug 的同款处理先例）。

### 附：文件命名偏差

DESIGN §11.3 规定 `example_fla_chunk_o.py` / `test_example_fla_chunk_o.py`（example_ 前缀，
对齐 example_indexer_bwd.py 先例）；本次调度指令明确要求 `fla_chunk_o.py` / `test_fla_chunk_o.py`。
按调度指令执行（调度方为权威）；上库/Stage 3 时如需对齐 DESIGN §11.3 再重命名。

### 附：L2 dtype 拒绝路径修正（Host Buffer 审计）

初版 L2 dtype 用例对 make_inputs 返回的 NPU 张量做 `.to(fp16)`（NPU 侧 aclnn cast，违反
"数据准备在 CPU 完成后单次 H2D"约定）；修正为 CPU 侧带 dtype 构造 + 单次 H2D。
经验证：错误 dtype 由 JIT adapter 以 `ValueError: Buffer dtype mismatch` 拒绝，无需 wrapper 加
dtype 断言。

### 附：perf 哨兵实测（DESIGN §9.4，非阻塞，交 Stage 3）

- shape (1, 8192, 64, 128, 128)，NPU 事件 warmup=10 / rep=10 均值（对齐上游 do_bench）。
- 实测 **3.263 ms**（首 rep 3.837，稳态 3.19–3.21）；目标 < 1.320 ms → 超标 2.5×。
- 精度参考检查（非门禁）：ratio=1.0000 / max_abs=1.250e-01 / passed=True。
- 带宽效率 ≈ 1.007GB / 3.20ms ≈ 315 GB/s ≈ 实测上限 1263GB/s 的 25%（IO 下限 797us）。
- 未超 60s 单 case 预算（搬运路径功能可行）；按 DESIGN §11.4 第 6 条进入 Stage 3 调优。

---

## 覆盖门禁（coverage_check.py，2026-10-22）

```
== Coverage Matrix: fla_chunk_o ==   category=Fusion
19 PASS / 0 MISS / 0 N/A -> PASS (exit 0)
D-DTYPE-bf16:5 D-DTYPE-fp32:2 D-EXC-DTYPE:2 D-EXC-SHAPE:8 D-PARAM-chunk_size:1
D-PARAM-scale:1 D-SHAPE-ALIGNED:2 D-SHAPE-EDGE:1 D-SHAPE-PRIME:2 D-SHAPE-TAIL-1:1
D-SHAPE-TAIL-MID:2 D-SPECIAL-DBOUND:1 D-SPECIAL-INF:2 D-SPECIAL-NAN:1 D-SPECIAL-ZERO:1
D-VALRANGE-ASYM:1 D-VALRANGE-L:1 D-VALRANGE-M:1 D-VALRANGE-S:1
```

覆盖映射说明（Fusion 类无豁免维度，域外场景走 L2 拒绝路径 + 域内非规范值压力）：
- TAIL-1/TAIL-MID/PRIME：域内计算路径按契约无非整除（S%64==0、DK exact-2、DV 单块）→
  同时覆盖（a）域内非规范值（DK=96→block 48 非二次幂 K 分块、DV=48/80 非二次幂 N 宽度、
  NT=3/H=3 质数自由轴）与（b）规范域外 shape 的契约拒绝（L2：S=129/96/127）。
- D-PARAM-chunk_size：chunk_size 契约固定 64（block_S 语义），非默认取值唯一行为是拒绝
  （L2 l2_chunk32 验证）。
- dtype 组合固定（bf16 in/out + fp32 gate），无 dtype 变体空间；D-DTYPE-fp32 由每用例的
  fp32 门控输入行使。

## 最终全量结果（--level all，二次校验，2026-10-22）

```
[PRECISION_PASS] l0_min (1,128,2,64,64)     ratio=1.0000 max_abs=1.562e-02
[PRECISION_PASS] l0_repr (2,256,4,128,128)  ratio=1.0000 max_abs=1.250e-01
[PRECISION_PASS] l1_aligned_repr / l1_bench / l1_edge_min / l1_d96_dv48 /
                 l1_prime_axes / l1_vrange_s / l1_vrange_l / l1_vrange_asym /
                 l1_scale1                    全部 ratio=1.0000
[BOUNDARY_PASS]  l2 ×10（S%64≠0 ×3 / DK 32,80 / DV 100,144 / chunk32 / dtype ×2 全部正确拒绝）
[BOUNDARY_PASS]  bnd_zero / bnd_inf_v / bnd_nan_qv（结构比对精确一致）
[BOUNDARY_WARN]  bnd_inf_g / bnd_g_dbound（非阻塞，见 boundary_warnings）
Test Passed! (exit 0)
```

---

## PR #1953 评审修复（2026-10-30，ChaoyangJi 评审意见）

**问题**：乘法 mask 无法清除上三角 Inf/NaN（q=h=g=0、v=1、A 仅上三角 Inf 时，
正确输出应全零，但 Inf×0=NaN 污染输出；上游 Triton 用 tl.where 清零）。

**修复链**（example_fla_chunk_o.py + test 双侧）：
1. kernel：`T.tile.mul(a_f32, a_f32, msk_ub_fp32)` → `T.tile.select(a_f32[0:32, 0:64],
   msk_bits, a_f32[0:32, 0:64], 0.0, "VSEL_TENSOR_SCALAR_MODE")`，Msk 从 [64,64] fp32
   改为 [512] uint8 位打包（LSB-first，bit(t*64+j)=t>=j）
2. golden：`Am = Ac * M` → `torch.where(M_bool, Ac, 0)`
3. 测试门限：check_output 增加非有限位置结构比对；新增 bnd_a_upper_inf 回归用例
   （评审场景，max_abs=0.000e+00）

**踩坑（本轮最重要发现）——tile.select 的 vid 分裂缺口**：
- 直接替换后 l1_bench（唯一深流水 shape）FAIL：ratio=0.9994、max_abs 随运行变化
  （19.8→33.3）＝竞态表象；小 shape 全过
- 对照实验：mul 版+新 golden PASS → select 是唯一触发因子；独立 dst（候选 C）无效
- **根因（生成代码实证）**：vid-reduction pass 把二值算子（mul 等）改写为半量
  （repeat=32=2048 元素）+ buffer 半量分配（GetWithOffset(2048)），但 **tile.select
  不在改写集**：count 维持全量 4096。threads=2 下每个 vid 持有半量 buffer
  （[32,64]）与 256B 位图，全量 select 每个 vid 写 4096 元素 → **越界 8KB 砸掉
  per-vid 邻居 buffer**；深流水（136 batch）下被污染数据在重写前被读出 → 竞态
- **修复**：select 作用于半高 region `[0:BS//2, 0:BS]` —— count=2048 恰为每 vid
  有效行，与各 vid 本地 [0..2048) 元素 + 256B 位图完全对齐，零越界

**验证**：全量 27 用例（25 PASS + 2 既有 WARN）；l1_bench 三连跑稳定 ratio=1.0000、
max_abs=1.25e-01（与 mul 版一致）；性能 1.410-1.424ms（与修前 1.417ms 同带，V 侧
throttle-bound 下 select 与 mul 等价）；交付版 3 用例 + 冒烟全过；ruff/format clean。

**工具链结论**（上游反馈项）：① vid-reduction 不改写 tile.select（count 不减半，
dst 越界）；② VSEL selMask 要求 uint8 打包位图（LSB-first），不接受逐元素 0/1
fp32；③ tile.compare/select 共享位打包约定（compare 文档已明示，select 文档
selMask 描述不足）。
