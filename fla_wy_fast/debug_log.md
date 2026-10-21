# debug_log — fla_wy_fast（KDA chunked WY 表示，fwd + bwd）

> Stage 2 开发踩坑记录。环境：tilelang 0.1.4+1d752539（ascendc_pto 基线）、CANN 9.2.0、
> 单卡 910B3。精度口径 precision-standard.md 混合容差；性能口径 do_bench（warmup 20 /
> rep 10 / 均值 / 仅设备侧）。

## Attempt 1 — 2026-10-21（环境修复 + API 探针）

- mode: first_impl（前置探针）
- classification: n/a（未到精度判定）

### 环境问题（阻塞所有编译）
- **现象**：任何 NPU kernel 编译报 `tl.transform.InjectTmpBuffer expects 1 arguments, but 2 were provided`。
- **根因**：主仓 Oct 21 checkout 回 `ascendc_pto`（HEAD=1d752539）后 `build/libtilelang_module.so`
  仍是 Oct 15 旧分支编译的 1 参数版本（`nm -D` 确认旧符号），Python 新代码与 .so 错位。
- **修复**：`cmake --build build -j16` 增量重建，符号变为 2 参数版，编译恢复。主仓只 build 未 commit。

### [B,S,H] beta 的 3D 中维切片搬运不可用（DESIGN API 误用，实现层修复）
- DESIGN §3.4 的 `T.copy(Beta[bb, s0:s1, bh], ub[64])` 数据错位：blockLen=1 元素（2B/4B，
  子 32B 粒度）+ dstStride≠0，MTE 不可靠；dst 改 [64,1] 仍错位。
- 上游 CUDA 的标量 Parallel 加载在 Ascend 编译期崩（AscendLowerParallelToVector）；
  `T.serial` 版 GM 标量 store 被编译器拒绝。→ CUDA 写法不可移植。
- **修复（bf16/fp32 均实测 PASS）**：`BetaFlat: [B, S*H]` 平铺视图（host 零拷贝 view）→
  连续整块 span 拷入 UB → `T.tile.arith_progression`（int32 字节偏移，bf16×2/fp32×4）→
  `T.reinterpretcast` uint32 → `T.tile.gather`。rope.py 同款链先例。
- IO 代价：载入整 chunk 全部 H 列（fwd bench +63MiB ≈ +6%；bwd +7MiB +0.5%）。
- **threads=2 陷阱**：gather 链在含 Cube 算子的 kernel 中被 vid 切分劈半而 gather 本体
  不可切分 → 越界读垃圾（纯 Vector kernel 中则侥幸通过）。全局改 threads=1 规避。

### dbeta [B,S,H] 反向 scatter：两处编译器 bug，GM scratch 中转绕过（实测 PASS）
- **bug A（地址重写）**：computed 来源的 [64]→[1,64,1] scatter，serial pack 被
  copy-propagation 消除后 scatter 被重定向，标记值实验证明落点错乱（cid=8 数据落
  bb=1 而非 bh=1）。threads=1 同败 → 非 vid 问题。GM→UB 直载的 [64,1] 源 + scatter
  （PB 模式）运行时正确（标记值逐位验证）。
- **bug B（GM 别名无同步）**：同 kernel 内写 Scratch 再读其视图 ScratchV，编译器不追踪
  参数别名 → MTE3 写与 MTE2 读无依赖 → 读到全零。显式 `T.set_flag("MTE3","MTE2",0);
  T.wait_flag("MTE3","MTE2",0)` 修复（R3 链实测 PASS）。
- 最终 dbeta 路径：acc[64] → GM scratch 写 → 显式 flag → [64,1] 视图载入 → PB scatter。
  代价 +2MiB IO（bwd bench 的 0.14%）。

### 循环内 reduce_sum(clear=False) 累加错误（工具链 bug）
- serial/Pipelined 循环内 `clear=False` merge 结果错（数值接近但有偏移）；unrolled 正确
  → 循环携带依赖 sync 缺失。仓库自带 reduce 测试不含该场景。
- **修复（实测 PASS）**：循环内 `clear=True` 归约到 r_part[64] + `T.tile.add` 累加
  （flash_attn sumexp 模式）。keepdim out [64,1] 的 reduce 同样错误，一并回避。

### 其他已验证约束
- `T.exp(内联算术表达式)` lowering 失败；裸 buffer 元素 / 整块 tile.exp 可用。
- 2D T.Parallel 列广播操作数 `gn_ub[j]` 不可用（vectorizer 崩/静默删 buffer）→ fwd kg
  改 serial 行级 tile ops（lightning_indexer 先例）。
- 4D `[1,64,1,blockD]` fold 搬运（仓库 test_fold_middle_scalar 同款）可靠，全部
  [B,S,H,D] 张量 IO 走此路径；gk 末行切片为连续 1D ✓。
- 跨 dtype T.copy cast（bf16↔fp32↔fp16）✓（gdn/flash_attn 先例）。

## Attempt 2 — 2026-10-22（kernel 结构收敛）

- mode: first_impl（结构迭代）
- classification: n/a（未到精度判定）

### C/V 融合结构约束（CrossCorePipeline / CombineCV）
1. **单 kernel 仅一个 cross-core Pipelined 体**（多则 `only support one cross core
   pipeline body` 崩）→ fwd 拆 u-kernel + wkg-kernel。
2. **Pipelined extent 必须 ≥2**（extent-1 codegen 崩：`Cannot find pre-allocated
   address for buffer: vb_l1`）。
3. **serial 循环内的 V→C 交换同步坏**（GM workspace 中转无正确 CrossCore flag）。
4. **Pipelined 内分支（if i_d < nv）仍走 GM workspace 且数据错** → 不用分支融合。
5. **prologue 的 V→C 交换被 pipeline pass 错误版本化**（C 侧按 stage 偏移读一次性写入的
   workspace → 第 2 迭代读垃圾）→ A 的 cast/拆分移入循环内（flash_attn 模式）。
6. **后置段多跳 C/V 交替（V→C→V→C→V）打破 sync-point 计数**（`Mismatch in sync
   points ... cube has 1, vec has 0/2`）→ bwd 的 dA 后处理拆成 z/da 两个单 gemm kernel；
   共享 staging buffer 也触发同类失配 → 每 split 独立 buffer。
7. **多 gemm 组共享 L1 操作数（角色互换复用）损坏 L0 staging**：bwd_k 中 dw 同时作
   dAp 左操作数与 X 右操作数 → dA_part 全错（X/K_bg/dw 输入均经 dg2/dk2/dbeta 旁路
   验证正确）。专用副本方案又触发 6 → **最终：每 kernel 单 gemm 组**，bwd 拆 6 个
   kernel（k1/k2/v1/v2/z/da）。
8. **Pipelined extent=3 时第 3 迭代损坏**（d96 用 block 32 → 3 迭代恰 2/3 正确，
  双缓冲回卷无同步保护）→ **强制恰好 2 次迭代**：block = D//2 ∈ {16,32,48,64}，
  即 D ∈ {32,64,96,128}；其余 16 倍数 D（16/48/80/...）显式拒绝（L2 验证）。
9. **out_idx 必须绝对索引**：workspace 自动注入追加参数，负索引错位。

### fp32 gemm_v0 是坏的（工具链级）
- 隔离测试：最简 fp32 matmul（threads=1/2、kL0Size 16-256）全部产出 1e37 级交替垃圾；
  仓库测试矩阵只覆盖 fp16（无 fp32 先例）。
- **修复：拆分模拟精度 gemm**（CPU mimic 按混合容差门验证）：
  - fwd：A 保持 bf16（精确操作数），V 侧乘积拆 bf16 hi/lo → 2 gemms（误差 ~2⁻¹⁶，
    bf16 门 2⁻⁶ 余量巨大）。
  - bwd（fp32 门 2⁻¹⁰）：全部操作数拆 fp16 hi/lo（hi@hi+hi@lo+lo@hi，L0C 内累加）→
    3 gemms，mimic ratio=1.0000。**已知限制：|操作数|>65504（fp16 范围）精度退化**
    （bnd_bwd_gk_extreme 如预期 WARN，非阻塞）；fwd 因 gemm 走 bf16 无此限制。
- L0C(fp32)→GM(bf16) 直存 fixpipe 转换坏（fp32 位流当 bf16 对写）→ fwd 输出经
  UB 中转 + 显式 cast。
- 早期版本 vb 预 cast bf16 的舍入噪声使小值输出相对误差超 bf16 门（ratio 94.7%），
  CPU mimic 精确复现（869/16384 逐位一致）→ 拆分方案消除。

### 行广播 scalar 索引 bug
- out-of-place 的 2D `T.Parallel` 行广播被 lower 成 `bt.GetValue(i*2)`（scalar 索引 ×2，
  fwd 的 in-place 版本正确）→ bwd 全部 bt 乘法改 serial 行级 `T.tile.mul(row, row, bt[i])`
  （lightning_indexer 先例）。

### golden 修复（DESIGN §9.1 两处 bug，语义不变）
- T1/T2 einsum 轴序：`"bjhi,bmhj->bihj"` 静默算错、`"bihj,bnjh->binh"` 直接崩；修正为
  `"bjhi,bjhm->bihm"` / `"bihm,bnhm->bihn"`（对逐 (b,h) 显式矩阵乘验证 + 上游 Triton
  的转置加载语义佐证：dA = Aᵀ@S@Aᵀ）。
- `torch.tril` 作用维度错误：[B,C,H,C] 上 tril 落在 (H,C) 而非 chunk 的 (C,C)——
  **kernel 的 msk 掩码一直是对的，golden 错了**（dA ratio 恰为结构零占比 0.5078 的
  佐证）。修正：permute 到 [b,h,m,j] 再 tril。

- changes: fla_wy_fast.py（6 kernel：fwd×2 + bwd×4，threads=1，fp16/bf16 拆分 gemm）、
  test_fla_wy_fast.py（golden 修复 + L0/L1/L2/Boundary）
- test_level: all
- coverage: L0:4 L1:4 L2:7 Boundary:6
- boundary_warnings: bnd_bwd_gk_extreme [BOUNDARY_WARN]（fp16 拆分范围限制，已文档化）
- 最终：**L0/L1 全部 [PRECISION_PASS]，L2 全部正确拒绝，Test Passed! (exit 0)**
- next_hint: 性能基线已记录；Stage 3 调优方向见 perf_tuning/fla_wy_fast_baseline.md

## Stage 3 — 2026-10-22（性能调优踩坑）

- mode: perf_tuning（Stage 3）
- classification: n/a（精度全量保持 pass，见 perf_log）

### T.serial 行循环与 in-place 2D T.Parallel 行广播等价（都是反模式）
- 生成代码证据：两者都 lower 成 64 次标量循环（`bt.GetValue(i)` + 每行一条
  `Muls` repeat=1）。Stage 2 的"in-place 2D Parallel 正确"只解决了正确性，
  没解决性能。
- **修复**：`T.tile.broadcast(bt2d, bt, axis=1)`（行向量）+ 一条全量 `tile.mul`；
  列向量用 axis=0（kg 的 gn[j]）。M=N 时 auto-infer 有歧义，**axis 必须显式**。
  先例：mha_sink_fwd_bhsd / opt_gdn_solve_tril。逐 kernel -26% ~ -71%。

### str.replace 批量改 kernel 源码的误伤（流程坑）
- 一次多点位 restore 的 `str.replace` 把 k1/k2 的 gexp 链也换成了 fwd-wkg 专属
  的 gL_ub 版本（未声明 → TVM DiagnosticError）。修复后教训：**多 kernel 文件
  内做脚本化替换必须逐点位带上下文唯一锚定，替换后 grep 校验引用-声明一致**。

### k2 UB 边缘：任何生命周期延长都会溢出
- A cast 链 hoist（+~16KB live）→ `dw_lo_f32` 分配失败；CSE Xg（r_tmpk 与
  dk2_ub 生命重叠 +16KB）→ `r_tmpk` 分配失败。均回滚。k2 是 memory-planning
  零余量 kernel，后续优化需先减 buffer 再延长生命周期。

### Pipelined 软件流水已自动重叠循环不变量
- A cast 链手动 prologue hoist 实测中性（v2 1.258→1.278）：pass 本身会把不变
  量的计算与前一 slice 的 gemm 重叠。**手动 hoist 前先消融**，别白付 UB 压力。

### dbeta scatter 的 MTE barrier 成本 ~0
- v2 尾部 set_flag/wait_flag(MTE3→MTE2) 实测被流水吸收（移除仅 -0.009ms）；
  而 da（单发无循环 kernel）接收 scatter 反而 +0.047ms。**barrier 成本要看
  kernel 流水形态，不能只数同步指令**。

### GM↔UB cast-on-copy（bf16↔fp32）codegen 损坏（工具链级）
- codegen 直接生成 `copy_gm_to_ub<bfloat16_t>(LocalTensor<float>, ...)`——
  C++ 模板要求 dst/src 同型，编译失败。cast_on_copy 测试矩阵（fp16/int 系）
  不含 bf16↔fp32。fwd 的 bf16 staging buffer + 显式 UB cast 无法去除。

### Vector 核上 div 与 exp 同价
- kg 的 div 变体（k·2^gn / 2^g，1D exp 64 次 + 2D div 4096 次）与 sub+exp
  变体（2D sub + 2D exp 4096 次）耗时持平（2.987 vs 2.971ms）。

### wrapper 内部 sync 的口径问题（已修正）
- Stage 2 在 host wrapper 内加 `torch.npu.synchronize()`：上游 do_bench 只在
  rep 循环外 sync，上游算子 wrapper 内部无 sync。内部 sync 把 JIT 分发延迟
  （~1ms/launch，cython wrapper 每次 call 重建 TVM Analyzer）暴露进设备侧
  计时。移除后对齐上游（端到端 -1.37ms）。两种口径数字均已记录在
  perf_tuning/fla_wy_fast_final_summary.md。

### 实测 HBM 带宽修正（重要）
- D2D 大拷贝（256MiB-1GiB）实测 **1259-1273 GB/s (r+w)**，为 DESIGN §9.5
  假设标称 392GB/s 的 3.2×。"350-400GB/s 已饱和"结论不成立；1370us 目标
  需 1.44× 实测上限，仍物理不可达但倍数需修正（原 4.6×）。

- changes: fla_wy_fast.py（broadcast 化 6 kernel + msk 缓存 + sync 移除 +
  文档更新）、perf_tuning/{optimization_log,perf_log,fla_wy_fast_tuning,
  fla_wy_fast_final_summary}.md、bench_per_kernel.py
- test_level: all（Test Passed! exit 0；Boundary 行为与基线逐位一致）
- 最终：fwd 4.783 / bwd 6.935 / 合计 11.719 ms（上游口径，基线 21.636，
  -45.8%；Stage 2 口径 23.151 → 13.084，-43.5%）
