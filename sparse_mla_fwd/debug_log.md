# sparse_mla_fwd 开发日志（debug_log）

> **归档出处说明（2026-10-22 按算子拆分回迁）**：开发期记录原写入沙箱共享 `debug_log.md`
> （多算子共写的历史形态），首轮归档时未拆分、随 indexer_bwd 会话整体冻结于
> `../indexer_bwd/debug_log.md`。本文件为 sparse_mla_fwd 相关章节的逐字提取
> （出处行号 L165-372 / L396-405 / L486-489 / L553-664），完整性以该文件为兜底副本。
> 自调优 Round 2 起按 tilelang-op-migration Phase 2 规范以独立 `debug_log_{op}.md` 维护。

## sparse_mla_fwd — Stage 2 分块 A（kernel 实现 + 编译冒烟，attempt 1 / first_impl 分片）

- mode: first_impl（分块 A：仅 kernel + 冒烟；test 文件属 B 块）
- date: 2026-10-14
- artifacts: sparse_mla_fwd.py（569 行，ruff 全过）

### R1（gemm_v0 L1 操作数列切片）— FAIL → 回退（F6b）
- 最小复现 /tmp/opencode/r1_gemm_colslice.py：`gemm_v0(A_l1, kv_l1[0:128, 256:512], C)` ascendc 与 pto
  均错（max_diff 7.075e+01，两后端逐位一致）；r1b_isolate.py 隔离：切片 [0:256] 单独正确，
  [256:512]/[64:320]/[512:576] 任何非零列偏移均读错且不匹配任何干净窗口 → 共享前端（分形布局
  offset 计算）缺陷，非后端差异。
- r1c：A 操作数行切片偏移 64（q_l1[64:128,:]）同样双后端 FAIL（1.274e+02）→ 结论：L1/L0C gemm
  操作数的非零偏移 BufferRegion 一律不可用（零偏移缩宽可用）。
- 设计注 C 字面回退（ws_main [BI,512] 整块 + PV 单 gemm N=512）不可行：acc_o [64,512] fp32 单独
  即 128KB，加 acc_s 32KB 超 L0C。实际落地 F6b（见下）。

### R2（pto GM 标量读）— PASS
- /tmp/opencode/r2_scalar_read.py：TokenIndices/Offsets GM 标量 → T.float32(s_i+1) tile 右操作数 +
  标量行寻址 Q copy，pto 与 ascendc 均 max_abs_diff=0.000e+00。

### R4（3D L1/UB region，F6b 依赖项）— PASS
- /tmp/opencode/r1d_f6_probe.py：GM→L1 partial-inner 列段拷贝（含 3D GM 固定行 + 列偏移）、
  QK 3 段 K-split（init=T/F/F）、PV 整 B 不可转置 gemm：双后端 diff ≤ 4.2e-05。
- /tmp/opencode/r4_3d_regions.py：3D L1 [S,BI,DS] 前导维切片 [s,:,:] 作 T.copy dest 与 gemm_v0
  B 操作数、3D UB tile fill/add：双后端 diff ≤ 3.4e-05。

### R3（mask/rows 链形态）— PASS
- /tmp/opencode/r3_chain_forms.py：min/max 含 symbolic 标量界（seq_len_kv−1）、int32↔fp32 auto-cast、
  mul(−1)+add 规避 sub 标量左操作数：双后端 rows/mask diff = 0。

### 与设计的主要偏差（F6b，均已双后端实证）
1. **AUTO_CV_SYNC 语句计数约束**（首编失败："cube has 3, vec has 1"）：pass 按 workspace 名要求
   C/V 两侧访问语句数相等且按序配对（ascend_combinecv.cc L375）。设计 §3.3 的 r 循环
   （RH×QK 写 ws_s）与 3 段 ws_kv 读在此约束下不成立。
2. **落地结构**：QK/PV 单粒度 M=Hseg（gqa_pto 语句粒度）+ 3D 分段缓冲循环：
   - ws_kv 单缓冲 [grid,BI,576]；V 侧 gather 每行按列段（N_SEG×BI 宽 + tail 64）写出（语句数与
     C 侧读配对 2:2）；C 侧 strided GM→L1 读入 v_all [N_SEG,BI,BI] + tail_l1。
   - QK = 每 K 段一个 gemm（首段 init=True，其余与 tail init=False 累加）→ acc_s [Hseg,BI]，
     ws_s 单写单读（1:1）。
   - PV = 每 N 段一个 gemm（init=True）→ acc_o_l0c [Hseg,BI]，ws_o N_SEG 写 ↔ N_SEG 读。
   - V 侧 acc_o_all [N_SEG,v_block,BI] 3D UB 跨块累加；段读经 acc_s_ub 复用槽。
   - Q 装载：每 K 段一次 partial-inner GM→L1（q_all [N_SEG,Hseg,BI] + qt_l1）。
3. **DS = BI**（QK K 段宽 = PV N 段宽 = V o-chunk 宽）：BI=128 → N_SEG=4；BI=64 → N_SEG=8。
   （dim=512 下 dim%DS==0 且 V 读槽与 acc_s_ub 形状复用要求 DS==BI。）
4. **支持域收紧**：(dim, tail_dim)=(512,64)（576 分解下唯一满足全部约束的组合）；RH ≤ 2
   （head_kv=128 上限，head_kv≥192 的 RH>2 未实现——3D 结构已天然支持，如需支持改 assert 即可）。
5. **L0C**：Hseg=128×BI=128 恰好 128KB（acc_s 64KB + acc_o 64KB，含边界），实测编译+运行正确；
   BI=64 时 64KB。
6. **UB**：acc_o_all 128KB + acc_s_ub 32KB + cast_bf16 16KB（p 写出与 o 写出共用槽）+ 1D 链
   （m1 复用为 bias、idx_f 复用为 rows_f）≈ 183KB（BI=128）/ 167KB（BI=64）< 192KB，未依赖
   MEMORY_PLANNING 合并。
7. gemm 数/块 = 2×N_SEG+1（BI=128 → 9 vs 设计 6）；V 侧 gather 写事务 ×(N_SEG+1)/行（字节量不变）
   ——两者均为 Stage 3 优化菜单（ws_kv 双槽/批量 staging/列段加宽若前端修复切片 bug 可回设计形态）。

### 冒烟结果（驱动 /tmp/opencode/smoke_fwd.py，golden = kernel 语义逐行 torch 复刻）
| 用例 | shape | ascendc | pto |
|------|-------|---------|-----|
| mini | S=64 topk=64 BI=64 H=128 G=1 | PASS（out matched 99.95%，max_abs 1.56e-2；lse 100%，1.4e-6） | PASS（99.95% / 1.4e-6） |
| l0small | S=1088 topk=1024 BI=128（段 543/545 < topk，重无效槽） | PASS（99.997%） | PASS（99.997%） |
| l0shape | S=2050 offsets=[0,1024,2050]（设计 L0-small 原形） | PASS（99.998%） | PASS（99.998%） |
| g2_rh1 | S=200 H=128 HKV=2（RH=1 路径 + kv_group=2） | PASS（99.98%） | PASS（99.98%） |
| repeat | S=128 topk=64 重复索引（§1.4 #1 重复计贡献语义） | PASS（99.97%） | PASS（99.97%） |

精度口径：bf16 atol 2⁻¹⁰ / rtol 2⁻⁶ / matched≥0.99；lse fp32 atol 2⁻¹⁶ / rtol 2⁻¹⁰。max_abs
1.56e-2 均落在 rtol 项内（matched ≥ 99.95% 全过）。golden 曾有 bug（漏 +bos），修正后 batch0/1
一致——错在冒烟 golden 不在 kernel（诊断记录 /tmp/opencode/diag_mini.py：rows 0:32 3% vs 32:64
99% 的按行分布即 bos 缺失指纹）。

### 已知边界（非本块缺陷，B 块注意）
- 全无效行（sumexp=0）：kernel 输出 NaN/±inf（÷0），主仓 kernel 同型行为；主仓 golden（scatter
  mask softmax）为 NaN——L0 randperm 输入不触发；L1 语义用例若造全无效行需与 golden 口径对齐。
- pto/ascendc 两后端 lse max_abs ≤ 1.7e-6（远小于 atol），Output matched ≥ 99.95%。
- prepare_token_indices 带 tensor_cache（单条目、张量身份键）；_BUILD_CACHE 键含 target。

### 待续（B 块）
1. test_sparse_mla_fwd.py：按 DESIGN §9.2 L0 计划（4 用例 × 双后端）+ ref golden
   （ref_sparse_mla_fwd_interface NPU 版 + assert_tensors_similar eps=1e-2 并列门禁）+ main/--level。
2. L0 用例输入构造须含 offsets=[0, 1024, 2050]（段≥topk）；l0_main（S=4096）golden 成本与
   perf 哨兵归 B/C 块。
3. 冒烟 golden（/tmp/opencode/smoke_fwd.py 的 kernel 语义复刻版）可作 L1 重复索引语义用例的
   golden 模板（§9.1 已预告）。
4. 性能：本块未计时（并发负载下仅定性）；F6b 的 gemm 数（9/块）与 gather 写事务增多是 Stage 3
   首查项；若前端修复 L1 切片 offset bug，可回设计 §3.3 形态（6 gemm/块、单条整行 gather 写）。

## sparse_mla_fwd — Stage 2 分块 B（L0 测试落地，attempt 1 / first_impl 分片续）

- mode: first_impl（分块 B：test_sparse_mla_fwd.py；上轮 B 调度被基础设施截断，本次重建）
- date: 2026-10-14
- artifacts: test_sparse_mla_fwd.py（344 行，分 5 段构建，ruff 全过）
- kernel 修改：无（sparse_mla_fwd.py 569 行未动，精度一次通过，无需备份/回滚）

### L0 结果矩阵（DESIGN §9.2 四用例，双门禁 + Lse 门禁全过）
| 用例 | out similarity(eps=1e-2) | out matched/max_abs (bf16 门禁) | lse matched/max_abs (fp32 门禁) | 判定 |
|------|--------------------------|--------------------------------|--------------------------------|------|
| l0_small_ascendc | 0.999998 | 1.0000 / 1.562e-02 | 1.0000 / 1.431e-06 | [PRECISION_PASS] |
| l0_small_pto | 0.999998 | 1.0000 / 1.562e-02 | 1.0000 / 1.431e-06 | [PRECISION_PASS] |
| l0_main_ascendc | 0.999998 | 1.0000 / 1.562e-02 | 1.0000 / 1.431e-06 | [PRECISION_PASS] |
| l0_main_pto | 0.999998 | 1.0000 / 1.562e-02 | 1.0000 / 1.431e-06 | [PRECISION_PASS] |

`python examples_experiment/dsa_sparse_finetune/test_sparse_mla_fwd.py --level l0` → "Test Passed!" exit 0，
全程 ~2min（输入构造 S=4096 ~27s、golden score 2.15GB/段、每后端编译 ~13s、双 shape 共享编译缓存——
seq_len 为 symbolic，_BUILD_CACHE 键不含 S）。预探针 /tmp/opencode/probe_smla_l0.py（S=260, topk=128）
先行验证：跨设备 slice 赋值、NPU golden（einsum/scatter/logsumexp/masked_fill）、端到端双 PASS。

### Lse 校验做法
主仓 golden 不算 Lse（返回值仅 o）。测试内新增 `ref_sparse_mla_fwd_lse`：复刻 ref 的 fp32 score/mask 链
（scatter mask & causal & 首行特例行 `mask[:, :, :1-1, 0]=True` 逐行照搬，该行切片为空实为 no-op），
`score.masked_fill(~mask,-inf).mul(sm_scale)` 后 `logsumexp(-1)`，按段拼回 [S, H] fp32，套 §9.3 fp32 混合容差。

### 移植记录（与主仓的微小差异，均已注释标注）
1. `assert_tensors_similar`/`calculate_tensor_similarity`：utils.py 逐行原样，仅函数末尾加 `return sim`
   （单行增补，避免二次 fp64 遍历；调用点传 .cpu() 张量规避 NPU fp64 支持风险）。
2. `ref_sparse_mla_fwd_interface`：主仓 L214-257 逐行，仅 `device="cuda"` → `device=Q.device`。
3. `make_l0_inputs`：主仓 L274-286 逐行（manual_seed(0)/offsets=[0,S//2-1,S]/fill=S/randperm 前缀填充），
   device="npu"；S=2050 时 offsets 自然为 [0,1024,2050]（设计 l0_small 原形）。
4. lse 函数中主仓 verbatim 链的 `dim`/`h_index` 在该路径未使用 → `# noqa: F841` 注释保留原行。

### 观察与遗留（C 块注意）
1. combinecv.cc:527 Warning "Cube loop times (32) not enough to catch up vec (512)"：编译期性能提示
   （小任务量下 C/V 循环次数失配），正确性不受影响（matched=1.0000）；与 A 块同型，Stage 3 关注。
2. 输入构造跨设备赋值（CPU randperm → NPU slice assign）实测可行且快（S=4096 ~27s）。
3. 全无效行（sumexp=0 → kernel 输出 NaN，A 块已知）L0 randperm 输入不触发（每行 ≥1 有效位含自注意）；
   C 块 L1 若造全无效行需与 golden 口径对齐（A 块遗留 #2 仍有效）。
4. l1/l2/boundary/perf 均为桩（print "pending chunk C"; return 0，exit 0）；C 块按场景 B 扩展
   L1（重复索引/-1/越界/多段非整除，golden 用 A 块 kernel 语义复刻模板 /tmp/opencode/smoke_fwd.py）、
   L2（host assert 域：return_p_sum=True、dim≠576、topk%BI≠0、heads%kv_group≠0、head_kv 域、
   (dim,tail)≠(512,64)）、Boundary（全零/极值，避开全无效行 NaN 口径）、perf（§9.4 哨兵，目标 <24288us）。
5. 覆盖门禁 coverage_check.py 未跑（本块仅 L0，属 C 块扩展后强制项）。

### 分块 C1（L1/L2/Boundary 测试扩展 + 全量验证，attempt 1 / first_impl 分片续）

- mode: first_impl（分块 C1：test 扩展；C2 = 覆盖门禁 + perf）
- date: 2026-10-14
- artifacts: test_sparse_mla_fwd.py（615 行，ruff check + format 全过）；kernel sparse_mla_fwd.py 未改动（无备份/回滚需求）
- 新增组件：kernel_semantic_golden（A 块 /tmp/opencode/smoke_fwd.py 模板，段直迭代等价 prepare_token_indices 映射）、
  make_varlen_inputs（offsets 自由参数化，段长<topk/空段允许，无主仓 seq_len>=topk 约束；每行保底 1 个有效槽）、
  mutate_repeat / mutate_neg_oob / mutate_same_index、run_semantic_case（blocking 标记复用 L0 双门禁 + Lse 门禁）、
  expect_raise（L2 异常类型精确匹配）。

### L1 结果（8 配置 × 双后端 = 16 运行，全 [PRECISION_PASS]，阻塞层全绿）
| 用例 | 配置 | out matched/max_abs | lse matched/max_abs |
|------|------|--------------------|---------------------|
| l1_repeat_indices | S=128 topk=64 BI=64 offs=[0,64,128] 重复+无效槽 | 0.9997 / 1.56e-2 | 1.0000 / 1.9e-6 |
| l1_neg_oob_indices | S=200 topk=64 BI=64 offs=[0,100,200]（-1/-5/S/S+10/段边界混合） | 0.9998 / 1.56e-2 | 1.0000 / 1.4e-6 |
| l1_short_segments | S=1088 topk=1024 BI=128 offs=[0,543,1088]（段<topk） | 1.0000 / 1.56e-2 | 1.0000 / 1.9e-6 |
| l1_kvgroup2 | S=200 H=128 HKV=2 topk=128 BI=128 | 0.9998 / 1.56e-2 | 1.0000 / 1.9e-6 |
| l1_ni1_topk_eq_bi | S=128 topk=128 BI=128 offs=[0,60,128] | 0.9997 / 1.56e-2 | 1.0000 / 1.4e-6 |
| l1_ni2_topk_2bi | S=600 topk=256 BI=128 offs=[0,300,600] | 0.9999 / 1.56e-2 | 1.0000 / 1.4e-6 |
| l1_empty_segment | S=128 topk=64 BI=64 offs=[0,0,128]（空段） | 0.9999 / 1.56e-2 | 1.0000 / 1.2e-6 |
| l1_single_batch | S=128 topk=64 BI=64 offs=[0,128]（单段） | 0.9999 / 1.56e-2 | 1.0000 / 1.2e-6 |
（ascendc 与 pto 双后端逐用例同判；相似度 eps=1e-2 门禁全过 0.999998+）

### L2 结果（4/4 正确拒绝，[BOUNDARY_PASS]，非阻塞）
- l2_return_p_sum → AssertionError("This kernel file is for fwd only")
- l2_bad_topk（topk=100%BI=128≠0）→ AssertionError（编译期 topk%block_I 域检查）
- l2_bad_dtype（Q fp16）→ ValueError("Buffer dtype mismatch for parameter Q: expected torch.bfloat16, got torch.float16")（运行时分发期，探针 /tmp/opencode/probe_c1_dtype.py 实证）
- l2_bad_dims（d_v=256→tail=320）→ AssertionError("tail_dim must be a power of 2")（探针 /tmp/opencode/probe_c1_l2.py 实证）

### Boundary 结果（bnd_zero/bnd_all_same PASS；bnd_extreme WARN ×4 行，非阻塞）
- bnd_zero_qkv：双后端 out/lse 全 0 误差精确一致（[BOUNDARY_PASS]）。
- bnd_extreme_values（q[::5]×1e4、kv[::3]×1e4）：[BOUNDARY_WARN]（双后端 out max_abs=1.28e+02 > bf16 帽 1e0；lse max_abs=1.92e+02 > fp32 帽 1e-2；matched_ratio 0.9999 / 1.0000，相似度 1.000000）。根因定性：±1e4 量级下 score ~1e10，fp32 累加顺序差异（kernel gemm 分段累加 vs golden 整列 matmul）使个别行 argmax 翻转（out 差 ~|v1−v2|~1e2）与 m 值抖动（lse 差 ~1e2，远小于 rtol 项 9.77e-4×1e10≈1e7 但超 max_abs 帽）。属极值尺度下的固有数值噪声，非 kernel 缺陷（无 inf/NaN，m-shift+BIGC 溢出保护生效）；按口径记 WARN 不阻塞。
- bnd_all_same_index：全同索引行（极端重复）双后端 PASS（0.9998 / 1.9e-6）。

### 已知边界豁免（known-boundary，不设用例）
- 全无效行（sumexp=0 → kernel 输出 NaN/±inf）：A 块已定性，主仓 kernel 同型行为；本块构造器保证每行 ≥1 有效槽，不触发。

### 全量验证
`python examples_experiment/dsa_sparse_finetune/test_sparse_mla_fwd.py --level all` → "Test Passed!" EXIT_CODE=0
（L0 8/8 + L1 32/32 [PRECISION_PASS]；L2 4/4 [BOUNDARY_PASS]；Boundary 8 PASS + 4 WARN；日志 /tmp/opencode/c1/all.log）。
注：nohup 后台跑法需 setsid 脱离进程组，否则 shell 超时击杀会连带后台任务（本次踩坑 1 次，重启无损）。

### C2 注意事项
1. 覆盖门禁 coverage_check.py（proto_sparse_mla_fwd.yaml 派生维度）未跑——属 C2 强制项。
2. perf 哨兵（§9.4，目标 <24288us）仍为桩（--level perf 打印 pending）。
3. bnd_extreme_values 的 WARN 口径若 C2 质检质疑：可引用本条根因定性（极值尺度累加序噪声，rtol 项内）。

### 分块 C2（覆盖门禁 + perf 哨兵基线，attempt 1 / first_impl 分片续）

- mode: first_impl（分块 C2：coverage_check + --level perf 实现；kernel sparse_mla_fwd.py 未改动——perf 只测不调优）
- date: 2026-10-14
- artifacts: test_sparse_mla_fwd.py（~730 行，ruff check + format 全过；新增 4 个 L1 用例、3 个 Boundary 特殊值用例、COVERAGE_* 标注块、perf 双口径实现）

### 覆盖门禁（工具：.agents/skills/tilelang-op-test-design/scripts/coverage_check.py --proto proto_sparse_mla_fwd.yaml）
初次跑 25 MISS —— checker 以文件内 tags/COVERAGE_MANIFEST 为判定来源，C1 产物未带标注。补标注 + 缺维度用例后复跑：
**COVERAGE: 25 PASS / 0 MISS / 0 N/A -> PASS**（category=Fusion，声明 COVERAGE_CATEGORY；Fusion 无可豁免维度，COVERAGE_NA={}）
- D-DTYPE-bf16/int32=22；D-PARAM-* 全过（heads 128/64、dim 512/256 拒、tail 64/320 拒、topk 64/128/256/1024、kv_group 1/2、sm_scale 显式 0.5、block_I 64/128、target ascendc/pto）
- 新增 4 个 L1（均双后端 PASS）：l1_tail1_prime（S=257=2×128+1 质数单段 → TAIL-1+PRIME）、l1_smscale_heads64（H=64+sm_scale=0.5）、l1_vrange_s（±1）、l1_vrange_asym（randn×3+5）
- 新增 3 个 Boundary：bnd_inf_q / bnd_nan_kv 双后端 [BOUNDARY_PASS]（golden 非有限位按 §9.3 跳过，finite 位 matched 0.9997+）；bnd_dbound（±bf16_max）双后端 [BOUNDARY_WARN]（matched 0.9312/0.9315，max_abs=nan——±3.39e38 在 fp32 累加序上 golden/kernel 溢出判定翻转，双后端逐位同判说明是数值路径系统行为而非随机 bug，非阻塞记录）
- 豁免说明：本算子固定 dtype 组合（bf16/int32/fp32-lse），无 dtype 扫描维度；manifest 中 D-DTYPE-fp32=18（lse 门禁）供参考，proto inputs 不派生该维度
- 语义维度（任务矩阵）：重复索引 l1_repeat、-1/越界 l1_neg_oob、NI 变体 l1_ni1/ni2、空段/单段/多段 l1_empty/single/short —— 均已有用例；后端维度 ascendc/pto 全用例双跑

### perf 哨兵（DESIGN §9.4，B=1,S=4096,H=128,HKV=1,DQK=576,DV=512,topk=1024,bf16，make_l0_inputs 复用）
| 后端 | 纯 kernel（warmup5+20 轮，NPU 事件） | 接口级（warmup5+50 轮，tensor_cache 生效） | vs 24288us |
|------|--------------------------------------|-------------------------------------------|-----------|
| ascendc | 138551 us ± 205（min 138311 / max 139284） | 137856 us ± 279（min 137647 / max 139614） | **OVER 570.5%** |
| pto | 133774 us ± 297（min 133499 / max 134910） | 134576 us ± 212（min 134311 / max 135543） | **OVER 550.8%** |

- 测量环境：NPU 空闲（npu-smi 无进程，AICore 0%）串行跑，无并发污染
- 瓶颈初判（不深调优，Stage 3 的事）：稳态 std <0.3%（非抖动）、双后端同速（pto 快 3.5%）→ 瓶颈在 kernel 结构而非 codegen 差异。有效算力 1.167TFLOP/138.6ms≈8.4 TFLOPS（预算需 48）；聚合带宽 ~33.5GB/138.6ms≈242GB/s（HBM/L2 均未吃满）→ MTE 发射效率低 + C/V 未流水。结构性根因：F6b 每 task 48 次小 gemm（M=64，L0C 反复装填、Cube 利用率低）+ 每 task 1024 行 KV 逐行 gather 事务碎片化。Stage 3 按 §9.5 优化档（M=128 合并 gemm / gather 批量 staging / ws_o bf16）重构，基线已立

### 全量复跑（perf 改造后回归）
`--level all` → "Test Passed!" EXIT_CODE=0：[PRECISION_PASS] 56（L0 8 + L1 48）/ [PRECISION_FAIL] 0；[BOUNDARY_PASS] 16（L2 4 + Boundary 12）；[BOUNDARY_WARN] 6（bnd_extreme ×4 C1 已定性 + bnd_dbound ×2 新增，均非阻塞）。日志 /tmp/opencode/c2/{l1,bnd,perf,all}.log

### 遗留
1. perf 超标 5.5-5.7×（OVER）——Stage 3 主战场；双后端双口径基线已固化在 --level perf
2. bnd_dbound WARN 若 Stage 3 重构累加链可复测；bnd_extreme_values WARN 维持 C1 定性口径
3. Stage 2 至此收口：L0/L1 阻塞层全绿 + 覆盖门禁全 PASS → [PRECISION_PASS]


## Stage 3 性能调优（iter 1-7 摘要行，详细章节见 perf_tuning/sparse_mla_fwd_perf_log.md）

### Stage 3 迭代 1（性能画像 + 快赢 A/B，2026-10-14）
perf_tuning/sparse_mla_fwd_perf_log.md 已建：画像（var_nogather 差分 + topk 缩放 + msprof PipeUtilization）定量结论——gather 碎片化中转 100.7ms/72%（AIV MTE3 51% 占用 @2.15GB/s、AIC 91% 等 flag、cube 纯算仅 3.1%）；三项快赢（BI64/ws_o bf16/grid40）全部否决，主文件零改动（备份 history_version/sparse_mla_fwd_perf_iter1.py 即基线）。迭代 2 排序：P0 gather 批量 staging → P1 softmax broadcast → P2 C/V 跨块流水（indexer_bwd pair 先例）。

- 迭代 2（P0 gather 批量 staging，2026-10-14）：sparse_mla_fwd.py gather 段重构——kv_stage [GKB=HALF/2,576] 两批整 2D MTE3 写（320 碎片写→2 大写+1 幂等小写），AUTO_CV_SYNC 配对 2:2→3:3（C 前置 [16,16] dummy 读、V 尾部单行幂等重写；首版 2:2 因"C 语句只能依赖配对及更早 V 语句"语义竞态双后端 97% 损坏，生成代码核验修复）。探针：ascendc 2D UB 列段切片源损坏（pto 正常）、取模区域索引损坏；安全形式=整 2D 源/1D 行切片源/行 dest staging。UB：kv_stage 36KB 与 acc_s_ub/cast_bf16 生命期别名（峰值 188928/196352B）。精度 L0 4/4+L1 48/0+L2 4/4+Boundary 16 过 2 WARN（基线逐位）双后端；perf ascendc 81700±270us（-41.1%）/pto 78565±308（-41.3%），vs 24288 预算 OVER 336%。**采纳**（新基线，备份 history_version/sparse_mla_fwd_perf_iter2.py=改前版）。事故：l1 首跑被 cgroup OOM 静默 kill（与 bwd 会话编译叠加），等待后串行重跑全过；本轮未 kill 任何非本会话进程。详情 perf_tuning/sparse_mla_fwd_perf_log.md 迭代 2 章节。

- 迭代 3（P1 softmax 向量化，2026-10-14）：sparse_mla_fwd.py softmax 链 448 行算子/块→~74 组级 op（T.tile.broadcast axis=0/1 + [8,BI] 行组 temp + 2D/3D 区域算子 + reduce 拆组压 workspace 16K→2K）+ mask 链 min 形态 11→8 op。新平台坑：①整 3D GM→UB T.copy 只拷第一 2D chunk（双后端，probe2；主 kernel 无此形态）；②[64,128] fp32 bcast temp 与 acc_o_all+acc_s_ub 互斥（196,608B>196,352B）且 16K/8K 版 planner 溢出（reduce workspace 生命期冲突，尾余 7.4KB）→ 8 行组 4KB 通过；③VEC UB 32B 对齐——1D 切片 broadcast 源 EG=4（v_block=32）偏移 16B 双后端 aicore 崩溃，EG=8 固定行组修复。smoke_modified.py 对 S=128/TOPK=128 config 容差误报（备份同值 1274-mismatch，已备份对照排除）。精度 --level all 全过（L0 逐位 + L1 48/0 + 6 WARN 逐位零恶化）双后端；重画像：gather 43.4ms(53%)/softmax 链 21.8ms(27%)/mask 链 ~0/残差 16.5ms，候选实测与分解闭合；perf ascendc 64437±211us（-21.1%）/pto 75635±194（-3.7%），vs 24288 OVER 165%。**采纳**（备份 history_version/sparse_mla_fwd_perf_iter3.py=改前版）。测量纪律：bwd 会话测试窗口全程等待，未 kill 任何进程。详情 perf_tuning/sparse_mla_fwd_perf_log.md 迭代 3 章节。

- [perf-iter4] ring 探针（t3i4）：AUTO_CV_SYNC ring 被 5 层机制封死——flag 预算 16（GetffstMsg &0xf）、静态语句配对+循环乘数匹配、planner 单区间生命期（环形交错失别名→UB 超限→空 free_blocks 段错误）、Mod 区域索引硬件异常（RowBlock 修复后独立复现）；发现主 kernel gather 的 rows_int GetValue 为正确性隐性依赖（is_sliced→PipeBarrier_ALL，纯算术地址→PIPE_MTE2→跨迭代 kv_stage WAR 竞态）；ws_o bf16 读侧 UB hop 绕开 copy_gm_to_ub 无 cast（双后端 L0 8/8 过）但 ascendc -1.17%<3% 阈不采纳（pto -4.78% 留档）；主文件零改动，基线 64437/75635 保留；混合升级判据已触发（详见 perf_log 迭代 4）。
- [perf-iter5] 混合模式升级落地：Expert ring fast path（DEPTH=2×4 ws、8 cross-flag、手动 UB 布局 194,560B、C 侧零手动 flag）+ host 分派（ascendc & 主配置族→fast）；绕开 iter-4 RC1-RC5 全部 5 层。平台发现：单 cast 槽 [全 cast;全 store] 批处理=全 store 读末组内容（7/8 确定性损坏，probe_a_only）→ 逐组 interleave；copy_ub_to_gm 无 cast（需 UB→UB cast 槽）；2D 行段 src cast 拷贝可用；Expert 探针必须自带全部相邻 flag 对；bos if/else 分支教训 load-bearing。精度：--level all 56/0 + 16+6 WARN 与基线逐位一致；fast vs portable oracle 全 bit-exact（含主 shape ×3）。perf：fast ascendc 58763±196 vs 同会话 portable 70206±265（-16.3%）/ vs 记录基线 64437（-8.8%，portable 同会话漂移 +8.9% 如实记录）→ 采纳，新基线 58763/75635；vs 24288 剩 2.42×。iter-6：msprof 校准 + p-store/gather ping-pong 同步往返削减。
- 2026-10-15 perf iter-6：msprof 画像分解完成（C/V cross-flag 等待均≈0、ring 已充分重叠、V 侧略绑定）；4 候选全否决回滚（D 非 bit-exact：Divs/Div 末位差；C2/B2/Blite/B2b +7.6%~+18.7%）；发现硬规则 R-AIV-FLAG（AIV set_flag=源管道阻塞标记，悬空挂死/延迟消费停转→仅相邻全排空对合法）与 R-UB-LAYOUT（V 侧 UB 微布局 256B 腾挪即 -7.6%）；实现零改动（md5 同 iter-5）；探针与 A/B 日志 /tmp/opencode/t3i6/。
- 2026-10-15 perf iter-7：[DESIGN_ERROR] 判定（纯分析轮，零改动，md5 同 iter-5）——架构地板 ~52-56ms = 34.2us/块 × ~1638 块/核（理论墙 56ms 与实测 52.3/profiled 56.1 夹逼闭合；ring 100% 重叠、C/V 双侧自缚平齐）；逐相位归属：bf16 UB→L1 禁直连（common.h:358-361 static_assert half-only）→ws_kv 9.66GB 往返、L0C 128KB→acc_o 放逐 V-UB→ws_o 逐块 flush 16.8GB、R-AIV-FLAG→HardEvent 10.8us/块、R-UB-LAYOUT→UB 布局冻结、flag 预算 16→ring 深度/slot 流水表达受限；iter-6 已实证否决 4 案+分析排除 3 案（≥3 案门槛）；ws_o bf16 最后杠杆正式跳过（连续两次调度截断 + 上限 ~44ms 仍差 ~1.8× 双重理由，留档为 rev2 方向 a 从属项）；rev2 证据包 a-e：a=M=32 分片/L0C 驻留 acc_o（ws_o 16.8→2.1GB；alpha 对角 gemm/数学重排 + 4 分片 256KB>L0C 驻留编排两设计题）、b=BI=256（固定项 ~17us/块摊销，L1 295+147+66KB 贴线未核验，须与 a 联动）、c=fp16 直连档（消 ws_kv 9.66GB+F0 族，DESIGN §9.5 档 3 原案，|x|>65504 谓词+generic 回退成对，Q 需 fp16 契约）、d=deepseek_v4 大 tile 参考（S2_BASE=512/M_CHUNK=16/L1 ring 3 槽/L0A L0B 双缓冲，91 TFLOPS；本仓检索无源码，需外部核对）、e=可达性表（当前 38.7GB→需 1.59TB/s 不可达；a+c=14.3GB→0.59TB/s<实测 0.74 锚点 ✓；b 为固定项必要组成）；详见 perf_log 迭代 7。

## sparse_mla_fwd — Stage 2 rev2 分块 A2（探针 p1-p8 + v2-lite 骨架，attempt 1 / first_impl 续）

- mode: first_impl（分块 A2：前块中断续——探针执行 + fast v2 骨架 + 段①②填充）
- date: 2026-10-15
- artifacts: sparse_mla_fwd.py（1462 行：L1156-1462 新增 tl_sparse_mla_fwd_fast_v2 骨架 +
  _fast_v2_gate_ok；L1-1155 与 history_version/sparse_mla_fwd_rev2_attempt1_chunkA.py 逐字节一致）
- 探针证据链（全记录 /tmp/opencode/t3r2a/log.md，脚本+dump 留档同目录）：
  - p1 DSL UB→L1 行偏移：FAIL——AscendWorkspaceReduction 把 4 组拷贝坍缩到单 workspace 槽
    （dump 实证：4× copy_ub_to_gm 同偏移 + C 侧仅 1 读回且参数错乱 <32,576>/a_l1[768]）
  - p4/p6 raw copy_ub_to_l1（T._src_code）：4/4 组合 aicore 挂死；p6 对照组（同协议无 raw 行）
    正常完成 → 指令级硬件失速。平台事实：needs_gm_workspace_=(platform!="A5")（910B3→"A2"）
  - p2 DSL UB→L1 3D leading-dim（修同步后）：FAIL O(1) 垃圾——A2 上 DSL UB→L1 全形态不可靠
  - p7/p7b C 侧标量行 gather：FAIL——单行 GM→L1 生成参数错乱（模板=buffer 全高 vs 运行时 m=1）
  - g1(=p3) K=576 整 buffer gemm kL0Size=64 fp16+fp32：PASS（vs bf16 ref 1-ULP/1 元素）
  - p8 3D L0C 槽位 gemm C 操作数 + init=False 跨用累加：PASS（噪声级 5e-6）
- [DESIGN_ERROR] 判定：DESIGN §12.2(c) fp16 直连 + §12.1(a2) P 驻留传递依赖 UB→L1 直连，
  910B3/A2 硬件不支持（A5-only）→ §12.6 的 14.3-16.4ms 达标预算失效，24288us 目标在当前
  平台机制空间内不可达（最优路线 30.8-34.6GB ≈ 41-47ms）。需 Stage 1 重决策（目标重定标/
  架构重设计/平台假设修订）。a3 pass2 算术复查：v_all 单块驻留 → pass2 需 ws_kv DEPTH=NI 重读
  或 QK 重算，a3 不再净赚（详见骨架节头 A3 PASS-2 ARITHMETIC CAVEAT）。
- 骨架状态：a3 框架，编译冒烟 BUILD OK（6 gemm 调用、L0C 128KB 贴满、L1 464KB/UB ~101KB 预算内）；
  段① V gather prologue（rev1 形态 BI=256 适配）+ 段② C pass1 QK（2×3 段链 kL0Size=64）已填；
  段③a/③b/④/⑤ TODO（段④依赖 a3 pass2 路线决策）。ruff 15 处 F841 全部为待填段占位（v2 区）。
- 约束遵守：portable/fast v1/interface 零改动（diff 验证）；探针仅 /tmp/opencode/t3r2a/；
  未 commit/push；未读写 .orchestrator_state*；未动框架/ref_gpu/DESIGN/proto/其他算子。
- next_hint: 先让 Orchestrator/Stage-1 裁决 [DESIGN_ERROR]（目标 vs 架构 vs 平台）；若接受
  v2-lite 方向，段④前必须先定 a3-pass2 路线（ws_kv DEPTH=NI 重读 vs QK 重算 vs 改单遍半宽
  4D 架构），再填 ③a/③b/④/⑤ 并跑 L0 冒烟。

## sparse_mla_fwd — Stage 2 rev2 分块 B（段③a/③b 填充，attempt 1 / first_impl 续）

- mode: first_impl（分块 B：a3 两遍法 V 侧两 pass；段①②为前块产物，段④⑤留待下块）
- date: 2026-10-15
- artifacts: sparse_mla_fwd.py（1462 → 1681 行；段③a L1476-1594、段③b L1595-1674；L1-1155 与
  history_version/sparse_mla_fwd_rev2_attempt1_chunkA.py 逐字节一致，diff 复验）；改前快照
  history_version/sparse_mla_fwd_rev2_attempt1_chunkB.py
- 段③a（V pass1 m_final）：rev1 LEAD gather 前瞻逐字移植（BI=256 重定时，8×[16,576] 批，
  nslot*BI ring 偏移，四组 intra-V flag 对全保留）+ post1 = wait F1A+t → ws_s[t] 读 → BIGC mask 链
  （组级 broadcast，rev1 形态）→ reduce_max → m_fin 跨 NI 块 Max 累加；a3 pass1 只做 m（不做
  rev1 在线法 sumexp/alpha）✓
- 段③b（V pass2）：ws_s 重读（CAVEAT 决策按重读实现，QK 重算变体留档节头）→ mask_mem 恢复 +
  bias 2 指令重算 → 链重放 → m_fin broadcast Sub → Exp → reduce_sum → sumexp 累加 → 逐组交织
  cast+store → ws_p[t] + set_cross_flag("MTE3", F2_BASE+t)
- 关键决策：Indices 不重读 GM——pass1 存 clamp01 mask 至 mask_mem[NI,256] fp32（≤6KB，UB 富余
  ~87KB），pass2 恢复重算 bias（pass2 时 ring 双槽已被前瞻 gather 打穿，重读纯亏）；
  sumexp_i 双用（pass1 rowmax / pass2 rowsum，V-pipe 程序序覆盖无竞态）
- 同步协议：沿用 F0/F1A；③b 引入 F2（=任务书 "F1B"，V→C ws_p 就绪）。flag 总数 2NI+4 ≤ 16
  （NI=6 恰 16；dump 实证 F1A=t+2、F2=t_2+8 id 分配正确）。pass2 ws_s 重读零 cross-flag wait：
  F2 链式围栏论证（C 覆写 task k+1 ws_s[t] 前必消费完 task k 全部 F2，而 V 的 F2 set 在 pass2
  读之后）——详见 /tmp/opencode/t3r2a/log.md chunk B 节
- 节头修正：all-invalid-row corner 原 "P=1 饱和" 备注系算术笔误——BIGC=1e30 > 2^30 floor，
  m_fin=-2^30 时 P=exp(-1e30+2^30) 仍下溢 0 → 0/0 → NaN/-inf，与 rev1 角落行为一致（节头已改写）
- 编译冒烟（/tmp/opencode/t3r2b/build_smoke.py）：NI=1/2/6 三档 BUILD OK；C++ 源码核验
  CrossCore flag id / mask_mem 存取 / ws_s 地址 stride / Exp / reduce_max / reduce_sum /
  Broadcast axis=0,1 全部落码；资源 dump：UB 109312B/196352B（含 4KB tmp_ub）、L1 432KB
  （p_l1 被 DCE，段④启用后 464KB）、L0C 128KB 贴满。未跑 e2e（段④⑤ TODO，F3 无人 set 必挂）
- ruff：format 全文件清（顺带展开 chunkA 6 个 gemm 调用的 magic-trailing-comma，纯格式 48 行、
  全在 v2 区、L1-1155 未动、格式化后三档 BUILD 复验 OK）；check 余 2 处 F841 = p_l1（段④）/
  F3_BASE（段⑤）待填占位，符合预期
- 约束遵守：portable/fast v1/interface（L1-1155）零改动；探针/冒烟仅 /tmp/opencode/t3r2b/；
  未 commit/push；未读写 .orchestrator_state*；未动框架/ref_gpu/DESIGN/proto/测试文件
- next_hint: 段④（C pass2 PV）需先定 a3-pass2 路线（ws_kv DEPTH=NI 重读 vs QK 重算，CAVEAT 未决，
  待 Stage-1/Orchestrator 裁决）；段④将 wait F2_BASE+t（本块已 set）并首用 p_l1；段⑤ V 尾
  wait F3_BASE+vid + ws_o 归一化 + Output/Lse；随后 host 侧 interface 接 v2 + L0 冒烟

## Attempt 1 chunk C — 2026-10-15Txx (rev2b a3 两遍法 段④⑤)
- mode: first_impl (chunk C of attempt 1)
- classification: precision_pass (v2 内核 e2e 数值全绿；host 分派未切，非全量门禁)
- fail_category: none（e2e 首跑曾现 Lse 竞态，已定位修复——MTE3 Lse store vs 下一 task V-pipe
  fill(sumexp) WAR；修复=store 后追加 mte3→v 排空对；证据链：log-sum-exp 平移不变性 ⇒ err=
  m_epi−m_p2、ws_p 残留 X 提取、双探针内核仅末 task 正确）
- test_level: all (/tmp 直调驱动 5 用例 + 4 档 BUILD 冒烟)
- coverage: L0 等价 5 用例（NI=1 masked/full/repeat、NI=2 full、NI=4 masked）；BUILD NI=1/2/4/6
- boundary_warnings: none
- changes: sparse_mla_fwd.py 1681→1811（段④ s-outer PV 重读 DEPTH=NI+1 + F0 奇偶 flag；段⑤
  ws_o 归一化+Output/Lse+竞态修复；节头路线裁决+两留档变体）；L1-1155 零改动
- error_summary: 首跑 Lse max_abs≈5 全行 FAIL（Output 全 PASS）→ WAR 竞态，2 行修复后
  Output max_abs 1.56e-2 / Lse 1.7e-6，5/5 PASS
- design_error_reason: none（DEPTH=NI→NI+1 为裁决内参数修正，鸽笼论证留档节头）
- rollback: no
- backup_path: history_version/sparse_mla_fwd_rev2_attempt1_chunkC.py（改后快照）
- instrumentation_cleaned: yes（探针内核均在 /tmp，主文件无插桩）
- next_hint: 下一块 host interface 分派切 v2（_fast_v2_gate_ok 接入 use_fast 链 + ws_kv
  DEPTH=NI+1 形状分配）→ 跑 test_sparse_mla_fwd.py --level l0 全量门禁

## Attempt 1 chunk D — 2026-10-15Txx (rev2b 收官：host 分派切 v2 + 全量精度 + perf 实测)
- mode: first_impl (chunk D of attempt 1, 分块收官)
- classification: precision_pass（host 分派切 v2 后 --level all 全量 56/0；perf A/B v2 vs v1 −10.7% 确认切换）
- fail_category: none
- test_level: all（--level l0 预跑 8/8 → --level all：L0 8 + L1 48 PRECISION_PASS / 0 FAIL；
  L2 4/4；Boundary 16 PASS + 6 WARN 与 iter-5/6 基线逐类一致——WARN 均 portable 路径用例）
- coverage: L0:2cfg×2后端 L1:12cfg×2后端 L2:4 Boundary:6cfg×2后端；v2 路由用例 =
  l0_small/l0_main/l1_short_segments/l1_ni2_topk_2bi（topk∈{1024,256}）ascendc 侧
- boundary_warnings: bnd_extreme_values×4 + bnd_dbound×2（[BOUNDARY_WARN]，非阻塞，与基线一致）
- changes: sparse_mla_fwd.py 1811→1827：interface 分派链 v1→v2（结构性 gate + BI=256 内部定 +
  ws_kv[grid,(NI+1)*256,576]/ws_s,ws_p[grid,NI*128,256]/ws_o[grid,128,512] 新 workspace 分配 +
  fastv2 缓存键）+ docstring block_I 语义 + v1 节头退役注记；v2 段逐字节未动（diff 复验）；
  fast v1 退役（分派不指向，代码保留）；§12.2② 值域谓词核实不采用（v2 全链 bf16 无 fp16 环节）
- error_summary: none（L0 首跑即 8/8；全量一次全绿）
- design_error_reason: none（本块无新设计错误；chunk A 的 DESIGN_ERROR 目标重定标仍待 Stage-1）
- rollback: no（v2 −10.7% ≥3% 阈，切换确认）
- backup_path: history_version/sparse_mla_fwd_rev2_attempt1_chunkD.py（改前）+
  sparse_mla_fwd_rev2_attempt1_chunkD_final.py（改后，md5 2aed46c17ad300f772ecf9358b966648）
- instrumentation_cleaned: yes（perf/oracle 探针仅 /tmp/opencode/t3r2d/，主文件无插桩）
- next_hint: Stage-3 perf（v2 55258us vs 24288 预算 227.7%；pass2 双 N-seg 遍 6.3GB 流量复核，
  chunk C 节头留档变体 QK 重算/单遍半宽 P 可评估）；Stage-1 目标重定标悬置中
- perf 数字表（同会话串行，冻结主 shape 输入，20 轮/5 预热，NPU events）：
  v2 55258±296 / v1 61891±244 / portable 73503±252 / pto 80977±512 us；v2 vs v1 −10.7%、
  vs portable −24.8%；会话内括号复测 55232±347（漂移 <0.05%）；oracle v2 vs portable：
  out bitexact 74.87%、matched(bf16 容差)=1.0、max_abs 3.906e-03、sim 0.99999923；
  lse matched=1.0、max_abs 9.537e-07


## Round 2（交付后稳健优化包，2026-10-22）
：前次开发期 debug_log 未随 journal 归档而丢失；iter 1-7 与 rev2 的完整
> 踩坑史留档于 `perf_tuning/sparse_mla_fwd_perf_log.md`（含 RC1-RC5 / P1-P7 / R-AIV-FLAG /
> R-UB-LAYOUT / R-DIV 等规则）与 `DESIGN.md`。本文件自调优 Round 2 起按规范维护。

## [perf-R2] 2026-10-22 稳健优化包 A1a/A1b/A3a（三项全否，零采纳）

**Attempt 1 — A1a 任务书原案「V 侧 UB cast bf16→fp16」编译失败**
- 分类：硬件限制（新规则 **R-CAST-BF16FP16**）
- 根因：dav_c220 `CastIntrinsicsImpl` 无 bf16→fp16 重载（候选解析落到 int4b_t*，no matching function）
- 受控验证：/tmp/opencode/r2/probe_casts.py（编译错误全文在 logs/probe1.log）；重设计探针
  probe_casts2.py 证三步链 fp32→bf16→fp32→fp16 位保持 bit-exact（P2b），bf16→fp32 / L0C→GM bf16
  cast / fp16 同型 staging 均可用（P3/P4+P5/P6）
- 处置：A1a 重构为 Q/KV host 预转（gather 直读 fp16 副本，V 侧零 cast）+ P 三步位保持仿真

**Attempt 2 — A1a 变体 +19.7% 劣化（证伪审查 §4.1 流量模型）**
- 分类：文档前提错误（算术）+ 硬件（新规则 **R-FP16-NOBW**）
- 根因：① fp16 与 bf16 同为 2 字节，ws_kv/Q fp16 化不省任何 GM 字节（审查 A1a 行 -7.28GB 系 ÷2 笔误，
  与其自身「ws_p 流量不变」表述矛盾）；② 同字节量下 fp16 拷贝模板效率更低（msprof：C mte2 +27% /
  V mte3 +28% / V mte2 +15%，cube/mte1 持平）——零节省 × 负效率 = 纯亏
- 受控验证：同进程交错 A/B（v0 46554 vs a1a 55743，3 轮 std<1%）；msprof_a1a 管道表 vs 审查 §3.1
  同快态 v0；精度侧 a1a 仅 57/268M out 1-ULP + 1 lse 1-ULP（p3 累加序签名），谓词边界 65280/65536
  分派正确（diag_a1a.py）
- 处置：否决回滚；审查 §4.2 C1/C2 行作废（详见 perf_log Round 2 §3）

**Attempt 3 — A1b（-0.5%）/ A3a（快 -2.6% 慢 -1.2%）/ c23 叠加（-1.5%）均不过 3% 采纳阈**
- 分类：收益量化（非错误）
- 根因：A1b 的 ws_o 中转减半只作用于 C fixpipe 0.5ms 量级；A3a 的 LEAD=2 只兑现 F0 等待 5.9ms 的
  ~20%（V 侧 F3 尾部等待不受 gather 前瞻影响）
- 受控验证：A3a bit-exact（ndiff=0，纯调度）；三窗口交错计时全记录（ab_r2/ab_c23/ab_final.log）
- 处置：按「<3% = 无变化 → 回滚」规则全否；A3a 形态留档待拆分重构复用（F0 须 mod-3，parity-2 在
  LEAD=2 下有 set 竞态——任务书「奇偶制与深度无关」表述不成立，推导见 perf_log Round 2 §7）

**工程教训 — 后台进程被工具超时连带杀死**
- 分类：流程
- 根因：bash 工具 120s 超时按进程组清理，`nohup ... &` 的子进程一并被杀（首次 ab_r2 中途消亡，
  无 traceback——勿误判为 NPU/编译崩溃）
- 处置：长跑一律 `setsid nohup ... < /dev/null & disown` 完全脱离进程组

