# indexer_bwd — Ascend NPU 实现（Developer 模式）

DSA sparse finetune lightning indexer 反向算子（varlen KL 散度损失反传），语义复刻
`ref_gpu/indexer_bwd.py`：每 token 计算 3 个小 GEMM（logits / dIndexQ / dIndexK）+
向量后处理（relu、差值、mask、broadcast、reduce）+ 逐行 gather / DMA atomic
scatter，输出 `dIndexQ` / `dWeights` / `dIndexK`（atomic，host 零初始化）。

- `indexer_bwd.py`：纯 kernel（`@tilelang.jit`，Developer 4 True pass_configs，
  `threads=1`，grid = `seq_len`（T.symbolic 动态），可 import）+ `indexer_bwd_interface`
  host 封装（内部 `prepare_token_indices` + 输出/工作区分配 + 调 kernel）。
- `test_indexer_bwd.py`：golden（主仓 `ref_indexer_bwd` 逐字节复刻，NPU 上运行）+
  L0 4 用例（DESIGN §9.2）+ main（`--level {l0,l1,l2,boundary,all,perf}`）+ 性能基准。

## 与 DESIGN.md §3.3 伪代码的实现偏差（均有源码证据）

| # | DESIGN 计划 | 实际实现 | 证据 |
|---|------------|---------|------|
| 1 | `T.copy(dlog_ub, dlog_l1)` 等 **UB→L1 直连**（bf16） | 经 GM workspace 中转（UB→GM→L1） | `src/tl_templates/ascend/common.h` `copy_ub_to_l1` 有 `static_assert(std::is_same_v<T, half>)`——UB→L1 直连仅支持 fp16；同族 bf16 先例 `sparse_mla_topk_reducesum` / `sparse_flash_attention` 均走 GM workspace |
| 2 | L0C→UB 直连（logits relu / dk cast） | 经 GM workspace 中转（L0C→GM→UB） | 编译器会把 L0C→UB 自动改写为经 auto workspace 的 GM 往返（`ascend_workspace_reduction.cc`），但 auto workspace 的 wrapper 分配路径不支持符号（动态）grid 形状（`cython_wrapper.pyx` "Auto GM: Dynamic shape scenarios are not supported currently"，实测 `torch.empty(Var,...)` 崩溃）——故 4 个中转 buffer 全部为显式参数，由 host 分配 |
| 3 | Q 预乘 sm_scale（bf16）后进 L1 | Q 原样 GM→L1；sm_scale 折叠到 GEMM 后的 fp32 logits 与 d_logits（bf16 cast 前） | 数学等价（sm_scale>0 与 relu 可交换），且比主仓少 2 个 bf16 舍入点，精度 ≥ 主仓路径 |
| 4 | `dq` 末端补乘 sm_scale | 折叠进 d_logits（见 #3） | 同上 |

workspace 命名为 `workspace_*` 是刻意为之：自动跨核同步 pass（`TL_ASCEND_AUTO_CV_SYNC`）
按缓冲区名含 "workspace" 子串配对 GM 写/读并自动插入 C/V 握手 flag
（`src/transform/ascend_combinecv.cc::FetchWorkspaceName`），已用最小复现 kernel
验证 4 组往返的 `CrossCoreSetFlag/WaitFlag` 全部正确生成、按 cid 分槽、动态 grid
多形状复用同一编译产物。

## L0 精度结果（vs NPU golden，§9.3 bf16 混合容差）

| 用例 | dq | dw | dk |
|------|----|----|----|
| l0_default (S=2048, varlen 1023+1025) | PASS 1.0000 | PASS 0.9915 | **FAIL 0.9611** |
| l0_single_seq (S=1024) | PASS 1.0000 | PASS 0.9918 | **FAIL 0.9608** |
| l0_small (S=128) | PASS 1.0000 | PASS 0.9932 | **FAIL 0.9484** |
| l0_blockI32 (S=512, NI=2) | PASS 1.0000 | PASS 0.9929 | **FAIL 0.9405** |

dq 全部 1.0000、dw 全过门限、`NI=2` 跨块累加（`init=(it==0)`、`reduce_sum
clear=False`）验证正确——GEMM/cast/mask/reduce/atomic 实现层正确。

## dk 失败根因（受控仿真定位，非实现 bug）

- 失配元素**恰好**落在 atomic 热行（l0_default：128/2048 行 = 两段各自的
  topk 行带 [0..63] 与 [1023..1086]）；非热行平均误差 5e-7（零偏置）；
  误差随 S 增长（S=128→0.043, S=2048→0.28），呈 √N 累加噪声特征。
- **仿真复现**：用 kernel 完全相同的逐 token 贡献（dlog bf16 cast + 部分和
  bf16 cast），顺序 bf16 累加 → ratio 0.9610 ≈ kernel 的 0.9611（同失败）；
  **同样贡献 fp32 累加 → ratio 0.9972 / max_abs 0.031（通过）**。
- 结论：唯一不达标环节 = **GM bf16 atomic 累加**（DESIGN §1.3/§3.1 选定路径）。
  DESIGN §9.2 "容差内收敛" 的断言被实验否定；主仓 GPU 版同路径但主仓测试
  从未对 dk 设容差门限（仅打印误差），该断言在上游未被验证过。

## 性能实测

l0_default（S=2048,H=16,D=128,topk=64,bf16）纯 kernel 平均 **1522.6 us**
（torch.npu 事件计时，20 轮取平均，排除 H2D/同步/host 分配；目标 < 80 us）。
主要开销：GM workspace 4 组中转（约 280MB/调用 GM 流量）+ 每 token 64 行
gather/atomic 短 DMA。性能调优属 Stage 3（DESIGN §10.2 已列为临界风险）。

## 给 Stage 1 的量化修复方向（dk 精度）

1. **fp32 GM 累加 + 尾部 cast kernel**（2-kernel）：仿真 ratio 0.9972 通过，
   保留通用 varlen 语义；代价 ~32MB fp32 workspace + 一个小 cast kernel。
2. **共享索引快路径提级**（§9.4 Stage 3 路径前移）：每 block G 个 token 片上
   fp32 预累加后单次 atomic；实测 G=16 → 0.9903 通过、G=32 → 0.9937；
   仅在 topk_indices 跨 token 共享契约下成立，通用路径仍需 1 或 3。
3. **dk 专项容差**（原子 scatter 输出放宽 ratio，如 ≥0.95）：最小改动，但
   随 S 增长误差继续发散（S=2048 已 0.961），固定门限在长序列下不稳。
