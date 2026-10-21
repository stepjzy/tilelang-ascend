# iter-1 诊断记录（2026-10-15，Step 1 差分定量）

口径：k2 kernel 级 NPU 事件计时（warmup5 + 30 轮 mean，npu-smi 无进程串行），
ascendc，考核 shape（S=2048, H=64, topk=512, natural seed=0）。
宿主 k2 本会话复核：56145.7us（iter-0 记录 55691us，会话间漂移 +0.8%，闭合）。

## 差分矩阵（var_noatomic / var_writescatter，/tmp/opencode/b3i1/）

| 变体 | k2 mean us | 差分含义 |
|------|-----------|---------|
| stock（iter-0 采纳版） | 56145.7 | 基线 |
| var_noatomic（删 C 侧 G5/G6 + V 侧 scatter 段，ws_dkv 0:0，14 对 flag） | 33369.5 | **dkv 散射路径边际成本 = 22776.2us（40.6%）** |
| var_writescatter（仅 atomic_add → 同地址普通写，其余不动） | 54778.7 | **原子溢价仅 1367.0us（2.4%）**；读回+逐行写发射 = 21409.2us（38.1%） |

两个变体 dQ 均与 stock 逐位一致（dq 数据流未动，差分有效）。

## 结论（指导 iter-1 取舍）

1. **DESIGN §6.5 方案 2（私有 + 归约）否决**：原子操作本身近乎免费（溢价 2.4%），
   换普通散写只省 ~1.4ms，还要加 94MB 私有缓冲清零 + 归约遍——净收益为负。
2. **真瓶颈 = 逐行串行发射路径**（生成源码 kernel_src_A.cpp 实证）：
   scatter 循环每行 3×PipeBarrier<PIPE_ALL> + Indices.GetValue（同步 GM 标量读）
   + 5 段微读（各带 MTE2_MTE3 SetFlag/WaitFlag 握手）+ 1 次 2304B 原子写；
   gather 循环同病（每行 PIPE_ALL + rows_int.GetValue + 1 读 + 5 微写 + 4 握手）。
   26112 行/AIV 全串行 → ~21.4ms。
3. **fwd iter-4 平台结论复用**（同栈同约束）：flag 硬件上限 16（GetffstMsg(id&0xf)
   源码级）；ring/跨块流水被 AUTO_CV_SYNC 五层机制封死（RC1-RC5）；
   rows_int UB 标量读触发的 PIPE_ALL 是 staging WAR 正确性依赖（必须保留）。
4. **iter-1 采纳路线**（对准 21.4ms + gather 微写池）：
   - Edit-1 ws_p/ws_dpb hi+lo 合并为单条 3D 语句（flag 16→14，纯预算腾挪）；
   - Edit-2 pass-1 gather 批量 staging（fwd iter-2 3:3 生产形态，flag →15）；
   - Edit-3 scatter staging：ws_dkv 整块 2D 批量读回 [32,576]×2 + 条件/地址改
     UB 源（mask_ub/rows_int，消除 GM 标量读）+ 逐行条件原子（保留 if 语义，
     避免 NaN×0 毒化）；V 侧 3 读（dummy,batch0,batch1）对 C 侧 3 写
     （G5,G6,idempotent 重写）→ flag →16 贴线。
   - [#5] P/dP h-loop 向量化降级至 iter-2（vec 池仅 1.8ms/3%）。

## 变体正确性口径

- dQ bit-identical（torch.equal）为 smoke 硬判据（数据流重组不改值的形态）；
- dkv 在 stock 自身原子序噪声带内（iter-0 det_check 口径：max_abs ≤3.2e-2、
  frac ≤1e-3、sim ≥1-1e-6）；
- 每个变体 smoke 集合含显式 delta 路径（iter-0 教训）。
