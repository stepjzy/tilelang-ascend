#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR GK_handle,  GM_ADDR Beta_handle,  GM_ADDR workspace_a_handle,  GM_ADDR workspace_b_handle,  GM_ADDR o_relay_handle,  GM_ADDR Aqk_handle,  GM_ADDR Akk_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> Q;
  Q.SetGlobalBuffer((__gm__ bfloat16_t*)Q_handle);
  AscendC::GlobalTensor<bfloat16_t> K;
  K.SetGlobalBuffer((__gm__ bfloat16_t*)K_handle);
  AscendC::GlobalTensor<float> GK;
  GK.SetGlobalBuffer((__gm__ float*)GK_handle);
  AscendC::GlobalTensor<bfloat16_t> Beta;
  Beta.SetGlobalBuffer((__gm__ bfloat16_t*)Beta_handle);
  AscendC::GlobalTensor<float> workspace_a;
  workspace_a.SetGlobalBuffer((__gm__ float*)workspace_a_handle);
  AscendC::GlobalTensor<float> workspace_b;
  workspace_b.SetGlobalBuffer((__gm__ float*)workspace_b_handle);
  AscendC::GlobalTensor<float> o_relay;
  o_relay.SetGlobalBuffer((__gm__ float*)o_relay_handle);
  AscendC::GlobalTensor<bfloat16_t> Aqk;
  Aqk.SetGlobalBuffer((__gm__ bfloat16_t*)Aqk_handle);
  AscendC::GlobalTensor<bfloat16_t> Akk;
  Akk.SetGlobalBuffer((__gm__ bfloat16_t*)Akk_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto A_l1 = ascend_l1.GetWithOffset<float>(32768, 0);
  auto B_l1 = ascend_l1.GetWithOffset<float>(16384, 131072);
  auto l0a = ascend_l0a.GetWithOffset<float>(16384, 0);
  auto l0b = ascend_l0b.GetWithOffset<float>(8192, 0);
  auto C_l0c_0 = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto C_l0c_1 = ascend_l0c.GetWithOffset<float>(8192, 32768);
  auto idx = ascend_ub.GetWithOffset<float>(16, 0);
  auto j_grid = ascend_ub.GetWithOffset<float>(256, 64);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(256, 1088);
  auto i_grid = ascend_ub.GetWithOffset<float>(256, 1344);
  auto lt_m = ascend_ub.GetWithOffset<uint8_t>(256, 2368);
  auto slt_m = ascend_ub.GetWithOffset<uint8_t>(256, 2880);
  auto idx16 = ascend_ub.GetWithOffset<float>(16, 3392);
  auto j16b = ascend_ub.GetWithOffset<float>(256, 3456);
  auto beta_h = ascend_ub.GetWithOffset<bfloat16_t>(256, 4480);
  auto beta_f = ascend_ub.GetWithOffset<float>(256, 4992);
  auto a_agg = ascend_ub.GetWithOffset<float>(16384, 6016);
  auto b_agg = ascend_ub.GetWithOffset<float>(8192, 71552);
  auto q_h_row = ascend_ub.GetWithOffset<bfloat16_t>(2048, 104320);
  auto k_h_row = ascend_ub.GetWithOffset<bfloat16_t>(2048, 108416);
  auto g2_row = ascend_ub.GetWithOffset<float>(2048, 112512);
  auto q_f_row = ascend_ub.GetWithOffset<float>(2048, 120704);
  auto k_f_row = ascend_ub.GetWithOffset<float>(2048, 128896);
  auto bc_row = ascend_ub.GetWithOffset<float>(2048, 137088);
  auto e = ascend_ub.GetWithOffset<float>(2048, 145280);
  auto kgj = ascend_ub.GetWithOffset<float>(2048, 153472);
  auto eq_m = ascend_ub.GetWithOffset<uint8_t>(256, 161664);
  auto beta_sel = ascend_ub.GetWithOffset<float>(256, 162176);
  auto beta_col = ascend_ub.GetWithOffset<float>(16, 163200);
  auto r_ub = ascend_ub.GetWithOffset<float>(512, 163264);
  auto aqk_s = ascend_ub.GetWithOffset<float>(1024, 165312);
  auto akk_s = ascend_ub.GetWithOffset<float>(256, 169408);
  auto r1_m = ascend_ub.GetWithOffset<float>(256, 170432);
  auto a_m = ascend_ub.GetWithOffset<float>(256, 171456);
  auto aqk_h = ascend_ub.GetWithOffset<bfloat16_t>(1024, 172480);
  auto akk_h = ascend_ub.GetWithOffset<bfloat16_t>(256, 174528);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag(0);
    tl::ascend::copy_gm_to_l1<float, 256, 128>(A_l1[0], workspace_a[(cid * 65536)], 128, 256, 128, true);
    AscendC::CrossCoreWaitFlag(3);
    tl::ascend::copy_gm_to_l1<float, 128, 128>(B_l1[0], workspace_b[(cid * 32768)], 128, 128, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
    tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[0], 128, 128);
    tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[0], 128, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
    tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_0[0], (bool)1, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[(cid * 32768)], C_l0c_0[0], 64, 128, 64, 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
    AscendC::PipeBarrier<PIPE_MTE1>();
    tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[1024], 128, 128);
    tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[512], 128, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(5);
    tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_1[0], (bool)1, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[((cid * 32768) + 8192)], C_l0c_1[0], 64, 128, 64, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    AscendC::CrossCoreWaitFlag(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::PipeBarrier<PIPE_MTE2>();
    tl::ascend::copy_gm_to_l1<float, 256, 128>(A_l1[0], workspace_a[((cid * 65536) + 32768)], 128, 256, 128, true);
    AscendC::CrossCoreWaitFlag(4);
    tl::ascend::copy_gm_to_l1<float, 128, 128>(B_l1[0], workspace_b[((cid * 32768) + 16384)], 128, 128, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::PipeBarrier<PIPE_MTE1>();
    tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[0], 128, 128);
    tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[0], 128, 64);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(3);
    AscendC::PipeBarrier<PIPE_M>();
    tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_0[0], (bool)1, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[((cid * 32768) + 16384)], C_l0c_0[0], 64, 128, 64, 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(5);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(5);
    AscendC::PipeBarrier<PIPE_MTE1>();
    tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[1024], 128, 128);
    tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[512], 128, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(6);
    tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_1[0], (bool)1, 128);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(7);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(7);
    AscendC::PipeBarrier<PIPE_FIX>();
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[((cid * 32768) + 24576)], C_l0c_1[0], 64, 128, 64, 0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(4);
    for (int32_t tt = 0; tt < 205; ++tt) {
      AscendC::CrossCoreWaitFlag(2);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_l1<float, 256, 128>(A_l1[0], workspace_a[((cid * 65536) + ((tt % 2) * 32768))], 128, 256, 128, true);
      AscendC::CrossCoreWaitFlag(5);
      tl::ascend::copy_gm_to_l1<float, 128, 128>(B_l1[0], workspace_b[((cid * 32768) + ((tt % 2) * 16384))], 128, 128, 128, true);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
      AscendC::PipeBarrier<PIPE_MTE1>();
      tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[0], 128, 128);
      tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[0], 128, 64);
      AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
      AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(4);
      AscendC::PipeBarrier<PIPE_M>();
      tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_0[0], (bool)1, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(5);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(5);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[((cid * 32768) + ((tt % 2) * 16384))], C_l0c_0[0], 64, 128, 64, 0);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(6);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(6);
      AscendC::PipeBarrier<PIPE_MTE1>();
      tl::ascend::copy_l1_to_l0a<float, 256, 128, false>(l0a[0], A_l1[1024], 128, 128);
      tl::ascend::copy_l1_to_l0b<float, 128, 128, true>(l0b[0], B_l1[512], 128, 64);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(7);
      tl::ascend::mma<float, float, 128, 64>(l0a[0],l0b[0],C_l0c_1[0], (bool)1, 128);
      AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
      AscendC::PipeBarrier<PIPE_FIX>();
      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(o_relay[(((cid * 32768) + ((tt % 2) * 16384)) + 8192)], C_l0c_1[0], 64, 128, 64, 0);
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(((tt % 2) + 3));
    }
  }
  if ASCEND_IS_AIV {
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::ArithProgression<float>(idx[0], 0.000000e+00f, 1.000000e+00f, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(j_grid[0],idx[0],tmp_ub[0],(uint32_t[]){16, 16}, (uint32_t[]){1, 16});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 1, false>(i_grid[0],idx[0],tmp_ub[0],(uint32_t[]){16, 16}, (uint32_t[]){16, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(lt_m[0], i_grid[0], j_grid[0], AscendC::CMPMODE::GE, 256);
    AscendC::Compare(slt_m[0], i_grid[0], j_grid[0], AscendC::CMPMODE::GT, 256);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::ArithProgression<float>(idx16[0], 0.000000e+00f, 1.000000e+00f, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::Broadcast<float, 2, 0, false>(j16b[0],idx16[0],tmp_ub[0],(uint32_t[]){16, 16}, (uint32_t[]){1, 16});
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
    AscendC::Duplicate<bfloat16_t, false>(beta_h[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 4, 16>(beta_h[0], Beta[((cid * 1024) + (vid * 4))], 64, 16, 4, bfloat16_t(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 256>(beta_f[0], beta_h[0], 16, 16, 16, 16, 16, 16);
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 16384);
    AscendC::Duplicate<float, false>(a_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8192);
    AscendC::Duplicate<float, false>(b_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    for (int32_t h = 0; h < 4; ++h) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(q_h_row[0], Q[(((cid * 131072) + (vid * 512)) + (h * 128))], 8192, 16, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(k_h_row[0], K[(((cid * 131072) + (vid * 512)) + (h * 128))], 8192, 16, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 16>(g2_row[0], GK[(((cid * 131072) + (vid * 512)) + (h * 128))], 8192, 16, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<float, bfloat16_t, false>(q_f_row[0], q_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<float, bfloat16_t, false>(k_f_row[0], k_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      {
      AscendC::Muls<float, false>(g2_row[0], g2_row[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::Broadcast<float, 2, 0, false>(bc_row[0],g2_row[1024],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){1, 128});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Sub<float, false>(e[0], bc_row[0], g2_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(kgj[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Sub<float, false>(e[0], g2_row[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(q_f_row[0], q_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(e[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::CompareScalar(eq_m[0], j16b[0], float(h), AscendC::CMPMODE::EQ, 256);
      AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(beta_sel[0], eq_m[0], beta_f[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 16, 16, -1, 16, false>(beta_col[0], beta_sel[0], tmp_ub.ReinterpretCast<float>()[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::Broadcast<float, 2, 1, false>(bc_row[0],beta_col[0],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){16, 1});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(e[0], e[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[(h * 4096)], q_f_row[0], 16, 128, 128, 16, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[((h * 4096) + 2048)], e[0], 16, 128, 128, 16, 128, 128);
      tl::ascend::copy_ub_to_ub<float, float, 2048>(b_agg[(h * 2048)], kgj[0], 16, 128, 128, 16, 128, 128);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    tl::ascend::copy_ub_to_gm<float, 128, 128>(workspace_a[((cid * 65536) + (vid * 16384))], a_agg[0], 128, 128, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
    tl::ascend::copy_ub_to_gm<float, 128, 64>(workspace_b[((cid * 32768) + (vid * 8192))], b_agg[0], 128, 64, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
    for (int32_t h_1 = 0; h_1 < 4; ++h_1) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<float, 16, 32>(r_ub[0], o_relay[(((cid * 32768) + (vid * 8192)) + (h_1 * 2064))], 64, 32, 16, 0.000000e+00f);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
      AscendC::Duplicate<float, false>(aqk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Duplicate<float, false>(akk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::Select<float, uint8_t>(r1_m[0], lt_m[0], r_ub[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 256>(aqk_s[((cid % 4) * 16)], r1_m[0], 16, 16, 16, 16, 16, 64);
AscendC::Select<float, uint8_t>(a_m[0], slt_m[0], r_ub[256], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 256>(akk_s[0], a_m[0], 16, 16, 16, 16, 16, 16);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
      AscendC::Cast<bfloat16_t, float, false>(aqk_h[0], aqk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Cast<bfloat16_t, float, false>(akk_h[0], akk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 16>(Aqk[(((cid * 65536) + (vid * 256)) + (h_1 * 64))], aqk_h[0], 4096, 16, 64);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 16, 16>(Akk[(((cid * 16384) + (vid * 64)) + (h_1 * 16))], akk_h[0], 1024, 16, 16);
    }
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
    AscendC::Duplicate<bfloat16_t, false>(beta_h[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    tl::ascend::copy_gm_to_ub<bfloat16_t, 4, 16>(beta_h[0], Beta[(((cid * 1024) + (vid * 4)) + 20480)], 64, 16, 4, bfloat16_t(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
    tl::ascend::copy_ub_to_ub<float, bfloat16_t, 256>(beta_f[0], beta_h[0], 16, 16, 16, 16, 16, 16);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 16384);
    AscendC::Duplicate<float, false>(a_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetMaskCount();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8192);
    AscendC::Duplicate<float, false>(b_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    for (int32_t h_2 = 0; h_2 < 4; ++h_2) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(q_h_row[0], Q[((((cid * 131072) + (vid * 512)) + (h_2 * 128)) + 2621440)], 8192, 16, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(k_h_row[0], K[((((cid * 131072) + (vid * 512)) + (h_2 * 128)) + 2621440)], 8192, 16, 128, bfloat16_t(0.000000e+00f));
      tl::ascend::copy_gm_to_ub<float, 128, 16>(g2_row[0], GK[((((cid * 131072) + (vid * 512)) + (h_2 * 128)) + 2621440)], 8192, 16, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<float, bfloat16_t, false>(q_f_row[0], q_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
      AscendC::Cast<float, bfloat16_t, false>(k_f_row[0], k_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      {
      AscendC::Muls<float, false>(g2_row[0], g2_row[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::Broadcast<float, 2, 0, false>(bc_row[0],g2_row[1024],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){1, 128});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Sub<float, false>(e[0], bc_row[0], g2_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(kgj[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Sub<float, false>(e[0], g2_row[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(q_f_row[0], q_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(e[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      AscendC::CompareScalar(eq_m[0], j16b[0], float(h_2), AscendC::CMPMODE::EQ, 256);
      AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(beta_sel[0], eq_m[0], beta_f[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 16, 16, -1, 16, false>(beta_col[0], beta_sel[0], tmp_ub.ReinterpretCast<float>()[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      tl::ascend::Broadcast<float, 2, 1, false>(bc_row[0],beta_col[0],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){16, 1});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
      AscendC::Mul<float, false>(e[0], e[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
      tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[(h_2 * 4096)], q_f_row[0], 16, 128, 128, 16, 128, 128);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[((h_2 * 4096) + 2048)], e[0], 16, 128, 128, 16, 128, 128);
      tl::ascend::copy_ub_to_ub<float, float, 2048>(b_agg[(h_2 * 2048)], kgj[0], 16, 128, 128, 16, 128, 128);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::PipeBarrier<PIPE_MTE3>();
    tl::ascend::copy_ub_to_gm<float, 128, 128>(workspace_a[(((cid * 65536) + (vid * 16384)) + 32768)], a_agg[0], 128, 128, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
    tl::ascend::copy_ub_to_gm<float, 128, 64>(workspace_b[(((cid * 32768) + (vid * 8192)) + 16384)], b_agg[0], 128, 64, 128);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
    for (int32_t h_3 = 0; h_3 < 4; ++h_3) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
      AscendC::PipeBarrier<PIPE_MTE2>();
      tl::ascend::copy_gm_to_ub<float, 16, 32>(r_ub[0], o_relay[((((cid * 32768) + (vid * 8192)) + (h_3 * 2064)) + 16384)], 64, 32, 16, 0.000000e+00f);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
      AscendC::Duplicate<float, false>(aqk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Duplicate<float, false>(akk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
AscendC::Select<float, uint8_t>(r1_m[0], lt_m[0], r_ub[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 256>(aqk_s[((cid % 4) * 16)], r1_m[0], 16, 16, 16, 16, 16, 64);
AscendC::Select<float, uint8_t>(a_m[0], slt_m[0], r_ub[256], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 256>(akk_s[0], a_m[0], 16, 16, 16, 16, 16, 16);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
      AscendC::Cast<bfloat16_t, float, false>(aqk_h[0], aqk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Cast<bfloat16_t, float, false>(akk_h[0], akk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 16>(Aqk[((((cid * 65536) + (vid * 256)) + (h_3 * 64)) + 1310720)], aqk_h[0], 4096, 16, 64);
      tl::ascend::copy_ub_to_gm<bfloat16_t, 16, 16>(Akk[((((cid * 16384) + (vid * 64)) + (h_3 * 16)) + 327680)], akk_h[0], 1024, 16, 16);
    }
    for (int32_t tt_1 = 0; tt_1 < 205; ++tt_1) {
      AscendC::CrossCoreWaitFlag(((tt_1 % 2) + 3));
      for (int32_t h_4 = 0; h_4 < 4; ++h_4) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 16, 32>(r_ub[0], o_relay[((((cid * 32768) + ((tt_1 % 2) * 16384)) + (vid * 8192)) + (h_4 * 2064))], 64, 32, 16, 0.000000e+00f);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
        AscendC::Duplicate<float, false>(aqk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
        AscendC::Duplicate<float, false>(akk_s[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
AscendC::Select<float, uint8_t>(r1_m[0], lt_m[0], r_ub[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t condval;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval = ((tt_1 * 20) + cid);
        } else {
          condval = cid;
        }
        int32_t condval_1;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_1 = ((tt_1 * 20) + cid);
        } else {
          condval_1 = cid;
        }
        tl::ascend::copy_ub_to_ub<float, float, 256>(aqk_s[((condval_1 % 4) * 16)], r1_m[0], 16, 16, 16, 16, 16, 64);
AscendC::Select<float, uint8_t>(a_m[0], slt_m[0], r_ub[256], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 256>(akk_s[0], a_m[0], 16, 16, 16, 16, 16, 16);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 1024);
        AscendC::Cast<bfloat16_t, float, false>(aqk_h[0], aqk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
        AscendC::Cast<bfloat16_t, float, false>(akk_h[0], akk_s[0], AscendC::RoundMode::CAST_RINT, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
        AscendC::PipeBarrier<PIPE_MTE3>();
        int32_t condval_2;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_2 = ((tt_1 * 20) + cid);
        } else {
          condval_2 = cid;
        }
        int32_t condval_3;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_3 = ((tt_1 * 20) + cid);
        } else {
          condval_3 = cid;
        }
        int32_t condval_4;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_4 = ((tt_1 * 20) + cid);
        } else {
          condval_4 = cid;
        }
        int32_t condval_5;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_5 = ((tt_1 * 20) + cid);
        } else {
          condval_5 = cid;
        }
        int32_t condval_6;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_6 = ((tt_1 * 20) + cid);
        } else {
          condval_6 = cid;
        }
        int32_t condval_7;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_7 = ((tt_1 * 20) + cid);
        } else {
          condval_7 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 16>(Aqk[((((((condval_5 / 4096) * 33554432) + ((condval_6 % 512) * 65536)) + (((condval_7 % 4096) / 512) * 512)) + (vid * 256)) + (h_4 * 64))], aqk_h[0], 4096, 16, 64);
        int32_t condval_8;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_8 = ((tt_1 * 20) + cid);
        } else {
          condval_8 = cid;
        }
        int32_t condval_9;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_9 = ((tt_1 * 20) + cid);
        } else {
          condval_9 = cid;
        }
        int32_t condval_10;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_10 = ((tt_1 * 20) + cid);
        } else {
          condval_10 = cid;
        }
        int32_t condval_11;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_11 = ((tt_1 * 20) + cid);
        } else {
          condval_11 = cid;
        }
        int32_t condval_12;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_12 = ((tt_1 * 20) + cid);
        } else {
          condval_12 = cid;
        }
        int32_t condval_13;
        if (((tt_1 * 5) + (cid / 4)) < 1024) {
          condval_13 = ((tt_1 * 20) + cid);
        } else {
          condval_13 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 16, 16>(Akk[((((((condval_11 / 4096) * 8388608) + ((condval_12 % 512) * 16384)) + (((condval_13 % 4096) / 512) * 128)) + (vid * 64)) + (h_4 * 16))], akk_h[0], 1024, 16, 16);
      }
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 256);
      AscendC::Duplicate<bfloat16_t, false>(beta_h[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
      int32_t condval_14;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_14 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_14 = cid;
      }
      int32_t condval_15;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_15 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_15 = cid;
      }
      int32_t condval_16;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_16 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_16 = cid;
      }
      int32_t condval_17;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_17 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_17 = cid;
      }
      int32_t condval_18;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_18 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_18 = cid;
      }
      int32_t condval_19;
      if (((tt_1 * 5) + (cid / 4)) < 1014) {
        condval_19 = (((tt_1 * 20) + cid) + 40);
      } else {
        condval_19 = cid;
      }
      tl::ascend::copy_gm_to_ub<bfloat16_t, 4, 16>(beta_h[0], Beta[(((((condval_17 / 4096) * 524288) + ((condval_18 % 512) * 1024)) + (((condval_19 % 4096) / 512) * 8)) + (vid * 4))], 64, 16, 4, bfloat16_t(0.000000e+00f));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
      tl::ascend::copy_ub_to_ub<float, bfloat16_t, 256>(beta_f[0], beta_h[0], 16, 16, 16, 16, 16, 16);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 16384);
      AscendC::Duplicate<float, false>(a_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 8192);
      AscendC::Duplicate<float, false>(b_agg[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      for (int32_t h_5 = 0; h_5 < 4; ++h_5) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::PipeBarrier<PIPE_MTE2>();
        int32_t condval_20;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_20 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_20 = cid;
        }
        int32_t condval_21;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_21 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_21 = cid;
        }
        int32_t condval_22;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_22 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_22 = cid;
        }
        int32_t condval_23;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_23 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_23 = cid;
        }
        int32_t condval_24;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_24 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_24 = cid;
        }
        int32_t condval_25;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_25 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_25 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(q_h_row[0], Q[((((((condval_23 / 4096) * 67108864) + ((condval_24 % 512) * 131072)) + (((condval_25 % 4096) / 512) * 1024)) + (vid * 512)) + (h_5 * 128))], 8192, 16, 128, bfloat16_t(0.000000e+00f));
        int32_t condval_26;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_26 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_26 = cid;
        }
        int32_t condval_27;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_27 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_27 = cid;
        }
        int32_t condval_28;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_28 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_28 = cid;
        }
        int32_t condval_29;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_29 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_29 = cid;
        }
        int32_t condval_30;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_30 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_30 = cid;
        }
        int32_t condval_31;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_31 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_31 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 16>(k_h_row[0], K[((((((condval_29 / 4096) * 67108864) + ((condval_30 % 512) * 131072)) + (((condval_31 % 4096) / 512) * 1024)) + (vid * 512)) + (h_5 * 128))], 8192, 16, 128, bfloat16_t(0.000000e+00f));
        int32_t condval_32;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_32 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_32 = cid;
        }
        int32_t condval_33;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_33 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_33 = cid;
        }
        int32_t condval_34;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_34 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_34 = cid;
        }
        int32_t condval_35;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_35 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_35 = cid;
        }
        int32_t condval_36;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_36 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_36 = cid;
        }
        int32_t condval_37;
        if (((tt_1 * 5) + (cid / 4)) < 1014) {
          condval_37 = (((tt_1 * 20) + cid) + 40);
        } else {
          condval_37 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 128, 16>(g2_row[0], GK[((((((condval_35 / 4096) * 67108864) + ((condval_36 % 512) * 131072)) + (((condval_37 % 4096) / 512) * 1024)) + (vid * 512)) + (h_5 * 128))], 8192, 16, 128, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
        AscendC::Cast<float, bfloat16_t, false>(q_f_row[0], q_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 2048);
        AscendC::Cast<float, bfloat16_t, false>(k_f_row[0], k_h_row[0], AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams(1, 1, 8, 4));
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        {
        AscendC::Muls<float, false>(g2_row[0], g2_row[0], 6.931472e-01f, AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::Broadcast<float, 2, 0, false>(bc_row[0],g2_row[1024],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){1, 128});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Sub<float, false>(e[0], bc_row[0], g2_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(kgj[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Sub<float, false>(e[0], g2_row[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Exp<float, false>(e[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::UnaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(q_f_row[0], q_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(e[0], k_f_row[0], e[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::CompareScalar(eq_m[0], j16b[0], float(h_5), AscendC::CMPMODE::EQ, 256);
        AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(beta_sel[0], eq_m[0], beta_f[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 256);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 16, 16, -1, 16, false>(beta_col[0], beta_sel[0], tmp_ub.ReinterpretCast<float>()[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::Broadcast<float, 2, 1, false>(bc_row[0],beta_col[0],tmp_ub[0],(uint32_t[]){16, 128}, (uint32_t[]){16, 1});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(e[0], e[0], bc_row[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[(h_5 * 4096)], q_f_row[0], 16, 128, 128, 16, 128, 128);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 2048>(a_agg[((h_5 * 4096) + 2048)], e[0], 16, 128, 128, 16, 128, 128);
        tl::ascend::copy_ub_to_ub<float, float, 2048>(b_agg[(h_5 * 2048)], kgj[0], 16, 128, 128, 16, 128, 128);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      tl::ascend::copy_ub_to_gm<float, 128, 128>(workspace_a[(((cid * 65536) + ((tt_1 % 2) * 32768)) + (vid * 16384))], a_agg[0], 128, 128, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
      tl::ascend::copy_ub_to_gm<float, 128, 64>(workspace_b[(((cid * 32768) + ((tt_1 % 2) * 16384)) + (vid * 8192))], b_agg[0], 128, 64, 128);
      AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* K_handle, uint8_t* GK_handle, uint8_t* Beta_handle, uint8_t* workspace_a_handle, uint8_t* workspace_b_handle, uint8_t* o_relay_handle, uint8_t* Aqk_handle, uint8_t* Akk_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(Q_handle, K_handle, GK_handle, Beta_handle, workspace_a_handle, workspace_b_handle, o_relay_handle, Aqk_handle, Akk_handle, fftsAddr);
}
