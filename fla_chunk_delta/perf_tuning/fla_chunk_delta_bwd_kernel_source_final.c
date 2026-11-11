#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q3_handle,  GM_ADDR K3_handle,  GM_ADDR W3_handle,  GM_ADDR Do3_handle,  GM_ADDR Dv3_handle,  GM_ADDR GT_handle,  GM_ADDR GE_handle,  GM_ADDR GD_handle,  GM_ADDR Dht_handle,  GM_ADDR Dh_handle,  GM_ADDR Dh0_handle,  GM_ADDR Dv2_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_3b_handle,  GM_ADDR workspace_4_handle,  GM_ADDR workspace_5_handle,  GM_ADDR workspace_6_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<bfloat16_t> Q3;
  Q3.SetGlobalBuffer((__gm__ bfloat16_t*)Q3_handle);
  AscendC::GlobalTensor<bfloat16_t> K3;
  K3.SetGlobalBuffer((__gm__ bfloat16_t*)K3_handle);
  AscendC::GlobalTensor<bfloat16_t> W3;
  W3.SetGlobalBuffer((__gm__ bfloat16_t*)W3_handle);
  AscendC::GlobalTensor<bfloat16_t> Do3;
  Do3.SetGlobalBuffer((__gm__ bfloat16_t*)Do3_handle);
  AscendC::GlobalTensor<bfloat16_t> Dv3;
  Dv3.SetGlobalBuffer((__gm__ bfloat16_t*)Dv3_handle);
  AscendC::GlobalTensor<float> GT;
  GT.SetGlobalBuffer((__gm__ float*)GT_handle);
  AscendC::GlobalTensor<float> GE;
  GE.SetGlobalBuffer((__gm__ float*)GE_handle);
  AscendC::GlobalTensor<float> GD;
  GD.SetGlobalBuffer((__gm__ float*)GD_handle);
  AscendC::GlobalTensor<float> Dht;
  Dht.SetGlobalBuffer((__gm__ float*)Dht_handle);
  AscendC::GlobalTensor<bfloat16_t> Dh;
  Dh.SetGlobalBuffer((__gm__ bfloat16_t*)Dh_handle);
  AscendC::GlobalTensor<float> Dh0;
  Dh0.SetGlobalBuffer((__gm__ float*)Dh0_handle);
  AscendC::GlobalTensor<bfloat16_t> Dv2;
  Dv2.SetGlobalBuffer((__gm__ bfloat16_t*)Dv2_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_1_handle);
  AscendC::GlobalTensor<float> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ float*)workspace_2_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_3_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_3b;
  workspace_3b.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_3b_handle);
  AscendC::GlobalTensor<bfloat16_t> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ bfloat16_t*)workspace_4_handle);
  AscendC::GlobalTensor<float> workspace_5;
  workspace_5.SetGlobalBuffer((__gm__ float*)workspace_5_handle);
  AscendC::GlobalTensor<float> workspace_6;
  workspace_6.SetGlobalBuffer((__gm__ float*)workspace_6_handle);

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
  auto k_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 0);
  auto dh16_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 16384);
  auto dvp_frag_a = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto k_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 32768);
  auto dh16_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 49152);
  auto dvp_frag_b = ascend_l0c.GetWithOffset<float>(4096, 16384);
  auto do_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(4096, 65536);
  auto qt16_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 73728);
  auto qw_frag_a = ascend_l0c.GetWithOffset<float>(8192, 32768);
  auto do_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(4096, 90112);
  auto qt16_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 98304);
  auto qw_frag_b = ascend_l0c.GetWithOffset<float>(8192, 65536);
  auto w_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(8192, 114688);
  auto dv2_16_l1_a = ascend_l1.GetWithOffset<bfloat16_t>(4096, 131072);
  auto w_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(8192, 139264);
  auto dv2_16_l1_b = ascend_l1.GetWithOffset<bfloat16_t>(4096, 155648);
  auto dh_ub_a = ascend_ub.GetWithOffset<float>(4096, 0);
  auto dh_ub_b = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto dh16_ub_a = ascend_ub.GetWithOffset<bfloat16_t>(4096, 32768);
  auto dv16_ub_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 40960);
  auto gate_ub_a = ascend_ub.GetWithOffset<float>(32, 45056);
  auto gexp_ub_a = ascend_ub.GetWithOffset<float>(32, 45184);
  auto dsrc_ub_a = ascend_ub.GetWithOffset<float>(64, 45312);
  auto q_half_a_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 45568);
  auto q_half_b_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 49664);
  auto q_ub_a = ascend_ub.GetWithOffset<float>(2048, 53760);
  auto tmp_qa = ascend_ub.GetWithOffset<float>(256, 61952);
  auto qt16_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 62976);
  auto q_ub_b = ascend_ub.GetWithOffset<float>(2048, 67072);
  auto tmp_qb = ascend_ub.GetWithOffset<float>(256, 75264);
  auto qt16_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 76288);
  auto dh16_ub_b = ascend_ub.GetWithOffset<bfloat16_t>(4096, 80384);
  auto dv16_ub_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 88576);
  auto gate_ub_b = ascend_ub.GetWithOffset<float>(32, 92672);
  auto gexp_ub_b = ascend_ub.GetWithOffset<float>(32, 92800);
  auto dsrc_ub_b = ascend_ub.GetWithOffset<float>(64, 92928);
  auto q_half_a_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 93184);
  auto q_half_b_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 97280);
  auto dvp_ub = ascend_ub.GetWithOffset<float>(2048, 101376);
  auto tmp_gate = ascend_ub.GetWithOffset<float>(256, 109568);
  auto dv_ub = ascend_ub.GetWithOffset<float>(2048, 110592);
  auto dv2_ub = ascend_ub.GetWithOffset<float>(2048, 118784);
  auto dv2_16_ub_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 126976);
  auto dv2_16_ub_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 131072);
  auto qdw_ub = ascend_ub.GetWithOffset<float>(4096, 135168);
  auto tmp_decay = ascend_ub.GetWithOffset<float>(512, 151552);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t kb = 0; kb < 4; ++kb) {
      for (int32_t it_f = 0; it_f < 128; ++it_f) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(6);
        int32_t condval;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval = ((kb * 40) + cid);
        } else {
          condval = cid;
        }
        int32_t condval_1;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_1 = ((kb * 40) + cid);
        } else {
          condval_1 = cid;
        }
        int32_t condval_2;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_2 = ((kb * 40) + cid);
        } else {
          condval_2 = cid;
        }
        int32_t condval_3;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_3 = ((kb * 40) + cid);
        } else {
          condval_3 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(k_l1_a[0], K3[(((((condval_2 / 128) * 67108864) + (((condval_3 % 128) / 2) * 128)) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(0);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 64>(dh16_l1_a[0], workspace_1[((cid * 131072) + ((it_f % 8) * 16384))], 64, 128, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 128, false, false, 128>(k_l1_a[0], dh16_l1_a[0], dvp_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_2[((cid * 65536) + ((it_f % 8) * 8192))], dvp_frag_a[0], 64, 64, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(2);
        int32_t condval_4;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_4 = (((kb * 40) + cid) + 20);
        } else {
          condval_4 = cid;
        }
        int32_t condval_5;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_5 = (((kb * 40) + cid) + 20);
        } else {
          condval_5 = cid;
        }
        int32_t condval_6;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_6 = (((kb * 40) + cid) + 20);
        } else {
          condval_6 = cid;
        }
        int32_t condval_7;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_7 = (((kb * 40) + cid) + 20);
        } else {
          condval_7 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(k_l1_b[0], K3[(((((condval_6 / 128) * 67108864) + (((condval_7 % 128) / 2) * 128)) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(1);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 64>(dh16_l1_b[0], workspace_1[(((cid * 131072) + ((it_f % 8) * 16384)) + 8192)], 64, 128, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
        tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 128, false, false, 128>(k_l1_b[0], dh16_l1_b[0], dvp_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_2[(((cid * 65536) + ((it_f % 8) * 8192)) + 4096)], dvp_frag_b[0], 64, 64, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(3);
        int32_t condval_8;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_8 = ((kb * 40) + cid);
        } else {
          condval_8 = cid;
        }
        int32_t condval_9;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_9 = ((kb * 40) + cid);
        } else {
          condval_9 = cid;
        }
        int32_t condval_10;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_10 = ((kb * 40) + cid);
        } else {
          condval_10 = cid;
        }
        int32_t condval_11;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_11 = ((kb * 40) + cid);
        } else {
          condval_11 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(do_l1_a[0], Do3[(((((condval_10 / 128) * 67108864) + ((condval_11 % 128) * 64)) + 66584576) - (it_f * 524288))], 8192, 64, 64, true);
        AscendC::CrossCoreWaitFlag(4);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1_a[0], workspace_3[((cid * 65536) + ((it_f % 8) * 8192))], 64, 64, 64, true);
        AscendC::CrossCoreWaitFlag(6);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1_a[4096], workspace_3b[((cid * 65536) + ((it_f % 8) * 8192))], 64, 64, 64, false);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(5);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(qt16_l1_a[0], do_l1_a[0], qw_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_5[((cid * 131072) + ((it_f % 8) * 16384))], qw_frag_a[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(10);
        int32_t condval_12;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_12 = (((kb * 40) + cid) + 20);
        } else {
          condval_12 = cid;
        }
        int32_t condval_13;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_13 = (((kb * 40) + cid) + 20);
        } else {
          condval_13 = cid;
        }
        int32_t condval_14;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_14 = (((kb * 40) + cid) + 20);
        } else {
          condval_14 = cid;
        }
        int32_t condval_15;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_15 = (((kb * 40) + cid) + 20);
        } else {
          condval_15 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(do_l1_b[0], Do3[(((((condval_14 / 128) * 67108864) + ((condval_15 % 128) * 64)) + 66584576) - (it_f * 524288))], 8192, 64, 64, true);
        AscendC::CrossCoreWaitFlag(5);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1_b[0], workspace_3[(((cid * 65536) + ((it_f % 8) * 8192)) + 4096)], 64, 64, 64, true);
        AscendC::CrossCoreWaitFlag(7);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1_b[4096], workspace_3b[(((cid * 65536) + ((it_f % 8) * 8192)) + 4096)], 64, 64, 64, false);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(7);
        tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(qt16_l1_b[0], do_l1_b[0], qw_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_5[(((cid * 131072) + ((it_f % 8) * 16384)) + 8192)], qw_frag_b[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(11);
        int32_t condval_16;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_16 = ((kb * 40) + cid);
        } else {
          condval_16 = cid;
        }
        int32_t condval_17;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_17 = ((kb * 40) + cid);
        } else {
          condval_17 = cid;
        }
        int32_t condval_18;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_18 = ((kb * 40) + cid);
        } else {
          condval_18 = cid;
        }
        int32_t condval_19;
        if (((kb * 5) + (cid / 8)) < 16) {
          condval_19 = ((kb * 40) + cid);
        } else {
          condval_19 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(w_l1_a[0], W3[(((((condval_18 / 128) * 67108864) + (((condval_19 % 128) / 2) * 128)) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(8);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(dv2_16_l1_a[0], workspace_4[((cid * 65536) + ((it_f % 8) * 8192))], 64, 64, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(2);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(w_l1_a[0], dv2_16_l1_a[0], qw_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_6[((cid * 131072) + ((it_f % 8) * 16384))], qw_frag_a[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(12);
        int32_t condval_20;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_20 = (((kb * 40) + cid) + 20);
        } else {
          condval_20 = cid;
        }
        int32_t condval_21;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_21 = (((kb * 40) + cid) + 20);
        } else {
          condval_21 = cid;
        }
        int32_t condval_22;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_22 = (((kb * 40) + cid) + 20);
        } else {
          condval_22 = cid;
        }
        int32_t condval_23;
        if (((kb * 10) + (cid / 4)) < 27) {
          condval_23 = (((kb * 40) + cid) + 20);
        } else {
          condval_23 = cid;
        }
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(w_l1_b[0], W3[(((((condval_22 / 128) * 67108864) + (((condval_23 % 128) / 2) * 128)) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(9);
        tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(dv2_16_l1_b[0], workspace_4[(((cid * 65536) + ((it_f % 8) * 8192)) + 4096)], 64, 64, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(4);
        tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(w_l1_b[0], dv2_16_l1_b[0], qw_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(5);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(5);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_6[(((cid * 131072) + ((it_f % 8) * 16384)) + 8192)], qw_frag_b[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(13);
      }
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t kb_1 = 0; kb_1 < 4; ++kb_1) {
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 4096);
      AscendC::Duplicate<float, false>(dh_ub_a[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Duplicate<float, false>(dh_ub_b[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      for (int32_t it_f_1 = 0; it_f_1 < 128; ++it_f_1) {
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(dh16_ub_a[0], dh_ub_a[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(workspace_1[(((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096))], dh16_ub_a[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
        int32_t condval_24;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_24 = ((kb_1 * 40) + cid);
        } else {
          condval_24 = cid;
        }
        int32_t condval_25;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_25 = ((kb_1 * 40) + cid);
        } else {
          condval_25 = cid;
        }
        int32_t condval_26;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_26 = ((kb_1 * 40) + cid);
        } else {
          condval_26 = cid;
        }
        int32_t condval_27;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_27 = ((kb_1 * 40) + cid);
        } else {
          condval_27 = cid;
        }
        int32_t condval_28;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_28 = ((kb_1 * 40) + cid);
        } else {
          condval_28 = cid;
        }
        int32_t condval_29;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_29 = ((kb_1 * 40) + cid);
        } else {
          condval_29 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(Dh[(((((((condval_27 / 128) * 134217728) + (((condval_28 % 128) / 2) * 16384)) + (vid * 8192)) + ((condval_29 % 2) * 64)) + 133169152) - (it_f_1 * 1048576))], dh16_ub_a[0], 128, 64, 64);
        int32_t condval_30;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_30 = ((kb_1 * 40) + cid);
        } else {
          condval_30 = cid;
        }
        int32_t condval_31;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_31 = ((kb_1 * 40) + cid);
        } else {
          condval_31 = cid;
        }
        int32_t condval_32;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_32 = ((kb_1 * 40) + cid);
        } else {
          condval_32 = cid;
        }
        int32_t condval_33;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_33 = ((kb_1 * 40) + cid);
        } else {
          condval_33 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(dv16_ub_a[0], Dv3[((((((condval_32 / 128) * 67108864) + (vid * 262144)) + ((condval_33 % 128) * 64)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        int32_t condval_34;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_34 = ((kb_1 * 40) + cid);
        } else {
          condval_34 = cid;
        }
        int32_t condval_35;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_35 = ((kb_1 * 40) + cid);
        } else {
          condval_35 = cid;
        }
        int32_t condval_36;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_36 = ((kb_1 * 40) + cid);
        } else {
          condval_36 = cid;
        }
        int32_t condval_37;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_37 = ((kb_1 * 40) + cid);
        } else {
          condval_37 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 32>(gate_ub_a[0], GT[((((((condval_36 / 128) * 524288) + (((condval_37 % 128) / 2) * 64)) + (vid * 32)) + 520192) - (it_f_1 * 4096))], 64, 1, 32, 0.000000e+00f);
        int32_t condval_38;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_38 = ((kb_1 * 40) + cid);
        } else {
          condval_38 = cid;
        }
        int32_t condval_39;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_39 = ((kb_1 * 40) + cid);
        } else {
          condval_39 = cid;
        }
        int32_t condval_40;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_40 = ((kb_1 * 40) + cid);
        } else {
          condval_40 = cid;
        }
        int32_t condval_41;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_41 = ((kb_1 * 40) + cid);
        } else {
          condval_41 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 32>(gexp_ub_a[0], GE[((((((condval_40 / 128) * 524288) + (((condval_41 % 128) / 2) * 64)) + (vid * 32)) + 520192) - (it_f_1 * 4096))], 64, 1, 32, 0.000000e+00f);
        int32_t condval_42;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_42 = ((kb_1 * 40) + cid);
        } else {
          condval_42 = cid;
        }
        int32_t condval_43;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_43 = ((kb_1 * 40) + cid);
        } else {
          condval_43 = cid;
        }
        int32_t condval_44;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_44 = ((kb_1 * 40) + cid);
        } else {
          condval_44 = cid;
        }
        int32_t condval_45;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_45 = ((kb_1 * 40) + cid);
        } else {
          condval_45 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 64>(dsrc_ub_a[0], GD[((((((condval_44 / 128) * 1048576) + (((condval_45 % 128) / 2) * 128)) + (vid * 64)) + 1040384) - (it_f_1 * 8192))], 128, 1, 64, 0.000000e+00f);
        int32_t condval_46;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_46 = ((kb_1 * 40) + cid);
        } else {
          condval_46 = cid;
        }
        int32_t condval_47;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_47 = ((kb_1 * 40) + cid);
        } else {
          condval_47 = cid;
        }
        int32_t condval_48;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_48 = ((kb_1 * 40) + cid);
        } else {
          condval_48 = cid;
        }
        int32_t condval_49;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_49 = ((kb_1 * 40) + cid);
        } else {
          condval_49 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_a_a[0], Q3[((((((condval_48 / 128) * 67108864) + (vid * 262144)) + (((condval_49 % 128) / 2) * 128)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        int32_t condval_50;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_50 = ((kb_1 * 40) + cid);
        } else {
          condval_50 = cid;
        }
        int32_t condval_51;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_51 = ((kb_1 * 40) + cid);
        } else {
          condval_51 = cid;
        }
        int32_t condval_52;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_52 = ((kb_1 * 40) + cid);
        } else {
          condval_52 = cid;
        }
        int32_t condval_53;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_53 = ((kb_1 * 40) + cid);
        } else {
          condval_53 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_b_a[0], Q3[((((((condval_52 / 128) * 67108864) + (vid * 262144)) + (((condval_53 % 128) / 2) * 128)) + 66584640) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_a[0], q_half_a_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_qa[0], gexp_ub_a[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(q_ub_a[0], q_ub_a[0], tmp_qa[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_a[0], q_ub_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[(((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048))], qt16_a[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_b[0], q_half_b_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_qb[0], gexp_ub_a[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(q_ub_b[0], q_ub_b[0], tmp_qb[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_b[0], q_ub_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3b[(((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048))], qt16_b[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(6);
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(dh16_ub_b[0], dh_ub_b[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(workspace_1[((((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096)) + 8192)], dh16_ub_b[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
        int32_t condval_54;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_54 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_54 = cid;
        }
        int32_t condval_55;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_55 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_55 = cid;
        }
        int32_t condval_56;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_56 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_56 = cid;
        }
        int32_t condval_57;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_57 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_57 = cid;
        }
        int32_t condval_58;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_58 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_58 = cid;
        }
        int32_t condval_59;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_59 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_59 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(Dh[(((((((condval_57 / 128) * 134217728) + (((condval_58 % 128) / 2) * 16384)) + (vid * 8192)) + ((condval_59 % 2) * 64)) + 133169152) - (it_f_1 * 1048576))], dh16_ub_b[0], 128, 64, 64);
        int32_t condval_60;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_60 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_60 = cid;
        }
        int32_t condval_61;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_61 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_61 = cid;
        }
        int32_t condval_62;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_62 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_62 = cid;
        }
        int32_t condval_63;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_63 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_63 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(dv16_ub_b[0], Dv3[((((((condval_62 / 128) * 67108864) + (vid * 262144)) + ((condval_63 % 128) * 64)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        int32_t condval_64;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_64 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_64 = cid;
        }
        int32_t condval_65;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_65 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_65 = cid;
        }
        int32_t condval_66;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_66 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_66 = cid;
        }
        int32_t condval_67;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_67 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_67 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 32>(gate_ub_b[0], GT[((((((condval_66 / 128) * 524288) + (((condval_67 % 128) / 2) * 64)) + (vid * 32)) + 520192) - (it_f_1 * 4096))], 64, 1, 32, 0.000000e+00f);
        int32_t condval_68;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_68 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_68 = cid;
        }
        int32_t condval_69;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_69 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_69 = cid;
        }
        int32_t condval_70;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_70 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_70 = cid;
        }
        int32_t condval_71;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_71 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_71 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 32>(gexp_ub_b[0], GE[((((((condval_70 / 128) * 524288) + (((condval_71 % 128) / 2) * 64)) + (vid * 32)) + 520192) - (it_f_1 * 4096))], 64, 1, 32, 0.000000e+00f);
        int32_t condval_72;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_72 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_72 = cid;
        }
        int32_t condval_73;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_73 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_73 = cid;
        }
        int32_t condval_74;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_74 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_74 = cid;
        }
        int32_t condval_75;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_75 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_75 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 64>(dsrc_ub_b[0], GD[((((((condval_74 / 128) * 1048576) + (((condval_75 % 128) / 2) * 128)) + (vid * 64)) + 1040384) - (it_f_1 * 8192))], 128, 1, 64, 0.000000e+00f);
        int32_t condval_76;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_76 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_76 = cid;
        }
        int32_t condval_77;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_77 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_77 = cid;
        }
        int32_t condval_78;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_78 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_78 = cid;
        }
        int32_t condval_79;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_79 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_79 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_a_b[0], Q3[((((((condval_78 / 128) * 67108864) + (vid * 262144)) + (((condval_79 % 128) / 2) * 128)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        int32_t condval_80;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_80 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_80 = cid;
        }
        int32_t condval_81;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_81 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_81 = cid;
        }
        int32_t condval_82;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_82 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_82 = cid;
        }
        int32_t condval_83;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_83 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_83 = cid;
        }
        tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_b_b[0], Q3[((((((condval_82 / 128) * 67108864) + (vid * 262144)) + (((condval_83 % 128) / 2) * 128)) + 66584640) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_a[0], q_half_a_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_qa[0], gexp_ub_b[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(q_ub_a[0], q_ub_a[0], tmp_qa[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_a[0], q_ub_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048)) + 4096)], qt16_a[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_b[0], q_half_b_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_qb[0], gexp_ub_b[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(q_ub_b[0], q_ub_b[0], tmp_qb[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_b[0], q_ub_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3b[((((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048)) + 4096)], qt16_b[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(7);
        AscendC::CrossCoreWaitFlag(2);
        tl::ascend::copy_gm_to_ub<float, 64, 32>(dvp_ub[0], workspace_2[(((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048))], 64, 32, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        tl::ascend::brcb<float>(tmp_gate[0], gate_ub_a[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(dvp_ub[0], dvp_ub[0], tmp_gate[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(dv_ub[0], dv16_ub_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(dv2_ub[0], dv_ub[0], dvp_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(dv2_16_ub_a[0], dv2_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_4[(((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048))], dv2_16_ub_a[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
        int32_t condval_84;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_84 = ((kb_1 * 40) + cid);
        } else {
          condval_84 = cid;
        }
        int32_t condval_85;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_85 = ((kb_1 * 40) + cid);
        } else {
          condval_85 = cid;
        }
        int32_t condval_86;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_86 = ((kb_1 * 40) + cid);
        } else {
          condval_86 = cid;
        }
        int32_t condval_87;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_87 = ((kb_1 * 40) + cid);
        } else {
          condval_87 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(Dv2[((((((condval_86 / 128) * 67108864) + (vid * 262144)) + ((condval_87 % 128) * 64)) + 66584576) - (it_f_1 * 524288))], dv2_16_ub_a[0], 8192, 32, 64);
        AscendC::CrossCoreWaitFlag(3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        tl::ascend::copy_gm_to_ub<float, 64, 32>(dvp_ub[0], workspace_2[((((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048)) + 4096)], 64, 32, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
        tl::ascend::brcb<float>(tmp_gate[0], gate_ub_b[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(dvp_ub[0], dvp_ub[0], tmp_gate[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(dv_ub[0], dv16_ub_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(dv2_ub[0], dv_ub[0], dvp_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(dv2_16_ub_b[0], dv2_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_4[((((cid * 65536) + ((it_f_1 % 8) * 8192)) + (vid * 2048)) + 4096)], dv2_16_ub_b[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        int32_t condval_88;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_88 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_88 = cid;
        }
        int32_t condval_89;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_89 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_89 = cid;
        }
        int32_t condval_90;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_90 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_90 = cid;
        }
        int32_t condval_91;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_91 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_91 = cid;
        }
        tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(Dv2[((((((condval_90 / 128) * 67108864) + (vid * 262144)) + ((condval_91 % 128) * 64)) + 66584576) - (it_f_1 * 524288))], dv2_16_ub_b[0], 8192, 32, 64);
        AscendC::CrossCoreWaitFlag(10);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 64>(qdw_ub[0], workspace_5[(((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096))], 64, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        {
        AscendC::Muls<float, false>(qdw_ub[0], qdw_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
        }
        tl::ascend::brcb<float>(tmp_decay[0], dsrc_ub_a[0], 8, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(dh_ub_a[0], dh_ub_a[0], tmp_decay[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(dh_ub_a[0], dh_ub_a[0], qdw_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::CrossCoreWaitFlag(12);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        tl::ascend::copy_gm_to_ub<float, 64, 64>(qdw_ub[0], workspace_6[(((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096))], 64, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(dh_ub_a[0], dh_ub_a[0], qdw_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::CrossCoreWaitFlag(11);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 64>(qdw_ub[0], workspace_5[((((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096)) + 8192)], 64, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
        {
        AscendC::Muls<float, false>(qdw_ub[0], qdw_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
        }
        tl::ascend::brcb<float>(tmp_decay[0], dsrc_ub_b[0], 8, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(dh_ub_b[0], dh_ub_b[0], tmp_decay[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(dh_ub_b[0], dh_ub_b[0], qdw_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::CrossCoreWaitFlag(13);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        tl::ascend::copy_gm_to_ub<float, 64, 64>(qdw_ub[0], workspace_6[((((cid * 131072) + ((it_f_1 % 8) * 16384)) + (vid * 4096)) + 8192)], 64, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(dh_ub_b[0], dh_ub_b[0], qdw_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      }
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q3_handle, uint8_t* K3_handle, uint8_t* W3_handle, uint8_t* Do3_handle, uint8_t* Dv3_handle, uint8_t* GT_handle, uint8_t* GE_handle, uint8_t* GD_handle, uint8_t* Dht_handle, uint8_t* Dh_handle, uint8_t* Dh0_handle, uint8_t* Dv2_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_3b_handle, uint8_t* workspace_4_handle, uint8_t* workspace_5_handle, uint8_t* workspace_6_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(Q3_handle, K3_handle, W3_handle, Do3_handle, Dv3_handle, GT_handle, GE_handle, GD_handle, Dht_handle, Dh_handle, Dh0_handle, Dv2_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_3b_handle, workspace_4_handle, workspace_5_handle, workspace_6_handle, fftsAddr);
}
