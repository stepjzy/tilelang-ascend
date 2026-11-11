#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR K3_handle,  GM_ADDR W3_handle,  GM_ADDR U3_handle,  GM_ADDR GT_handle,  GM_ADDR GD_handle,  GM_ADDR H0_handle,  GM_ADDR Hout_handle,  GM_ADDR Vnew_handle,  GM_ADDR Ht_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_4_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> K3;
  K3.SetGlobalBuffer((__gm__ half*)K3_handle);
  AscendC::GlobalTensor<half> W3;
  W3.SetGlobalBuffer((__gm__ half*)W3_handle);
  AscendC::GlobalTensor<half> U3;
  U3.SetGlobalBuffer((__gm__ half*)U3_handle);
  AscendC::GlobalTensor<float> GT;
  GT.SetGlobalBuffer((__gm__ float*)GT_handle);
  AscendC::GlobalTensor<float> GD;
  GD.SetGlobalBuffer((__gm__ float*)GD_handle);
  AscendC::GlobalTensor<float> H0;
  H0.SetGlobalBuffer((__gm__ float*)H0_handle);
  AscendC::GlobalTensor<half> Hout;
  Hout.SetGlobalBuffer((__gm__ half*)Hout_handle);
  AscendC::GlobalTensor<half> Vnew;
  Vnew.SetGlobalBuffer((__gm__ half*)Vnew_handle);
  AscendC::GlobalTensor<float> Ht;
  Ht.SetGlobalBuffer((__gm__ float*)Ht_handle);
  AscendC::GlobalTensor<half> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ half*)workspace_1_handle);
  AscendC::GlobalTensor<float> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ float*)workspace_2_handle);
  AscendC::GlobalTensor<half> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ half*)workspace_3_handle);
  AscendC::GlobalTensor<float> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ float*)workspace_4_handle);

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
  auto w_l1_a = ascend_l1.GetWithOffset<half>(8192, 0);
  auto h16_l1_a = ascend_l1.GetWithOffset<half>(8192, 16384);
  auto bv_frag_a = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto w_l1_b = ascend_l1.GetWithOffset<half>(8192, 32768);
  auto h16_l1_b = ascend_l1.GetWithOffset<half>(8192, 49152);
  auto bv_frag_b = ascend_l0c.GetWithOffset<float>(4096, 16384);
  auto k_l1_a = ascend_l1.GetWithOffset<half>(8192, 65536);
  auto v16_l1_a = ascend_l1.GetWithOffset<half>(4096, 81920);
  auto kh_frag_a = ascend_l0c.GetWithOffset<float>(8192, 32768);
  auto k_l1_b = ascend_l1.GetWithOffset<half>(8192, 90112);
  auto v16_l1_b = ascend_l1.GetWithOffset<half>(4096, 106496);
  auto kh_frag_b = ascend_l0c.GetWithOffset<float>(8192, 65536);
  auto h_ub_a = ascend_ub.GetWithOffset<float>(4096, 0);
  auto h_ub_b = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto h16_ub_a = ascend_ub.GetWithOffset<half>(4096, 32768);
  auto u16_ub_a = ascend_ub.GetWithOffset<half>(2048, 40960);
  auto gate_ub_a = ascend_ub.GetWithOffset<float>(32, 45056);
  auto dsrc_ub_a = ascend_ub.GetWithOffset<float>(64, 45184);
  auto h16_ub_b = ascend_ub.GetWithOffset<half>(4096, 45440);
  auto u16_ub_b = ascend_ub.GetWithOffset<half>(2048, 53632);
  auto gate_ub_b = ascend_ub.GetWithOffset<float>(32, 57728);
  auto dsrc_ub_b = ascend_ub.GetWithOffset<float>(64, 57856);
  auto u_ub = ascend_ub.GetWithOffset<float>(2048, 58112);
  auto bv_ub = ascend_ub.GetWithOffset<float>(2048, 66304);
  auto vn_ub = ascend_ub.GetWithOffset<float>(2048, 74496);
  auto vn_g_ub = ascend_ub.GetWithOffset<float>(2048, 83712);
  auto tmp_gate = ascend_ub.GetWithOffset<float>(256, 82688);
  auto vn16_ub_a = ascend_ub.GetWithOffset<half>(2048, 91904);
  auto vn16v_ub_a = ascend_ub.GetWithOffset<half>(2048, 96000);
  auto vn16_ub_b = ascend_ub.GetWithOffset<half>(2048, 100096);
  auto vn16v_ub_b = ascend_ub.GetWithOffset<half>(2048, 104192);
  auto kh_ub = ascend_ub.GetWithOffset<float>(4096, 108288);
  auto tmp_decay = ascend_ub.GetWithOffset<float>(512, 124672);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t kb = 0; kb < 4; ++kb) {
      for (int32_t i_t = 0; i_t < 128; ++i_t) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(1);
        AscendC::PipeBarrier<PIPE_MTE2>();
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
        tl::ascend::copy_gm_to_l1<half, 64, 128>(w_l1_a[0], W3[((((condval_2 / 128) * 67108864) + (i_t * 524288)) + (((condval_3 % 128) / 2) * 128))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(0);
        tl::ascend::copy_gm_to_l1<half, 128, 64>(h16_l1_a[0], workspace_1[((cid * 131072) + ((i_t % 8) * 16384))], 64, 128, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<half, float, 64, 64, 128, false, false, 128>(w_l1_a[0], h16_l1_a[0], bv_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_2[((cid * 65536) + ((i_t % 8) * 8192))], bv_frag_a[0], 64, 64, 64, 0);
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
        tl::ascend::copy_gm_to_l1<half, 64, 128>(w_l1_b[0], W3[((((condval_6 / 128) * 67108864) + (i_t * 524288)) + (((condval_7 % 128) / 2) * 128))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(1);
        tl::ascend::copy_gm_to_l1<half, 128, 64>(h16_l1_b[0], workspace_1[(((cid * 131072) + ((i_t % 8) * 16384)) + 8192)], 64, 128, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
        tl::ascend::gemm_v0<half, float, 64, 64, 128, false, false, 128>(w_l1_b[0], h16_l1_b[0], bv_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_2[(((cid * 65536) + ((i_t % 8) * 8192)) + 4096)], bv_frag_b[0], 64, 64, 64, 0);
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
        tl::ascend::copy_gm_to_l1<half, 64, 128>(k_l1_a[0], K3[((((condval_10 / 128) * 67108864) + (i_t * 524288)) + (((condval_11 % 128) / 2) * 128))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(4);
        tl::ascend::copy_gm_to_l1<half, 64, 64>(v16_l1_a[0], workspace_3[((cid * 65536) + ((i_t % 8) * 8192))], 64, 64, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(5);
        tl::ascend::gemm_v0<half, float, 128, 64, 64, true, false, 128>(k_l1_a[0], v16_l1_a[0], kh_frag_a[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6);
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_4[((cid * 131072) + ((i_t % 8) * 16384))], kh_frag_a[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(6);
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
        tl::ascend::copy_gm_to_l1<half, 64, 128>(k_l1_b[0], K3[((((condval_14 / 128) * 67108864) + (i_t * 524288)) + (((condval_15 % 128) / 2) * 128))], 8192, 64, 128, true);
        AscendC::CrossCoreWaitFlag(5);
        tl::ascend::copy_gm_to_l1<half, 64, 64>(v16_l1_b[0], workspace_3[(((cid * 65536) + ((i_t % 8) * 8192)) + 4096)], 64, 64, 64, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(7);
        tl::ascend::gemm_v0<half, float, 128, 64, 64, true, false, 128>(k_l1_b[0], v16_l1_b[0], kh_frag_b[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_4[(((cid * 131072) + ((i_t % 8) * 16384)) + 8192)], kh_frag_b[0], 64, 128, 64, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(7);
      }
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t kb_1 = 0; kb_1 < 4; ++kb_1) {
      AscendC::SetMaskCount();
      AscendC::SetVectorMask<uint8_t>((uint64_t)0, 4096);
      AscendC::Duplicate<float, false>(h_ub_a[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Duplicate<float, false>(h_ub_b[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      for (int32_t i_t_1 = 0; i_t_1 < 128; ++i_t_1) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        tl::ascend::copy_ub_to_ub<half, float, 4096>(h16_ub_a[0], h_ub_a[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_1[(((cid * 131072) + ((i_t_1 % 8) * 16384)) + (vid * 4096))], h16_ub_a[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
        int32_t condval_16;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_16 = ((kb_1 * 40) + cid);
        } else {
          condval_16 = cid;
        }
        int32_t condval_17;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_17 = ((kb_1 * 40) + cid);
        } else {
          condval_17 = cid;
        }
        int32_t condval_18;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_18 = ((kb_1 * 40) + cid);
        } else {
          condval_18 = cid;
        }
        int32_t condval_19;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_19 = ((kb_1 * 40) + cid);
        } else {
          condval_19 = cid;
        }
        int32_t condval_20;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_20 = ((kb_1 * 40) + cid);
        } else {
          condval_20 = cid;
        }
        int32_t condval_21;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_21 = ((kb_1 * 40) + cid);
        } else {
          condval_21 = cid;
        }
        tl::ascend::copy_ub_to_gm<half, 64, 64>(Hout[((((((condval_19 / 128) * 134217728) + (i_t_1 * 1048576)) + (((condval_20 % 128) / 2) * 16384)) + (vid * 8192)) + ((condval_21 % 2) * 64))], h16_ub_a[0], 128, 64, 64);
        int32_t condval_22;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_22 = ((kb_1 * 40) + cid);
        } else {
          condval_22 = cid;
        }
        int32_t condval_23;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_23 = ((kb_1 * 40) + cid);
        } else {
          condval_23 = cid;
        }
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
        tl::ascend::copy_gm_to_ub<half, 64, 32>(u16_ub_a[0], U3[(((((condval_24 / 128) * 67108864) + (i_t_1 * 524288)) + (vid * 262144)) + ((condval_25 % 128) * 64))], 8192, 32, 64, half(0.000000e+00f));
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
        tl::ascend::copy_gm_to_ub<float, 32>(gate_ub_a[0], GT[(((((condval_28 / 128) * 524288) + (i_t_1 * 4096)) + (((condval_29 % 128) / 2) * 64)) + (vid * 32))], 64, 1, 32, 0.000000e+00f);
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
        tl::ascend::copy_gm_to_ub<float, 64>(dsrc_ub_a[0], GD[(((((condval_32 / 128) * 1048576) + (i_t_1 * 8192)) + (((condval_33 % 128) / 2) * 128)) + (vid * 64))], 128, 1, 64, 0.000000e+00f);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 4096>(h16_ub_b[0], h_ub_b[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_1[((((cid * 131072) + ((i_t_1 % 8) * 16384)) + (vid * 4096)) + 8192)], h16_ub_b[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
        int32_t condval_34;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_34 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_34 = cid;
        }
        int32_t condval_35;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_35 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_35 = cid;
        }
        int32_t condval_36;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_36 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_36 = cid;
        }
        int32_t condval_37;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_37 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_37 = cid;
        }
        int32_t condval_38;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_38 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_38 = cid;
        }
        int32_t condval_39;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_39 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_39 = cid;
        }
        tl::ascend::copy_ub_to_gm<half, 64, 64>(Hout[((((((condval_37 / 128) * 134217728) + (i_t_1 * 1048576)) + (((condval_38 % 128) / 2) * 16384)) + (vid * 8192)) + ((condval_39 % 2) * 64))], h16_ub_b[0], 128, 64, 64);
        int32_t condval_40;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_40 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_40 = cid;
        }
        int32_t condval_41;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_41 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_41 = cid;
        }
        int32_t condval_42;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_42 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_42 = cid;
        }
        int32_t condval_43;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_43 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_43 = cid;
        }
        tl::ascend::copy_gm_to_ub<half, 64, 32>(u16_ub_b[0], U3[(((((condval_42 / 128) * 67108864) + (i_t_1 * 524288)) + (vid * 262144)) + ((condval_43 % 128) * 64))], 8192, 32, 64, half(0.000000e+00f));
        int32_t condval_44;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_44 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_44 = cid;
        }
        int32_t condval_45;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_45 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_45 = cid;
        }
        int32_t condval_46;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_46 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_46 = cid;
        }
        int32_t condval_47;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_47 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_47 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 32>(gate_ub_b[0], GT[(((((condval_46 / 128) * 524288) + (i_t_1 * 4096)) + (((condval_47 % 128) / 2) * 64)) + (vid * 32))], 64, 1, 32, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);
        int32_t condval_48;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_48 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_48 = cid;
        }
        int32_t condval_49;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_49 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_49 = cid;
        }
        int32_t condval_50;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_50 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_50 = cid;
        }
        int32_t condval_51;
        if (((kb_1 * 10) + (cid / 4)) < 27) {
          condval_51 = (((kb_1 * 40) + cid) + 20);
        } else {
          condval_51 = cid;
        }
        tl::ascend::copy_gm_to_ub<float, 64>(dsrc_ub_b[0], GD[(((((condval_50 / 128) * 1048576) + (i_t_1 * 8192)) + (((condval_51 % 128) / 2) * 128)) + (vid * 64))], 128, 1, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        tl::ascend::copy_ub_to_ub<float, half, 2048>(u_ub[0], u16_ub_a[0], 32, 64, 64, 32, 64, 64);
        AscendC::CrossCoreWaitFlag(2);
        tl::ascend::copy_gm_to_ub<float, 64, 32>(bv_ub[0], workspace_2[(((cid * 65536) + ((i_t_1 % 8) * 8192)) + (vid * 2048))], 64, 32, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(vn_ub[0], u_ub[0], bv_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_gate[0], gate_ub_a[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(vn_g_ub[0], vn_ub[0], tmp_gate[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 2048>(vn16_ub_a[0], vn_g_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<half, 64, 32>(workspace_3[(((cid * 65536) + ((i_t_1 % 8) * 8192)) + (vid * 2048))], vn16_ub_a[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
        tl::ascend::copy_ub_to_ub<half, float, 2048>(vn16v_ub_a[0], vn_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
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
        int32_t condval_54;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_54 = ((kb_1 * 40) + cid);
        } else {
          condval_54 = cid;
        }
        int32_t condval_55;
        if (((kb_1 * 5) + (cid / 8)) < 16) {
          condval_55 = ((kb_1 * 40) + cid);
        } else {
          condval_55 = cid;
        }
        tl::ascend::copy_ub_to_gm<half, 64, 32>(Vnew[(((((condval_54 / 128) * 67108864) + (i_t_1 * 524288)) + (vid * 262144)) + ((condval_55 % 128) * 64))], vn16v_ub_a[0], 8192, 32, 64);
        tl::ascend::copy_ub_to_ub<float, half, 2048>(u_ub[0], u16_ub_b[0], 32, 64, 64, 32, 64, 64);
        AscendC::CrossCoreWaitFlag(3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 32>(bv_ub[0], workspace_2[((((cid * 65536) + ((i_t_1 % 8) * 8192)) + (vid * 2048)) + 4096)], 64, 32, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(vn_ub[0], u_ub[0], bv_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::brcb<float>(tmp_gate[0], gate_ub_b[0], 4, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(vn_g_ub[0], vn_ub[0], tmp_gate[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 2048>(vn16_ub_b[0], vn_g_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<half, 64, 32>(workspace_3[((((cid * 65536) + ((i_t_1 % 8) * 8192)) + (vid * 2048)) + 4096)], vn16_ub_b[0], 64, 32, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(5);
        tl::ascend::copy_ub_to_ub<half, float, 2048>(vn16v_ub_b[0], vn_ub[0], 32, 64, 64, 32, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
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
        tl::ascend::copy_ub_to_gm<half, 64, 32>(Vnew[(((((condval_58 / 128) * 67108864) + (i_t_1 * 524288)) + (vid * 262144)) + ((condval_59 % 128) * 64))], vn16v_ub_b[0], 8192, 32, 64);
        AscendC::CrossCoreWaitFlag(6);
        tl::ascend::copy_gm_to_ub<float, 64, 64>(kh_ub[0], workspace_4[(((cid * 131072) + ((i_t_1 % 8) * 16384)) + (vid * 4096))], 64, 64, 64, 0.000000e+00f);
        tl::ascend::brcb<float>(tmp_decay[0], dsrc_ub_a[0], 8, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(h_ub_a[0], h_ub_a[0], tmp_decay[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(h_ub_a[0], h_ub_a[0], kh_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::CrossCoreWaitFlag(7);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 64>(kh_ub[0], workspace_4[((((cid * 131072) + ((i_t_1 % 8) * 16384)) + (vid * 4096)) + 8192)], 64, 64, 64, 0.000000e+00f);
        tl::ascend::brcb<float>(tmp_decay[0], dsrc_ub_b[0], 8, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::mul_mask<float>(h_ub_b[0], h_ub_b[0], tmp_decay[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(h_ub_b[0], h_ub_b[0], kh_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
      }
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* K3_handle, uint8_t* W3_handle, uint8_t* U3_handle, uint8_t* GT_handle, uint8_t* GD_handle, uint8_t* H0_handle, uint8_t* Hout_handle, uint8_t* Vnew_handle, uint8_t* Ht_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_4_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(K3_handle, W3_handle, U3_handle, GT_handle, GD_handle, H0_handle, Hout_handle, Vnew_handle, Ht_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, fftsAddr);
}
