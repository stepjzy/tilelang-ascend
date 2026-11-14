#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR V_handle,  GM_ADDR BetaFlat_handle,  GM_ADDR A_handle,  GM_ADDR du_handle,  GM_ADDR dbeta_flat_handle,  GM_ADDR dbeta_flat_v_handle,  GM_ADDR dv_handle,  GM_ADDR dbeta_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_4_handle,  GM_ADDR workspace_5_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> V;
  V.SetGlobalBuffer((__gm__ float*)V_handle);
  AscendC::GlobalTensor<float> BetaFlat;
  BetaFlat.SetGlobalBuffer((__gm__ float*)BetaFlat_handle);
  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> du;
  du.SetGlobalBuffer((__gm__ float*)du_handle);
  AscendC::GlobalTensor<float> dbeta_flat;
  dbeta_flat.SetGlobalBuffer((__gm__ float*)dbeta_flat_handle);
  AscendC::GlobalTensor<float> dbeta_flat_v;
  dbeta_flat_v.SetGlobalBuffer((__gm__ float*)dbeta_flat_v_handle);
  AscendC::GlobalTensor<float> dv;
  dv.SetGlobalBuffer((__gm__ float*)dv_handle);
  AscendC::GlobalTensor<float> dbeta;
  dbeta.SetGlobalBuffer((__gm__ float*)dbeta_handle);
  AscendC::GlobalTensor<half> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ half*)workspace_1_handle);
  AscendC::GlobalTensor<half> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ half*)workspace_2_handle);
  AscendC::GlobalTensor<half> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ half*)workspace_3_handle);
  AscendC::GlobalTensor<half> workspace_4;
  workspace_4.SetGlobalBuffer((__gm__ half*)workspace_4_handle);
  AscendC::GlobalTensor<float> workspace_5;
  workspace_5.SetGlobalBuffer((__gm__ float*)workspace_5_handle);

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
    }
  auto a_hi_l1 = ascend_l1.GetWithOffset<half>(4096, 0);
  auto du_hi_l1 = ascend_l1.GetWithOffset<half>(4096, 8192);
  auto y_frag = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto du_lo_l1 = ascend_l1.GetWithOffset<half>(4096, 16384);
  auto a_lo_l1 = ascend_l1.GetWithOffset<half>(4096, 24576);
  auto beta_tile = ascend_ub.GetWithOffset<float>(512, 0);
  auto off_i32 = ascend_ub.GetWithOffset<int>(64, 2048);
  auto off_u32 = ascend_ub.GetWithOffset<uint32_t>(64, 2304);
  auto bt = ascend_ub.GetWithOffset<float>(64, 2560);
  auto acc = ascend_ub.GetWithOffset<float>(64, 2816);
  auto v_ub = ascend_ub.GetWithOffset<float>(8192, 3072);
  auto a_f32 = ascend_ub.GetWithOffset<float>(4096, 35840);
  auto a_hi_ub = ascend_ub.GetWithOffset<half>(4096, 52224);
  auto a_hi_f32 = ascend_ub.GetWithOffset<float>(4096, 60416);
  auto a_lo_f32 = ascend_ub.GetWithOffset<float>(4096, 76800);
  auto a_lo_ub = ascend_ub.GetWithOffset<half>(4096, 93184);
  auto du_f32 = ascend_ub.GetWithOffset<float>(4096, 101376);
  auto du_hi_ub = ascend_ub.GetWithOffset<half>(4096, 117760);
  auto du_hi_f32 = ascend_ub.GetWithOffset<float>(4096, 125952);
  auto du_lo_f32 = ascend_ub.GetWithOffset<float>(4096, 142336);
  auto du_lo_ub = ascend_ub.GetWithOffset<half>(4096, 158720);
  auto bt2d = ascend_ub.GetWithOffset<float>(4096, 166912);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(4384, 183296);
  auto y_ub = ascend_ub.GetWithOffset<float>(4096, 35840);
  auto dv_ub = ascend_ub.GetWithOffset<float>(4096, 52224);
  auto r_tmpv = ascend_ub.GetWithOffset<float>(4096, 52224);
  auto r_part = ascend_ub.GetWithOffset<float>(64, 187680);
  auto dbeta2 = ascend_ub.GetWithOffset<float>(64, 187936);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t i_v_outer = 0; i_v_outer < 1; ++i_v_outer) {
      for (int32_t i = 0; i < 2; ++i) {
        AscendC::CrossCoreWaitFlag(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(5);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_l1<half, 64, 64>(a_hi_l1[0], workspace_1[((i * 16777216) + (cid * 4096))], 64, 0, 0, true);
        AscendC::CrossCoreWaitFlag(2);
        tl::ascend::copy_gm_to_l1<half, 64, 64>(du_hi_l1[0], workspace_3[((i * 16777216) + (cid * 4096))], 64, 0, 0, true);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<half, float, 64, 64, 64, true, false, 128>(a_hi_l1[0], du_hi_l1[0], y_frag[0], ascend_l0a, ascend_l0b, (bool)1, 64);
        AscendC::CrossCoreWaitFlag(3);
        tl::ascend::copy_gm_to_l1<half, 64, 64>(du_lo_l1[0], workspace_4[((i * 16777216) + (cid * 4096))], 64, 0, 0, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(2);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<half, float, 64, 64, 64, true, false, 128>(a_hi_l1[0], du_lo_l1[0], y_frag[0], ascend_l0a, ascend_l0b, (bool)0, 64);
        AscendC::CrossCoreWaitFlag(1);
        tl::ascend::copy_gm_to_l1<half, 64, 64>(a_lo_l1[0], workspace_2[((i * 16777216) + (cid * 4096))], 64, 0, 0, true);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
        AscendC::PipeBarrier<PIPE_M>();
        tl::ascend::gemm_v0<half, float, 64, 64, 64, true, false, 128>(a_lo_l1[0], du_hi_l1[0], y_frag[0], ascend_l0a, ascend_l0b, (bool)0, 64);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
        AscendC::PipeBarrier<PIPE_FIX>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_5[((i * 16777216) + (cid * 4096))], y_frag[0], 64, 0, 0);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(4);
      }
    }
  }
  if ASCEND_IS_AIV {
    tl::ascend::copy_gm_to_ub<float, 512>(beta_tile[0], BetaFlat[((cid % 512) * 512)], 262144, 1, 512, 0.000000e+00f);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
    tl::ascend::ArithProgression<int>(off_i32[0], ((cid / 512) * 4), 32, 64);
    AscendC::LocalTensor<uint32_t> off_u32 = off_i32.ReinterpretCast<uint32_t>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::SetMaskNorm();
    AscendC::Gather(bt[0], beta_tile[0], off_u32[0], 0, 64);
    tl::ascend::copy_gm_to_ub<float, 64>(acc[0], dbeta_flat[(cid * 64)], 64, 1, 64, 0.000000e+00f);
    for (int32_t i_v_outer_1 = 0; i_v_outer_1 < 1; ++i_v_outer_1) {
      for (int32_t i_1 = 0; i_1 < 2; ++i_1) {
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        tl::ascend::copy_gm_to_ub<float, 64, 64>(v_ub[(i_1 * 4096)], V[((((cid % 512) * 65536) + ((cid / 512) * 128)) + (i_1 * 64))], 1024, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 64>(a_f32[0], A[(((cid % 512) * 32768) + ((cid / 512) * 64))], 512, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
        tl::ascend::copy_ub_to_ub<half, float, 4096>(a_hi_ub[0], a_f32[0], 64, 64, 64, 64, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, half, 4096>(a_hi_f32[0], a_hi_ub[0], 64, 64, 64, 64, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(a_lo_f32[0], a_f32[0], a_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 4096>(a_lo_ub[0], a_lo_f32[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_1[((i_1 * 16777216) + (cid * 4096))], a_hi_ub[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_2[((i_1 * 16777216) + (cid * 4096))], a_lo_ub[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(1);
        tl::ascend::copy_gm_to_ub<float, 64, 64>(du_f32[0], du[((((cid % 512) * 65536) + ((cid / 512) * 128)) + (i_1 * 64))], 1024, 64, 64, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
        tl::ascend::copy_ub_to_ub<half, float, 4096>(du_hi_ub[0], du_f32[0], 64, 64, 64, 64, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, half, 4096>(du_hi_f32[0], du_hi_ub[0], 64, 64, 64, 64, 64, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float, false>(du_lo_f32[0], du_f32[0], du_hi_f32[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<half, float, 4096>(du_lo_ub[0], du_lo_f32[0], 64, 64, 64, 64, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_3[((i_1 * 16777216) + (cid * 4096))], du_hi_ub[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
        tl::ascend::copy_ub_to_gm<half, 64, 64>(workspace_4[((i_1 * 16777216) + (cid * 4096))], du_lo_ub[0], 64, 64, 64);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
      }
      for (int32_t i_2 = 0; i_2 < 2; ++i_2) {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        tl::ascend::Broadcast<float, 2, 1, false>(bt2d[0],bt[0],tmp_ub[0],(uint32_t[]){64, 64}, (uint32_t[]){64, 1});
        AscendC::CrossCoreWaitFlag(4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::PipeBarrier<PIPE_MTE2>();
        tl::ascend::copy_gm_to_ub<float, 64, 64>(y_ub[0], workspace_5[((i_2 * 16777216) + (cid * 4096))], 64, 64, 64, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
        AscendC::Mul<float, false>(dv_ub[0], y_ub[0], bt2d[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<float, 64, 64>(dv[((((cid % 512) * 65536) + ((cid / 512) * 128)) + (i_2 * 64))], dv_ub[0], 1024, 64, 64);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<float, false>(r_tmpv[0], y_ub[0], v_ub[(i_2 * 4096)], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::reduce_2d<float, reduce2d_v2::Reduce2DKind::kSum, true, 64, 64, -1, 64, false>(r_part[0], r_tmpv[0], tmp_ub.ReinterpretCast<float>()[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(acc[0], acc[0], r_part[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::BinaryRepeatParams());
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(7);
    tl::ascend::copy_ub_to_gm<float, 64>(dbeta_flat[(cid * 64)], acc[0], 64, 1, 64);
    if ASCEND_IS_AIV {
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
    if ASCEND_IS_AIV {
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
    tl::ascend::copy_gm_to_ub<float, 1, 64>(dbeta2[0], dbeta_flat_v[(cid * 64)], 1, 64, 1, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    tl::ascend::copy_ub_to_gm<float, 1, 64>(dbeta[(((cid % 512) * 512) + (cid / 512))], dbeta2[0], 8, 64, 1);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* V_handle, uint8_t* BetaFlat_handle, uint8_t* A_handle, uint8_t* du_handle, uint8_t* dbeta_flat_handle, uint8_t* dbeta_flat_v_handle, uint8_t* dv_handle, uint8_t* dbeta_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_4_handle, uint8_t* workspace_5_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<4096, nullptr, stream>>>(V_handle, BetaFlat_handle, A_handle, du_handle, dbeta_flat_handle, dbeta_flat_v_handle, dv_handle, dbeta_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_4_handle, workspace_5_handle, fftsAddr);
}
