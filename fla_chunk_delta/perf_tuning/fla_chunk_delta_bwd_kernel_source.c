#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q3_handle,  GM_ADDR K3_handle,  GM_ADDR W3_handle,  GM_ADDR Do3_handle,  GM_ADDR Dv3_handle,  GM_ADDR GFlat_handle,  GM_ADDR Gk3_handle,  GM_ADDR Dht_handle,  GM_ADDR Dh_handle,  GM_ADDR Dh0_handle,  GM_ADDR Dv2_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle,  GM_ADDR workspace_3b_handle,  GM_ADDR workspace_4_handle,  GM_ADDR workspace_5_handle,  GM_ADDR workspace_6_handle, uint64_t fftsAddr) {
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
  AscendC::GlobalTensor<float> GFlat;
  GFlat.SetGlobalBuffer((__gm__ float*)GFlat_handle);
  AscendC::GlobalTensor<float> Gk3;
  Gk3.SetGlobalBuffer((__gm__ float*)Gk3_handle);
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
  auto k_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 0);
  auto dh16_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 16384);
  auto dvp_frag = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto do_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 32768);
  auto qt16_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 40960);
  auto qdo_frag = ascend_l0c.GetWithOffset<float>(8192, 16384);
  auto w_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 57344);
  auto dv2_16_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 73728);
  auto wdv2_frag = ascend_l0c.GetWithOffset<float>(8192, 49152);
  auto dh_ub = ascend_ub.GetWithOffset<float>(4096, 0);
  auto dh16_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 16384);
  auto dvp_ub = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto g_span = ascend_ub.GetWithOffset<float>(2048, 32768);
  auto glast_ub = ascend_ub.GetWithOffset<float>(1, 40960);
  auto gdiff_ub = ascend_ub.GetWithOffset<float>(32, 40992);
  auto gpre_ub = ascend_ub.GetWithOffset<float>(32, 41120);
  auto gate_ub = ascend_ub.GetWithOffset<float>(32, 41248);
  auto tmp_gate = ascend_ub.GetWithOffset<float>(256, 41376);
  auto dv16_ub = ascend_ub.GetWithOffset<bfloat16_t>(2048, 42400);
  auto dv_ub = ascend_ub.GetWithOffset<float>(2048, 46496);
  auto dv2_ub = ascend_ub.GetWithOffset<float>(2048, 54688);
  auto dv2_16_ub = ascend_ub.GetWithOffset<bfloat16_t>(2048, 62880);
  auto q_half_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 66976);
  auto q_ub_a = ascend_ub.GetWithOffset<float>(2048, 71072);
  auto q_half_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 79264);
  auto q_ub_b = ascend_ub.GetWithOffset<float>(2048, 83360);
  auto gexp_ub = ascend_ub.GetWithOffset<float>(32, 91552);
  auto tmp_qa = ascend_ub.GetWithOffset<float>(256, 91680);
  auto qt16_a = ascend_ub.GetWithOffset<bfloat16_t>(2048, 92704);
  auto tmp_qb = ascend_ub.GetWithOffset<float>(256, 96800);
  auto qt16_b = ascend_ub.GetWithOffset<bfloat16_t>(2048, 97824);
  auto gkl_ub = ascend_ub.GetWithOffset<float>(64, 101920);
  auto dsrc_ub = ascend_ub.GetWithOffset<float>(64, 102176);
  auto qdo_ub = ascend_ub.GetWithOffset<float>(4096, 102432);
  auto wdv2_ub = ascend_ub.GetWithOffset<float>(4096, 118816);
  auto tmp_decay = ascend_ub.GetWithOffset<float>(512, 135200);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t k_task = 0; k_task < 7; ++k_task) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((k_task * 5) + (cid / 4)) < 32) {
        for (int32_t it_f = 0; it_f < 128; ++it_f) {
          AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(7);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(7);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(k_l1[0], K3[((((((k_task * 10) + (cid / 2)) % 64) * 128) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
          AscendC::CrossCoreWaitFlag(0);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 64>(dh16_l1[0], workspace_1[(cid * 8192)], 64, 128, 64, true);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(1);
          AscendC::PipeBarrier<PIPE_M>();
          tl::ascend::gemm_v0<bfloat16_t, float, 64, 64, 128, false, false, 128>(k_l1[0], dh16_l1[0], dvp_frag[0], ascend_l0a, ascend_l0b, (bool)1, 64);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
          AscendC::PipeBarrier<PIPE_FIX>();
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_2[(cid * 4096)], dvp_frag[0], 64, 64, 64, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(do_l1[0], Do3[(((((((k_task * 10) + (cid / 2)) % 64) * 128) + ((cid % 2) * 64)) + 66584576) - (it_f * 524288))], 8192, 64, 64, true);
          AscendC::CrossCoreWaitFlag(2);
          AscendC::PipeBarrier<PIPE_MTE2>();
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1[0], workspace_3[(cid * 4096)], 64, 64, 64, true);
          AscendC::CrossCoreWaitFlag(3);
          AscendC::PipeBarrier<PIPE_MTE2>();
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(qt16_l1[4096], workspace_3b[(cid * 4096)], 64, 64, 64, false);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(3);
          tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(qt16_l1[0], do_l1[0], qdo_frag[0], ascend_l0a, ascend_l0b, (bool)1, 64);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(4);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_5[(cid * 8192)], qdo_frag[0], 64, 128, 64, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(5);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(w_l1[0], W3[((((((k_task * 10) + (cid / 2)) % 64) * 128) + 66584576) - (it_f * 524288))], 8192, 64, 128, true);
          AscendC::CrossCoreWaitFlag(4);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 64>(dv2_16_l1[0], workspace_4[(cid * 4096)], 64, 64, 64, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(5);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(5);
          tl::ascend::gemm_v0<bfloat16_t, float, 128, 64, 64, true, false, 128>(w_l1[0], dv2_16_l1[0], wdv2_frag[0], ascend_l0a, ascend_l0b, (bool)1, 64);
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(workspace_6[(cid * 8192)], wdv2_frag[0], 64, 128, 64, 0);
          AscendC::CrossCoreSetFlag<2, PIPE_FIX>(6);
        }
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t k_task_1 = 0; k_task_1 < 7; ++k_task_1) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((k_task_1 * 5) + (cid / 4)) < 32) {
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint8_t>((uint64_t)0, 4096);
        AscendC::Duplicate<float, false>(dh_ub[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        for (int32_t it_f_1 = 0; it_f_1 < 128; ++it_f_1) {
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 4096>(dh16_ub[0], dh_ub[0], 64, 64, 64, 64, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::PipeBarrier<PIPE_MTE3>();
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(Dh[((((((((k_task_1 * 10) + (cid / 2)) % 64) * 16384) + (vid * 8192)) + ((cid % 2) * 64)) + 133169152) - (it_f_1 * 1048576))], dh16_ub[0], 128, 64, 64);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(workspace_1[((cid * 8192) + (vid * 4096))], dh16_ub[0], 64, 64, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(0);
          AscendC::CrossCoreWaitFlag(1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
          tl::ascend::copy_gm_to_ub<float, 64, 32>(dvp_ub[0], workspace_2[((cid * 4096) + (vid * 2048))], 64, 32, 64, 0.000000e+00f);
          AscendC::PipeBarrier<PIPE_MTE2>();
          tl::ascend::copy_gm_to_ub<float, 2048>(g_span[0], GFlat[(((vid * 2048) + 520192) - (it_f_1 * 4096))], 524288, ((((k_task_1 * 10) + (cid / 2)) <= 63) ? 1 : 0), 2048, 0.000000e+00f);
          AscendC::PipeBarrier<PIPE_ALL>();
          glast_ub.SetValue(0, GFlat.GetValue((((((k_task_1 * 10) + (cid / 2)) % 64) + 524224) - (it_f_1 * 4096))));
          AscendC::SetMaskCount();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, 32);
          AscendC::Duplicate<float, false>(gdiff_ub[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
          for (int32_t i = 0; i < 32; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(6);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(7);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(7);
            AscendC::PipeBarrier<PIPE_ALL>();
            gdiff_ub.SetValue(i, ((g_span.GetValue(((i * 64) + (((k_task_1 * 10) + (cid / 2)) % 64))) - glast_ub.GetValue(0)) * -1.000000e+00f));
            AscendC::PipeBarrier<PIPE_ALL>();
            gpre_ub.SetValue(i, g_span.GetValue(((i * 64) + (((k_task_1 * 10) + (cid / 2)) % 64))));
          }
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          AscendC::SetMaskNorm();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)4294967295);
          AscendC::Exp<float, false>(gate_ub[0], gdiff_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_gate[0], gate_ub[0], 4, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(dvp_ub[0], dvp_ub[0], tmp_gate[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(dv16_ub[0], Dv3[(((((vid * 262144) + ((((k_task_1 * 10) + (cid / 2)) % 64) * 128)) + ((cid % 2) * 64)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(dv_ub[0], dv16_ub[0], 32, 64, 64, 32, 64, 64);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<float, false>(dv2_ub[0], dv_ub[0], dvp_ub[0], AscendC::MASK_PLACEHOLDER, 32, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(dv2_16_ub[0], dv2_ub[0], 32, 64, 64, 32, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(Dv2[(((((vid * 262144) + ((((k_task_1 * 10) + (cid / 2)) % 64) * 128)) + ((cid % 2) * 64)) + 66584576) - (it_f_1 * 524288))], dv2_16_ub[0], 8192, 32, 64);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_4[((cid * 4096) + (vid * 2048))], dv2_16_ub[0], 64, 32, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(4);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_a[0], Q3[((((vid * 262144) + ((((k_task_1 * 10) + (cid / 2)) % 64) * 128)) + 66584576) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_a[0], q_half_a[0], 32, 64, 64, 32, 64, 64);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64, 32>(q_half_b[0], Q3[((((vid * 262144) + ((((k_task_1 * 10) + (cid / 2)) % 64) * 128)) + 66584640) - (it_f_1 * 524288))], 8192, 32, 64, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
          tl::ascend::copy_ub_to_ub<float, bfloat16_t, 2048>(q_ub_b[0], q_half_b[0], 32, 64, 64, 32, 64, 64);
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, (uint64_t)4294967295);
          AscendC::Exp<float, false>(gexp_ub[0], gpre_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_qa[0], gexp_ub[0], 4, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(q_ub_a[0], q_ub_a[0], tmp_qa[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_a[0], q_ub_a[0], 32, 64, 64, 32, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3[((cid * 4096) + (vid * 2048))], qt16_a[0], 64, 32, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(2);
          tl::ascend::brcb<float>(tmp_qb[0], gexp_ub[0], 4, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(q_ub_b[0], q_ub_b[0], tmp_qb[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::copy_ub_to_ub<bfloat16_t, float, 2048>(qt16_b[0], q_ub_b[0], 32, 64, 64, 32, 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(workspace_3b[((cid * 4096) + (vid * 2048))], qt16_b[0], 64, 32, 64);
          AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(3);
          tl::ascend::copy_gm_to_ub<float, 64>(gkl_ub[0], Gk3[(((((((k_task_1 * 10) + (cid / 2)) % 64) * 128) + (vid * 64)) + 67100672) - (it_f_1 * 524288))], 8192, 1, 64, 0.000000e+00f);
          AscendC::SetMaskCount();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, 64);
          AscendC::Duplicate<float, false>(dsrc_ub[0], 0.000000e+00f, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
          for (int32_t k = 0; k < 64; ++k) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(7);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            AscendC::PipeBarrier<PIPE_ALL>();
            dsrc_ub.SetValue(k, (gkl_ub.GetValue(k) + glast_ub.GetValue(0)));
          }
          AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
          AscendC::SetMaskNorm();
          AscendC::SetVectorMask<uint8_t>((uint64_t)0, 0xffffffffffffffffULL);
          AscendC::Exp<float, false>(dsrc_ub[0], dsrc_ub[0], AscendC::MASK_PLACEHOLDER, 1, AscendC::UnaryRepeatParams());
          AscendC::CrossCoreWaitFlag(5);
          tl::ascend::copy_gm_to_ub<float, 64, 64>(qdo_ub[0], workspace_5[((cid * 8192) + (vid * 4096))], 64, 64, 64, 0.000000e+00f);
          AscendC::CrossCoreWaitFlag(6);
          tl::ascend::copy_gm_to_ub<float, 64, 64>(wdv2_ub[0], workspace_6[((cid * 8192) + (vid * 4096))], 64, 64, 64, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          {
          AscendC::Muls<float, false>(qdo_ub[0], qdo_ub[0], 8.838835e-02f, AscendC::MASK_PLACEHOLDER, 64, AscendC::UnaryRepeatParams());
          }
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(tmp_decay[0], dsrc_ub[0], 8, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(dh_ub[0], dh_ub[0], tmp_decay[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<float, false>(dh_ub[0], dh_ub[0], qdo_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub<float, false>(dh_ub[0], dh_ub[0], wdv2_ub[0], AscendC::MASK_PLACEHOLDER, 64, AscendC::BinaryRepeatParams());
        }
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q3_handle, uint8_t* K3_handle, uint8_t* W3_handle, uint8_t* Do3_handle, uint8_t* Dv3_handle, uint8_t* GFlat_handle, uint8_t* Gk3_handle, uint8_t* Dht_handle, uint8_t* Dh_handle, uint8_t* Dh0_handle, uint8_t* Dv2_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, uint8_t* workspace_3b_handle, uint8_t* workspace_4_handle, uint8_t* workspace_5_handle, uint8_t* workspace_6_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<20, nullptr, stream>>>(Q3_handle, K3_handle, W3_handle, Do3_handle, Dv3_handle, GFlat_handle, Gk3_handle, Dht_handle, Dh_handle, Dh0_handle, Dv2_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, workspace_3b_handle, workspace_4_handle, workspace_5_handle, workspace_6_handle, fftsAddr);
}
