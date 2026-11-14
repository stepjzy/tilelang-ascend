// Standalone AscendC scatter kernel — no TileLang.
// Proves/disproves: hardware supports UB→GM strided scatter (DataCopyPad + dstStride).
//
// Compile (on NPU env):
//   bisheng --npu-arch=dav-2201 -std=c++17 -xasc [includes] -fPIC --shared scatter_raw.cpp -O2 -o scatter_raw.so

#include "kernel_operator.h"

using namespace AscendC;

constexpr int NUM_HEADS = 8;
constexpr int ELEMS     = 64;
constexpr int BURST_BYTES = sizeof(float);         // 4
constexpr int DST_PITCH  = NUM_HEADS * sizeof(float); // 32
constexpr int DST_GAP    = DST_PITCH - BURST_BYTES;   // 28

class ScatterKernel {
public:
    __aicore__ inline ScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        cid_ = GetBlockIdx();
        bh_  = cid_;
        gm_in_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input) + cid_ * ELEMS);
        gm_out_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output) + bh_);
        ub_buf_.Init(ELEMS * sizeof(float));
    }

    __aicore__ inline void Load() {
        LocalTensor<float> ub = ub_buf_.Get<float>();
        // contiguous GM→UB: 1 block, 256 bytes
        DataCopyExtParams loadParams(1, ELEMS * sizeof(float), 0, 0, 0);
        DataCopyPad(ub, gm_in_, loadParams);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> ub = ub_buf_.Get<float>();
        for (int i = 0; i < ELEMS; i++) {
            float v = ub.GetValue(i);
            ub.SetValue(i, v * 2.0f);
        }
    }

    __aicore__ inline void Scatter() {
        LocalTensor<float> ub = ub_buf_.Get<float>();
        // strided UB→GM: 64 bursts of 4 bytes, 28-byte gap between GM dests
        // This is the SAME DataCopyPad call that TileLang's copy_ub_to_gm generates
        DataCopyExtParams scatterParams(
            ELEMS,          // blockCount = 64
            BURST_BYTES,    // blockLen = 4 (one fp32 per burst)
            0,              // srcStride = 0 (UB contiguous)
            DST_GAP         // dstStride = 28 (32B pitch - 4B burst)
        );
        DataCopyPad(gm_out_, ub, scatterParams);
    }

    __aicore__ inline void Process() {
        Load();
        Compute();
        Scatter();
    }

private:
    int cid_ = 0;
    int bh_  = 0;
    GlobalTensor<float> gm_in_;
    GlobalTensor<float> gm_out_;
    TBuf<TPosition::VECCALC> ub_buf_;
};

extern "C" __global__ __aicore__ void scatter_raw_kernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling
) {
    ScatterKernel op;
    op.Init(input, output);
    op.Process();
}
