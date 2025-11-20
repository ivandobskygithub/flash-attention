#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"

namespace flash::sm120 {

#ifndef CUTLASS_ARCH_SM120_ENABLED
namespace cutlass { namespace arch {
struct Sm120 {};
}}  // namespace cutlass::arch
#endif

struct Sm120Traits {
    static constexpr int kMinCudaArch = 1200;
    static constexpr bool kTNOnly = true;
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 128;
    static constexpr int kClusterM = 1;
    static constexpr int kClusterN = 1;
    static constexpr int kClusterK = 1;
    static constexpr bool kUseTma = true;
    static constexpr bool kUseMbarrier = true;
    static constexpr bool kUseWarpSpecialized = true;
    using ArchTag = cutlass::arch::Sm120;
};

static_assert(Sm120Traits::kTileM == 128 && Sm120Traits::kTileN == 128 && Sm120Traits::kTileK == 128,
              "SM120 FlashAttention uses 128x128x128 tiles by default");
static_assert(Sm120Traits::kTNOnly, "SM120 FlashAttention uses TN layout only");

}  // namespace flash::sm120
