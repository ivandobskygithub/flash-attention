#pragma once

#include "flash_bwd_kernel_sm90.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "sm120_traits.hpp"

namespace flash {

// Placeholder SM120 backward collective that reuses the SM90 implementation for
// syntax checks and templating. SM120 paths are TN-only and currently reuse the
// SM90 pipelines with fixed stage counts until specialized kernels are added.
template <bool SdP_swapAB, class TileShape_MNK, class ClusterShape,
          typename Element, typename ArchTag, bool Is_causal, bool Has_softcap,
          bool dKV_swapAB = false, bool dQ_swapAB = false, bool Is_local = false,
          bool Varlen = false, bool Deterministic = false,
          int NumMmaWarpGroups = 2, int AtomLayoutMSdP = 1, int AtomLayoutNdKV = 2,
          int AtomLayoutMdQ = 1, bool Mma_dP_is_RS = false>
using CollectiveMainloopBwdSm120 = CollectiveMainloopBwdSm90<2, 2, 2,
    ClusterShape, TileShape_MNK, Element, float, ArchTag, Is_causal, Is_local,
    Has_softcap, Varlen, Deterministic, SdP_swapAB, dKV_swapAB, dQ_swapAB,
    NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ,
    Mma_dP_is_RS>;

}  // namespace flash
