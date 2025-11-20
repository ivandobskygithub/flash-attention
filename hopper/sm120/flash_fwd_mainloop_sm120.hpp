#pragma once

#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "sm120_traits.hpp"

namespace flash {

// Placeholder SM120 forward collective that reuses the Sm90 implementation while
// keeping SM120 specific type aliases. This maintains buildability for syntax
// checks on platforms without SM120 hardware.
template <int Stages, class ClusterShape, class TileShape_MNK, int kHeadDimV,
          typename Element, typename ElementAccum, typename ArchTag, bool IsCausal,
          bool IsLocal, bool HasSoftcap, bool Varlen, bool PagedKVNonTMA,
          bool AppendKV, bool HasQv, bool MmaPVIsRS, bool IntraWGOverlap,
          bool PackGQA, bool Split, bool VColMajor>
using CollectiveMainloopFwdSm120 = CollectiveMainloopFwdSm90<Stages, ClusterShape,
    TileShape_MNK, kHeadDimV, Element, ElementAccum, ArchTag, IsCausal, IsLocal,
    HasSoftcap, Varlen, PagedKVNonTMA, AppendKV, HasQv, MmaPVIsRS, IntraWGOverlap,
    PackGQA, Split, VColMajor>;

}  // namespace flash
