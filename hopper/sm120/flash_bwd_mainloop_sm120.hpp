#pragma once

#include "flash_bwd_kernel_sm90.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "sm120_traits.hpp"

namespace flash {

// Placeholder SM120 backward collective that reuses the SM90 implementation for
// syntax checks and templating.
template <bool Is_TF32, class TileShape, class ClusterShape, typename Element,
          typename ArchTag, bool IsCausal, bool HasSoftcap>
using CollectiveMainloopBwdSm120 = CollectiveMainloopBwdSm90<Is_TF32, TileShape,
    ClusterShape, Element, ArchTag, IsCausal, HasSoftcap>;

}  // namespace flash
