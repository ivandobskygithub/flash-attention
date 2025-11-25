/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

constexpr int smem_estimate_bytes(int block_m, int block_n, int headdim, int headdim_v, int element_size) {
    // Double-buffer the residency for Q/K/V and the accumulators when the head/value dims are modest.
    // Value dimensions 256+ already drive the shared-memory footprint high, so treat them like the large
    // combined-head/value case and drop to a single buffer in that regime to avoid over-clamping.
    int const buffering = (headdim_v >= 256 || headdim + headdim_v >= 512) ? 1 : 2;
    return buffering * (block_m + block_n) * (headdim + headdim_v) * element_size;
}

constexpr int clamp_block_n_for_smem(int block_m, int block_n, int headdim, int headdim_v,
        int element_size, int smem_limit) {
    int const smem_usage = smem_estimate_bytes(block_m, block_n, headdim, headdim_v, element_size);
    if (smem_usage <= smem_limit) {
        return block_n;
    }
    // Keep the tile width aligned to 16 to satisfy GMMA tile constraints while allowing tight caps.
    int const denom = 2 * element_size * (headdim + headdim_v);
    int max_block_n = denom > 0 ? smem_limit / denom - block_m : block_n;
    if (max_block_n < 16) { max_block_n = 16; }
    max_block_n = (max_block_n / 16) * 16;
    return max_block_n > 0 ? max_block_n : 16;
}

constexpr std::tuple<int, int> enforce_smem_limit(int block_m, int block_n, int headdim, int headdim_v,
        int element_size, int smem_limit) {
    int adjusted_block_n = clamp_block_n_for_smem(block_m, block_n, headdim, headdim_v, element_size, smem_limit);
    int smem_usage = smem_estimate_bytes(block_m, adjusted_block_n, headdim, headdim_v, element_size);
    if (smem_usage > smem_limit && block_m > 64) {
        block_m = 64;
        adjusted_block_n = clamp_block_n_for_smem(block_m, adjusted_block_n, headdim, headdim_v, element_size, smem_limit);
        smem_usage = smem_estimate_bytes(block_m, adjusted_block_n, headdim, headdim_v, element_size);
    }
    if (smem_usage > smem_limit) {
        int const denom = 2 * element_size * (headdim + headdim_v);
        int max_block_m = denom > 0 ? smem_limit / denom - adjusted_block_n : block_m;
        if (max_block_m < 16) { max_block_m = 16; }
        max_block_m = (max_block_m / 16) * 16;
        block_m = max_block_m > 0 ? max_block_m : 16;
        adjusted_block_n = clamp_block_n_for_smem(block_m, adjusted_block_n, headdim, headdim_v, element_size, smem_limit);
    }
    return {block_m, adjusted_block_n};
}

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false) {
    constexpr int kSm120ConsumerSmemLimit = 101376;
    if (element_size == 2) {
        if (headdim <= 64) {
            // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
            // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
            // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
            if (headdim_v == 512) {
                // Keep the tile narrow to avoid blowing past the consumer shared-memory budget when values are very wide.
                auto const [block_m, block_n] = enforce_smem_limit(64, 64, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
                return {block_m, block_n, false, false};
            } else if (headdim_v == 256) {
                auto const [block_m, block_n] = enforce_smem_limit(64, 80, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
                return {block_m, block_n, true, true};
            } else {
                // Switch to tile size 192 x 192 for now
                bool const use_blockN_128 = is_causal || is_local || paged_kv_non_TMA;
                auto const [block_m, block_n] = enforce_smem_limit(192, use_blockN_128 ? 128 : 192, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
                return {block_m, block_n, use_blockN_128, true};
            }
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            // Large value dimensions inflate smem usage even at modest head sizes, so bias toward smaller tiles for dv >= 256.
            int const block_n = headdim_v >= 256 ? 96 : (is_local || paged_kv_non_TMA ? 128 : 144);
            auto const [block_m, block_n_capped] = enforce_smem_limit(block_n == 96 ? 128 : 192, block_n, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
            return {block_m, block_n_capped, false, true};
        } else if (headdim <= 128) {
            // Shared memory on consumer parts tops out at ~100KB, so prefer a BlockM=64 path that stays under that limit while
            // keeping BlockN as large as possible for throughput.
            int const block_n = paged_kv_non_TMA || is_local ? 80 : (headdim_v <= 128 ? 96 : 80);
            auto const [block_m, block_n_capped] = enforce_smem_limit(64, block_n, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
            return {block_m, block_n_capped, true, true};
            // {128, 192, true, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
        } else if (headdim <= 192) {
            // The 128x128 / 128x112 tiles exceed the ~100KB shared memory limit of consumer GPUs (for example, when running on
            // devices without the larger H100 shared memory carve‑out). Use smaller tiles for all value dims to guarantee we
            // stay below the per-block cap across head dimensions up to 192.
            int const block_n = paged_kv_non_TMA || is_local ? 64 : (headdim <= 160 ? 80 : 64);
            auto const [block_m, block_n_capped] = enforce_smem_limit(64, block_n, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
            return {block_m, block_n_capped, true, true};
        } else {
            // For head dims above 192 the shared-memory footprint grows quickly with BlockM, so stick to 64xN tiles even though
            // they are smaller than the H100-optimized 128xN shapes. Favor narrower BlockN when value dims are large to stay
            // under the ~100KB cap on consumer GPUs.
            int const block_n = paged_kv_non_TMA || is_local ? 48 : (headdim <= 256 ? 64 : 48);
            auto const [block_m, block_n_capped] = enforce_smem_limit(64, block_n, headdim, headdim_v, element_size, kSm120ConsumerSmemLimit);
            return {block_m, block_n_capped, true, true};
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224), true, true};
        } else if (headdim <= 192) {
            return {128, (paged_kv_non_TMA || softcap) && is_local ? 128 : 160, true, true};
        } else {
            return {128, is_local ? 64 : 128, true, !paged_kv_non_TMA};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}
