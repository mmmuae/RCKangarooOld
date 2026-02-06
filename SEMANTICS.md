# RCKangaroo Semantics Overview

This document summarizes the end-to-end control flow for the GPU-based kangaroo solver, with emphasis on where collisions are detected and how early-hit estimation works.

## Global state and setup
- Host-side globals keep the search configuration (`gStart`, `gEnd`, `gRangeWidth`, `gDP`, GPU mask) and runtime statistics such as DP buffers, per-herd counters, and gap-tracking structures (`gTameDistances`, `gWild1Distances`, `gWild2Distances`). These sets store distances for tame and wild walks so the host can track the closest cross-herd gap seen so far. 【F:RCKangaroo.cpp†L40-L101】
- Helpers convert serialized DP distances to signed `EcInt` values, compute unsigned gaps, and normalize candidate keys into the configured range to avoid overflow or out-of-interval estimates when reporting candidates. 【F:RCKangaroo.cpp†L116-L174】

## Gap-based collision estimation
- `EstimateKeyFromPair` evaluates a tame–wild or wild–wild distance pair, computing candidate private keys by adding or subtracting the half-range offset and checking whether the derived scalar reproduces the target public key. It also tracks the candidate closest to the start of the range to provide an early guess even before an exact collision is proven. 【F:RCKangaroo.cpp†L233-L338】
- `UpdateGlobalGap` and `ConsiderGapWithSet` maintain the best (smallest) distance gap observed across herds. Every time a new distance is inserted, the code compares it to neighbors in the opposite herd’s ordered set to refine the global best gap and the estimated key. This is the main host-side mechanism for anticipating a collision before it occurs. 【F:RCKangaroo.cpp†L340-L371】

## GPU initialization and worker threads
- `InitGpus` enumerates CUDA devices, filters unsupported cards, applies the mask provided via `-gpu`, and instantiates an `RCGpuKang` object per usable GPU with device metadata (index, SM count, L2 size). Each worker runs `RCGpuKang::Execute` in its own thread to drive jump computations and DP writes. 【F:RCKangaroo.cpp†L382-L451】
- `AddPointsToList` batches DP hits reported by GPUs into a shared buffer with basic overflow protection. `CheckNewPoints` later swaps this buffer for processing to avoid lock contention while GPUs continue generating points. 【F:RCKangaroo.cpp†L452-L520】

## Collision verification
- `Collision_SOTA` interprets a tame/wild match (or wild/wild pair) and derives the private key candidate by adding or subtracting offsets around the half-range. Both positive and negated variants are checked against the supplied meeting point to confirm a valid discrete log before declaring success. 【F:RCKangaroo.cpp†L467-L504】

## End-to-end flow
1. Host parses CLI options to set the range, DP bits, GPU mask, and optional tames/operation limits, then calls `InitGpus` to launch GPU workers.
2. GPUs perform kangaroo jumps and emit distinguished points; `AddPointsToList` collects them and `CheckNewPoints` deserializes each DP into a `DistanceEntry` tagged as tame, wild1, or wild2.
3. For each new DP, the host checks for an exact collision via the DP hash table; when found, `Collision_SOTA` computes and validates the key.
4. Even without a collision, `ConsiderGapWithSet` compares the new distance against nearby entries in the opposite herd to update `gLowestGap` and `gEstimatedKey`, which can expose imminent collisions and report the best candidate key seen so far.
5. Once a collision is confirmed or the ops/limit criteria are met, threads stop and the solution (or best estimate) is reported.
