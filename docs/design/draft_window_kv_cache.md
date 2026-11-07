# Draft-Window KV Cache

Some target-conditioned draft models use only a bounded trailing context even
when the target model attends to the full prefix. Giving both models one KV
block pool ties their lifetimes together: the target needs stable full-prefix
blocks, while the drafter can recycle blocks outside its window.

vLLM automatically separates these lifetimes for compatible DFlash and DSpark
models. There is no user-facing switch. In unsupported configurations, vLLM
uses the regular shared KV-cache layout.

## Compatibility

The independent draft-window pool is enabled when all of the following hold:

- the DFlash or DSpark speculator consumes target hidden states;
- every draft KV group uses sliding-window attention with the same block size
  as the target groups;
- Model Runner V2 and the hybrid KV-cache manager are enabled;
- pipeline, prefill-context, and decode-context parallel sizes are all one;
- HiSparse and model-defined custom KV grouping are not enabled.
- KV transfer is disabled or uses only Mooncake and Mooncake Store connectors.

Tensor and data parallel deployments are supported. The extensible KV cache is
disabled for DFlash and DSpark because its current resize operation assumes one
capacity shared by every KV group. Other transfer connectors retain the shared
KV-cache layout until their sparse recovery and allocator-local block IDs have
been validated.

## Storage

Target and draft groups have independent block allocators and backing tensors.
The target pool is sized from the memory left after reserving the draft pool.
The draft pool has a fixed upper bound derived from:

- each draft group's sliding-window size and retained tokens;
- speculative lookahead and in-flight tokens;
- asynchronous connector loads configured on KV consumers;
- the configured maximum number of sequences; and
- block-boundary and null-block space.

The bound is capped by the storage needed for full-length requests. It does not
add an arbitrary retention margin.

Block identifiers are local to their owning pool. Scheduler metadata therefore
keeps the pool identity for copy-on-write and block-zeroing operations.

## Prefix recovery

A target prefix hit can extend beyond the pages still resident in a draft
window. vLLM preserves the target hit and plans recovery independently for each
draft group:

1. Compute the historical token range needed by each draft window at the next
   draft boundary.
2. Reuse any cached draft pages in that range.
3. Find the first missing page across the draft groups.
4. Replay one continuous suffix from that page to the target hit boundary.

Continuous replay is intentional. It restores a valid attention history while
avoiding the target model's full common-endpoint replay. During replay, target
slot mappings below the preserved hit boundary are masked, so only draft KV is
rewritten.

This protection is based on the known target hit boundary. It does not maintain
an endpoint cache or add endpoint-specific eviction state.

## Prefill/decode transfer

For Mooncake disaggregated serving, the consumer sends the exact logical page
positions missing from every group. The producer maps those positions to the
pages it still retains instead of assuming that every request is a dense
suffix. If the producer no longer retains a requested page, the transfer fails
and the consumer falls back to local replay.

Positioned requests include a compatibility guard. A producer that predates
positioned transfer rejects the group shape instead of silently transferring
the wrong suffix.

## Correctness fallbacks

The optimization is not used when its storage or execution assumptions cannot
be proved from the resolved configuration. Transfer geometry mismatches,
missing producer pages, and failed remote loads use existing connector failure
handling and recover locally.
