# SC-18237: Qwen production-deferred calibration evidence

Captured and validated on 2026-08-09. This record resolves the unused q8 bounded-attention
measurement identified by SC-18101 and audits every shipped MLX calibration binding for a load shape
that an actual production branch can produce.

## Capture provenance

The release memory adapter loaded the real cached q8 artifact at
`SceneWorks/qwen-image-mlx@8080a4171f1c8b7fca6c30491eafbe6ffab754bf:q8` (36 GiB on disk) through
the production-deferred `LoadSpec`. Both repositories were clean at capture time:

| Repository | Exact revision |
|---|---|
| SceneWorks | `4a649d9c46e8b56b232422cb92550028108b44c9` |
| inference | `40fa7583a01974617e2a7275052d6d446688c956` |
| inference provider closure | `9930aa538259f7c576c13e3241e872a6486b049fe62b423e5dca3b3fe56f7bae` |
| SceneWorks matrix source | `source-tree:d7f313b635435309b547206faa6526e272735d41b47e6c903ec928cb29bd24fa` |

Capture host: Apple M5 Max (`Mac17,6`), 128 GiB unified memory, macOS 26.5.2, MLX memory
limit 130,567,005,798 bytes, derived wired/process ceiling 87,044,670,532 bytes. The two records
completed at 14:05:51Z and 14:08:53Z. The full q8-only cache was used because this host has no bf16
or q4 sibling artifact; the whole-snapshot verifier therefore cannot claim those absent tiers.

## Promoted records

| Rung | Record | Load shape | Predicted peak | Observed allocator/wired peak | Active peak | Reclaimable peak |
|---|---|---|---:|---:|---:|---:|
| Bounded attention | `imc-56c1f11bd03822d9c241` | deferred | 46,305,116,160 | 44,056,333,980 | 40,203,970,608 | 3,852,363,372 |
| Bounded transformer residency | `imc-a21d2ea9e2d95cf48e82` | deferred | 16,777,216,000 | 15,977,855,160 | 15,977,625,784 | 229,376 |

Both records passed loadability, the exact-fit/unknown-budget/stale/warm/cancel/error scenarios, and
the parity gate. Maximum error was 0.117647 against a 0.188235 threshold; mean error was 0.00050713
against a 0.00196078 threshold. Their negative mutations failed as expected.

## Production reachability and 48 GiB boundary

The ignored c0 probe resolves the real cached artifact, constructs the same production `LoadSpec`,
and drives the request selector. These are the observed results:

| Host cap | Policy | Production shape | Result |
|---:|---|---|---|
| 96 GiB | Resident | deferred | admitted `BoundedAttention`, record `imc-56c1f11bd03822d9c241` |
| 48 GiB | Sequential | deferred | admitted `BoundedTransformerResidency`, record `imc-a21d2ea9e2d95cf48e82` |
| 48 GiB | Resident | deferred | safely refused; smallest verified boundary is 60.73 GiB |

An MLX evidence record includes workload peak plus non-MLX foreign reserve. Copying the absolute
46.93 GiB reserve from a 128 GiB capture host onto a 48 GiB target made the target impossible by
construction. Admission now scales only the foreign reserve by live host capacity and keeps the
measured MLX peak and process ceiling absolute:

```text
foreign reserve at 48 GiB
  = ceil(50,394,282,940 * 51,539,607,552 / 137,438,953,472)
  = 18,897,856,103 bytes

q8 bounded-transformer boundary
  = 16,777,216,000 + 18,897,856,103
  = 35,675,072,103 bytes (33.225 GiB)
```

Stale evidence still widens the measured workload peak, and `apply_request_gpu_memory_limit` still
sets the absolute MLX soft limit before the fatal allocation boundary. This is a host-normalization
fix, not a blanket Legacy fallback and not a relaxation of process-termination safety.

## Shipped-binding audit

The manifest now ships only the q8 pair that was captured under the executable production-deferred
shape. Historical eager Qwen bf16/q4 records remain in the evidence corpus for provenance but do not
authorize the production route; their calibration-plan cases now request deferred recapture. The
production selector continues to serve those unbound tiers through its conservative estimate ladder.

`every_shipped_audited_mlx_binding_has_a_producible_production_load_shape` couples manifest bindings
to the real production shaping functions. It covers:

| Provider/tier | Shipped production shapes |
|---|---|
| Qwen q8 | deferred for bounded attention and bounded transformer residency |
| Z-Image Turbo q4 | eager for Resident-family rungs; deferred for Sequential bounded transformer residency |
| Krea 2 Turbo Control q4 | eager for bounded decode |

The companion Qwen manifest/evidence test flips every declared q8 load shape and requires calibrated
admission to disappear. The adapter test independently rejects a plan/provider shape mismatch. These
checks fail on a shape mutation rather than merely asserting that some evidence route remains green.
