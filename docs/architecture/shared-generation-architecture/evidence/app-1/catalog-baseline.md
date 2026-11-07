# SW-7028 · Before-side public catalog and UI behavior

## Source and boundary

Captured before registry, gating or studio changes, from SceneWorks
`5a28d413dd9f579cc9607d44f252a045d025bd8e`, paired with inference
`2b09938e33370915d4f853389961709b8bbf8bf9` (the locked Cargo dependency).
The checked-in fixture includes the full SHA, dependency source and Git tree/blob identities.
CodeGraph discovery was used first; its index reported an older commit, so exact behavior was
read from this source tree. Confidence: high for this weights-free product boundary.

The host fixture deliberately installs every builtin artifact, declared tier and conditional
package, with a fixed project, image, clip, audio and training dataset. It does not report this
Mac's installation state. Native MLX/Candle availability, preview, decoder and training metadata
come from production core functions. The additional API projections are source-identified:

- `apps/rust-api/src/models/catalog.rs`: `project_moss_edit_surface` adds editing through
  `moss_image::edit_backend_eligible`; `apply_last_only_video_capability` reads the H3 keyframe
  predicate. Both appear in the native dump, including MOSS edit rows.
- `apps/rust-api/src/models/catalog.rs`: the minimum quality tier is projected from `mlx`;
  installed variants/conditional packages are supplied by the explicit all-installed host.
- `apps/rust-api/src/training.rs:206`: `apply_candle_training_defaults` changes checkpointing,
  preview frequency and Mage full-finetune precision. The test-only platform adapter preserves
  those defaults. It does not change production training code.

This captures the public builtin population and controlled runtime observations. Arbitrary
user installations, filesystem freshness, hardware inference, visual output quality and native
numerical/performance acceptance belong to their own contracts. The registry identity and
cross-host import fixtures are owned by SW-7030; this item preserves workflow/recipe and asset
import behavior before their consumers change.

## Fixtures and comparisons

All fixtures are under `apps/web/src/testUtils/catalogBaseline/`:

| File | Contents |
| --- | --- |
| `old-catalog.json` | Complete 143-model typed input catalog, 26 training targets/presets, macOS/Windows/Linux capabilities and source identity. |
| `old-observations.json` | Frozen old population and real studio observations. Controls, calls and state are interned in pools to keep repeated platform rows readable and bounded. |
| `old-behavior.json` | Appearance menu navigation/cardinality, authored/effective request batches, actual recipe launch/re-import and regeneration, file import/source identity, valid/refused Iris guidance intervals, and invalid sampling overrides for every image/video model. |
| `harness.jsx` | Real React screens, native Simple picker sheets, fixed host/project inputs, request recorders and control round trips. |
| `candidate.js` | Today's manifest adapter; `CATALOG_CANDIDATE` accepts a native typed projection. APP-1 replaces this adapter when ModelSpec becomes the producer. The old fixture and population remain frozen. |

The population includes full Image, Video, Audio, Vector, Document, Character and Training
studios (including Character Test and composed Vector Create-from-Prompt), and Simple
Image/Video/Audio, on all three platforms. Every advertised studio mode is
kept even when the backend gates it. Non-studio utility models remain in the catalog identity
check. Controls record role, accessible label, option values, selected/default value,
min/max/step, availability, disable/tooltip text and persistence key. Native scalar/select
controls, Simple sheets and chip groups are changed and restored; observations include the
actual emitted payload or refusal, persisted changes/side effects and hints. Identity generation is seeded per mounted workflow, retaining unique attachment ids and their
relationships rather than omitting them from snapshots. A refused or
incomplete form stays visible as an observation, rather than being counted as a submitted job.
Action buttons that fetch or execute a model are not clicked indiscriminately.

Appearance witnesses select single Camera/Film/Lighting choices, multiple Looks, Lighting
subgroups, back navigation and search. They submit both native packed and one-job-per-prompt
batches, retain authored text separately from effective directives, import the public recipe
carrier through the studio launch hook, and regenerate without composing directives twice.
File uploads use the actual source picker and upload callback. Invalid provider overrides are
observed as today's reset, omission or refusal, without inventing a stronger legacy contract.

`unionPopulation` joins frozen old rows with candidate rows. Old rows are loaded from the
checked-in observations, never rebuilt using candidate eligibility helpers. Removing
`starvector_1b` from an actual candidate dump fails with its model id; the old rows remain in the
union. Adding/changing a candidate row requires explicit acceptance; `vitest -u` cannot replace
the before side. Capture writes require the full pinned old SHA and clean old production paths.

## Reproduce

From the repository root, with web dependencies installed:

```sh
cargo run --locked -q -p sceneworks-core --bin dump-ui-catalog > /tmp/ui-catalog-candidate.json
CATALOG_CANDIDATE=/tmp/ui-catalog-candidate.json npm --prefix apps/web test -- src/catalogBaseline.test.jsx src/catalogBaseline.behavior.test.jsx
cargo test --locked -p sceneworks-core --bin dump-ui-catalog
node --test scripts/rearchitecture/capture-catalog-baseline.test.mjs
```

The native test verifies the dump against the frozen source inputs and compares JSON numeric
meaning across Rust/browser serialization (`1.0` and `1`). The script regression verifies that
a candidate checkout cannot overwrite an old-source fixture. The web tests also contain the
StarVector removal negative control.

Explicit capture is only for the old checkout. If reproducing later, use an isolated checkout
at the pinned SHA and overlay this test/dump tooling without changing production sources:

```sh
node scripts/rearchitecture/capture-catalog-baseline.mjs --source 5a28d413dd9f579cc9607d44f252a045d025bd8e --output apps/web/src/testUtils/catalogBaseline/old-catalog.json
CAPTURE_OLD_CATALOG=5a28d413dd9f579cc9607d44f252a045d025bd8e npm --prefix apps/web test -- src/catalogBaseline.test.jsx src/catalogBaseline.behavior.test.jsx
```

The capture helper replaces the earlier signature/source-hash/regex script. It consumes a
native typed dump and Cargo metadata; it neither regex-parses worker source nor introduces a
source-digest gate. Existing capability/memory verification teardown remains unchanged.

## Validation

The frozen population contains 1,197 public studio/platform/model/mode rows. The capture records
12,552 scalar/select/chip changes and 246 explicit behavior witnesses; these are observations,
not model renders. Source input count is 143 models and 26 training targets.

Validation commands and results are recorded with the local acceptance commit in the handoff.
All checks are CPU-only, weights-free, and network/model calls are blocked or captured at the
product submission boundary. No workflow, inference pin or production behavior changes are
part of this item. Normal web/native test discovery picks up the regression checks; no learned
CPU/GPU execution is added to CI.
