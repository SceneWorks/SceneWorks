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
npm --prefix apps/web test -- src/catalogBaseline.test.jsx src/catalogBaseline.behavior.test.jsx src/catalogBaseline.fixture.test.jsx
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
12,555 scalar/select/chip changes and 246 explicit behavior witnesses; these are observations,
not model renders. Source input count is 143 models and 26 training targets.

Validation commands and results are recorded with the local acceptance commit in the handoff.
All checks are CPU-only, weights-free, and network/model calls are blocked or captured at the
product submission boundary. No workflow, inference pin or production behavior changes are
part of this item. Normal web/native test discovery picks up the regression checks; no learned
CPU/GPU execution is added to CI.

## Ordinary CI adapter repair

[Web job 128447191685](https://github.com/SceneWorks/SceneWorks/actions/runs/42387223932/job/128447191685)
at published head `24b248c8786e0f401017d537fb207b5ac699ed38` failed exactly six Vector
Create-from-Prompt rows. The earlier local check supplied `CATALOG_CANDIDATE`; ordinary CI
runs with it unset. The capture dumper supplies controlled model inputs in id order, but the
manifest bridge retained manifest source order. That changed the first and alternate raster
choices from Anima to MOSS. The same six failures reproduced locally without the override,
so this was an adapter-path mismatch, not an OS-specific inference issue. Earlier
`chunk unavailable` diagnostics were expected fault-injection output in passing tests.

The manifest bridge now uses the capture's code-point id order. Explicit native dumps keep
their producer order: a regression test reverses that input and verifies that the actual
submitted raster choice changes. Another test compares actual ordinary-path Vector controls
and requests with the frozen native inputs. Non-select controls omit the absent `options`
property rather than retaining `undefined`; select values, option order and disabled state
remain checked, including a strict JSON round-trip regression.

The three frozen JSON artifacts, 1,197-row population and StarVector removal negative control
are unchanged. No snapshots were regenerated. The previous native-input receipts remain
valid for that input lane; any claim that they established ordinary default-path parity at
`24b248c878` is withdrawn. The repair separately validates the ordinary Node 22.23.4 web lane
(the CI version) and the native-input lane. Production code, workflow configuration, engine
pin and learned-execution policy are unchanged.

Repair validation (Node 22.23.4): ordinary full web discovery passed **7,320 tests in 390 files**
with four bounded workers; the supplied native-dump lane passed **1,449 tests in three files**.
Full web ESLint, the native frozen-input test, clippy, rustfmt and the capture-protection script
test passed. An actual candidate with StarVector removed still failed as intended, naming
`starvector_1b`. The local proof resolves the fixture failure; the failed remote receipt is
retained and publication requires the new-head CI run.

### Workflow-share parity follow-up

[Parity Rust job 128447191618](https://github.com/SceneWorks/SceneWorks/actions/runs/42387223932/job/128447191618)
also reported the discovery scan at `harness.jsx:92`. `targetForPlatform` clones catalog target
and preset defaults into mocked AppContext inputs; only test files import the helper and its
submission callbacks are recorders. It is not a shipping workflow payload builder. The policy
in `workflow_share.rs:730-748` explicitly directs renaming non-payload locals because the scan
recognizes property writes by the identifier `advanced`.

The alias is now `trainingDefaults`. The `next.defaults.advanced` property and every field/value
remain unchanged. No builder registry entry, scanner exemption or gate change was added.
The exact focused command below reproduced the failure before the rename and passed afterward;
the complete `workflow_share` integration suite also passed all **92 tests**. The affected
Node 22.23.4 baseline suites passed **1,449 tests**, full web ESLint and `cargo fmt --all -- --check`
passed, and the three frozen JSON artifacts remain byte-identical to the published head.

```sh
cargo test --locked -p sceneworks-core --test workflow_share every_advanced_builder_in_the_web_app_is_accounted_for
cargo test --locked -p sceneworks-core --test workflow_share
```

The prior native-dump test was not workflow-share parity evidence; these receipts supply that
missing scope. The capture data and the ordinary adapter repair remain valid.

## Review capture corrections on head 613a36ff

All three [automatic review threads](https://github.com/SceneWorks/SceneWorks/pull/3339#discussion_r4502619704)
were read in full and checked against the owning product code. They exposed missing evidence in
the capture tooling rather than production registry changes:

- [Submission actions](https://github.com/SceneWorks/SceneWorks/pull/3339#discussion_r4502619704):
  full Video submits through `Render clip`/`Replace person`, and Training uses a `Start training`
  button with `type="button"`. The shared action selector now follows these actual controls;
  the invalid-override witnesses use the same selector. Focused tests assert real recorded Video
  and Training requests and distinguish the Replace-person submit action from its mode tab.
- [Converted installations](https://github.com/SceneWorks/SceneWorks/pull/3339#discussion_r4502619711):
  the controlled macOS host now supplies Anima's three converted tiers (`bf16`, `q8`, `q4`) and
  complete `mlxTierStates`, as emitted by the Anima converter and API install-state projection.
  The real Image Studio picker exposes all three and its submitted default is Q8. The other
  shipped convert-at-install models use flat output bundles, so they are marked converted where
  the API does so without inventing tier directories. Off-Mac Anima keeps the raw-checkpoint
  shape. Tier disk-size observations are unknown (`null`); no weight files or measurements are
  fabricated.
- [Independent recipe replay](https://github.com/SceneWorks/SceneWorks/pull/3339#discussion_r4502619715):
  each replay starts with a different Camera, Film, Looks and Lighting selection and an unrelated
  prompt/batch/count. The public recipe carrier alone must restore all four axes and reproduce
  the original effective prompts. The complete original persisted draft is no longer reused.

The corrected tooling was overlaid as untracked test files in an isolated detached checkout at
`5a28d413dd9f579cc9607d44f252a045d025bd8e`. Both capture guards verified the exact old HEAD and
unchanged production source paths. The native old-input capture remained byte-identical.
The old-tree capture passed **1,454 tests**, and only `old-observations.json` and
`old-behavior.json` were copied back. Their 1,197-row population and source identity are unchanged.
Semantic observation changes cover 78 Training rows, 117 Video rows (18 gated rows are unchanged),
and three macOS Anima rows in each of full and Simple Image Studio. Default requests are now
recorded for **66 Training** and **81 Video** rows. There are **12,555** scalar/select/chip changes.
The behavior artifact updates 18 appearance witnesses and 38 provider-override witnesses.
All 18 appearance witnesses retain identical original/replay payload batches while recording
and asserting independent destination state.

Mutation checks are load-bearing: reinstating the old submission omission fails all three action
checks; withholding converted tiers fails the Anima picker/request check; deleting appearance
restoration from production `recipeToStudioPatch` fails both the independent replay check and
the corrected menu/batch witness. The original same-state replay test stayed green under that
last mutation, demonstrating its prior false-green. Every temporary mutation was restored;
production source files are unchanged in the repair.

These receipts supersede the previous full Video/Training submission claims, macOS Anima tier
coverage and independent appearance-import claim. Other source data is retained. Full ordinary
Node 22.23.4 web discovery now passes **7,325 tests in 391 files**. Full web lint/build, the native
frozen-input test, all 92 workflow-share tests, clippy, full Cargo formatting and capture guards
are also checked. The candidate-removal control and ordinary/native input parity remain required.

At coordinator handoff, the previously published `613a36ff` head had no failed checks; only
hosted macOS workspace tests were still running. The coordinator owns that pending receipt and
publication. A `gh api` REST job-status read hit HTTP 403 rate limiting; GraphQL rollup reads and
the GitHub connector still worked. The last connector step snapshot showed worker build, both
clippy steps and memory-adapter tests passing before the in-progress workspace tests.
