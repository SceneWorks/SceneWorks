# Model download receipts and manifest updates

SceneWorks records a completed Hugging Face install in
`data/models/<repo>/.sceneworks-download-complete.json`. The versioned receipt identifies the repo,
model and variant, the manifest patterns used at install time, the exact resolved files, and (for
new receipts) the snapshot revision. A receipt is evidence of a completed install, not merely that a
download directory exists.

At startup the catalog evaluates the current manifest and retained receipts independently:

| Manifest change | Existing install | Catalog result | Required action |
| --- | --- | --- | --- |
| Primary files renamed or repacked, `breaking` omitted/false | Complete receipted old set | `installed`, `updateAvailable: true` | Existing generation keeps using the entire old receipted set; update is optional. |
| Primary files renamed or repacked, `breaking: true` | Complete receipted old set | missing/repairable, `updateAvailable: true` | Install the new set before use. |
| New soft co-requisite (`required: "soft"`) | Primary complete, dependency absent | `installed`, `updateAvailable: true` | Existing generation remains usable; update installs the enhancement. |
| New hard co-requisite (`required: "hard"` or omitted) | Primary complete, dependency absent | incomplete/blocked, repair available | Repair installs the required dependency. |
| Current files partially present, no complete receipt | Torn | `incomplete`, repair available | Repair; partial files are never treated as a completed install. |
| Pre-receipt install exactly matches the current manifest | Complete current set | `installed` and receipt backfilled | Future non-breaking manifest changes retain protection. |

LoRA receipts follow the same rule. If `source.file` changes, the loader resolves the old exact
receipted adapter and the catalog advertises an update. It must never silently substitute another
`.safetensors` file.

## Release checklist

1. Decide whether existing bytes remain runnable. Use `breaking: true` only when the old complete
   file set cannot safely load with the released runtime.
2. Mark newly required dependencies explicitly: `required: "hard"` blocks use; `"soft"` preserves
   use and offers an update.
3. Prefer immutable upstream revisions. An in-place rehost under the same model ID is appropriate
   only for a compatible, non-breaking refresh; publish a new model ID when behavior, architecture,
   licensing, or user-visible identity changes.
4. Exercise the Rust API install-state matrix, worker receipt resolution, LoRA resolution, schema
   validation, and web eligibility tests before shipping.
5. Verify a retained old snapshot loads as one atomic receipted set. Never combine old and current
   manifest files to make a seemingly complete model.
