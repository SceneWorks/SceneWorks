# sc-17775 hand mutations

Historical commits mix classes. These four isolate one variable each so the survey's per-class
claims are measured rather than reasoned.

Each mutation is a single-file edit committed **out of the inference working tree** on top of
`fbb00d6b4147bd220fe324050534708c12fb022d` — blob via `git hash-object -w`, tree via a temporary
`GIT_INDEX_FILE`, commit via `git commit-tree`, held by a throwaway `sc-17775-probe-*` branch that
was deleted after the runs. The inference working tree was never modified and nothing was pushed.

Author/committer identity and dates are fixed (`sc-17775 probe <probe@localhost>`,
`2026-08-05T00:00:00Z`) so the commit SHAs below are reproducible.

| # | Commit | File | Edit | Why this edit |
| --- | --- | --- | --- | --- |
| M1 | `08077be2f3f0b218c94ca30f8e9e78d604b277fb` | `crates/media/candle-gen/candle-gen-flux2/src/memory_strategy.rs` | `KLEIN_CALIBRATION_FINGERPRINT` `…-abi-v2` → `…-abi-v3` | A constant belonging to the crate-mate `flux2_klein_9b`. `flux2_dev` cannot execute it under any input, so a moved digest is a pure crate-mate over-trigger. |
| M2 | `e22dae41c8d63378f8bfebdf8e4492832ad1da7b` | same | append `#[cfg(test)] mod sc_17775_probe { #[test] fn … { assert_eq!(2 + 2, 4); } }` | Test-only code inside the audited crate. Never reachable from production `flux2_dev`, but linked into the `--lib` **test** binary the audit hashes. |
| M3 | `ba6ca1b4ed15fbf8ba3637fcae3d23381851b727` | same | dev `CALIBRATION_FINGERPRINT` `…-blocks-v2` → `…-blocks-v3` | **Positive control.** The provider explicitly declaring its own calibration invalid. If the audit absolved this it would be a false green, which is worse than every false positive in the survey. |
| M4 | `eba3c66eeea21f7ab111968178b3c0864b8ad819` | `crates/contracts/gen-core/src/lib.rs` | append an unreferenced `pub fn sc_17775_unlinked_probe(u64) -> u64` | A `gen-core` change nothing in FLUX.2's closure references. Isolates link-time DCE from the rest of a real `gen-core` commit. |

Run as, for each `<probe-sha>`:

```
node scripts/inference-artifact-audit.mjs --repo D:\repos\inference \
  --captured fbb00d6b4147bd220fe324050534708c12fb022d --compatible <probe-sha> \
  --lane cuda --workdir D:\repos\inference-audit\sc-17524 --out audit-M<n>.json
```

The emitted records are **not** checked in: they name commits that exist in no branch, so a checked-in
record would look like a compatibility claim about a revision nobody can fetch. Results are recorded
in `docs/calibration-invalidation-survey-sc-17775.md` §4.2, and the mutations above regenerate them.

To reconstruct a probe commit, apply the edit in the table to the named file at `fbb00d6b` and commit
it with the fixed identity and date above; the SHA must match.
