# MOSS same-run retry provenance review

The failed-job retry path may retain a successful backend from attempt 1 while its peer finishes in attempt 2. These changes allow that combination within one workflow run, while retaining each producing attempt and job. They do not permit cross-run or changed-source reuse.

## Contracts

- Capture and cross-load artifacts use exact `PHASE-BACKEND-attempt-N` names. Uploads preserve earlier attempts. Downloads remain untouched; a fresh sibling staging directory copies the newest named artifact for each required phase/backend into the existing canonical layout. Selection never falls back from a failed latest artifact to an older success.
- Selection retains original artifact names, attempts and capture-driver hashes. Finalization retrieves the run's paginated `jobs?filter=all` and artifact metadata with the existing Actions-read token. It requires the newest actual execution of each phase/backend to match the selected bundle, even when a later failed execution uploaded nothing. Skipped jobs do not replace an earlier execution.
- Exact deterministic backend job-name suffixes allow the reusable-workflow caller prefix but require one unique match. Job run, attempt, source head, completed/success conclusion and available runner identity must agree. Artifact ID, name, run/head and optional archive digest are retained with the authenticated job rows in fresh finalization output.
- Each inspection case retains the full workflow identity from its original preview index. Finalization compares it with that case's capture bundle. Named findings, actual review timestamps and reviewed PNG hashes remain mandatory. Old review documents lacking per-case workflow identity must be recorded again from their original preview indices; provenance is never invented.
- Each cross-load receipt binds the SHA-256 of the opposite-backend capture driver it actually consumed. Replacing a capture invalidates only the transfer evidence that consumed it. Original source, pin, manifest, dataset, weight, member and learning assertions remain unchanged.
- The pre-build cache inventory uses the native HF-variable precedence and OS-home default, follows ordinary Hub file symlinks, and checks only snapshot/file presence. It reads no weight payload or header. Missing files fail before Cargo; native byte/header, device and resource authentication remain authoritative.
- `real-weights` is an explicit Mac routing choice alongside the existing choices. `unspecified` remains the default and is refused. The workflow introduces no HF override, download, new device selection, host lock or matrix.

## Validation

Independent focused source review passed for retry selection, newest-job authentication, artifact and inspection provenance, cross-load dependency hashes, cache presence and routing. The cache module also received an independent native-resolver/catalog comparison during integration.

Commands run without model execution:

```sh
node --test scripts/moss-training-acceptance.test.mjs scripts/moss-training-artifacts.test.mjs scripts/moss-training-cache-inventory.test.mjs
# 47 passed
node --test scripts/moss-native-api-validation.test.mjs scripts/moss-consumer-assertions.test.mjs
# 61 passed
/tmp/codex-sc26584-audit/campaign-runtime-venv/bin/python -I scripts/ci/moss-training-acceptance/test_verify_evidence.py
# 42 passed; Python 3.12.14, jsonschema 4.26.0, Pillow 12.3.0
```

Node syntax, Python compilation, diff checks and actionlint passed. Actionlint used a temporary configuration declaring the repository's existing `cuda` self-hosted label; stock actionlint otherwise reports that existing custom label as unknown. Tests exercise the actual CLI's non-Actions/wrong-source refusal and the assembled controller's real missing-cache inventory before any Cargo command.

## CI_IMPACT

No GPU jobs, training cases, learned computation or numerical settings are added or reduced. The new tests use synthetic metadata and literal file fixtures. Cache checks add file/directory stats before compilation. Retry selection adds bounded local metadata reads and copies only the selected bundles into fresh staging; original downloaded attempts remain unchanged. Final authentication normally needs one jobs request and one artifacts request, paginated at 100 rows with a ten-page bound per resource and a shared 120-second retrieval deadline. Native campaign acceptance and final-head CI remain separate required evidence; these tooling tests do not establish model learning or device acceptance.
