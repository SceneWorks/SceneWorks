# Scaffold suite determinism investigation — sc-26664

Date: 2026-10-22. Baseline: `origin/main` at `4b4d2e817`.
Environment: macOS / Apple Silicon, Node v24.15.0, 18 available logical CPUs.
All full-suite runs are serial, normal priority, with the package's default test-file concurrency.
No retries, concurrency reduction, GPU campaigns, or production deadline changes.

## Unchanged baseline

Five invocations of `npm run check`, before any edits:

| Run | Parallel-segment tests | Passed | Failed | Skipped | Duration (s) | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1345 | 1336 | 3 | 6 | 164.265 | A, B, C |
| 2 | 1345 | 1336 | 3 | 6 | 164.426 | A, B, C |
| 3 | 1345 | 1336 | 3 | 6 | 163.531 | A, B, C |
| 4 | 1345 | 1336 | 3 | 6 | 164.804 | A, B, C |
| 5 | 1345 | 1335 | 4 | 6 | 167.480 | A, B, C, D |

- **A:** `exact pinned validator accepts sealed golden receipt and rejects every gate mutation`
- **B:** `preflight requires the exact clean inference checkout and current Cargo permanent pin`
- **C:** `inference preflight requires every exact inventory and native-hook artifact`
- **D:** `a wall-time ceiling fails closed and removes the owned group`

A–C have the same setup error: the producer fixture's default sibling checkout
`../sc-22261-inference-integration` does not exist. They are not timing failures.
Verification supplies the existing `STARVECTOR_TERMINAL_INFERENCE_TEST_ROOT` setting,
pointing to an inference clone containing the exact required commit
`0b084cbf46b5b4f4f561305ef116268b57eecf7f`. No tests are skipped to bypass these errors.
Because the first segment failed, the `&&` matrix segment did not run in the baseline.

D reproduced on run 5, taking 3616 ms and failing with
`AssertionError: the uncapped two-second sample sleep won`.
The test compared real event timestamps (`time.time()`), even though the guard's runtime
policy uses a quantized monotonic clock. Initial observation occurs before the runtime
deadline is armed. Startup and scheduling time therefore invalidated the elapsed-time assertion.

## Root causes and fixes

1. **Watchdog observation:** record requested policy sleeps through the existing test clock seam;
   assert they respect the 0.2-second deadline despite a 2-second sample interval.
   Retain the real owned-process-group exit, reason, and no-survivor assertions.
2. **Pip observation:** numeric progress overwrites the latest state, while heartbeats only sample
   it. Requiring an intermediate byte count is invalid when output is batched. Use terminal
   transfer state with a mocked parent policy clock. Exercise new-download and completed-transfer
   resets separately, keeping real-child privacy and stalled-child coverage.
3. **Failure cleanup:** pre-register watchdog close/error observers, bound its lifecycle, and
   stop its owned process tree on failed actions or waits. Remove fixture directories. Register
   inference-worktree cleanup independently of successful assertions and restore environment
   overrides with test teardown.
4. **Writable repository state:** the removed-source matrix probe edited tracked source artifacts.
   Its separate test segment prevented collisions only within one suite invocation. Run the real
   generator against copied modules and inputs in a unique temporary fixture instead; assert the
   source checkout stays unchanged while each fixture artifact is mutated.

The audit did not establish a common leaked lock or fixed port behind the named failures.
Node test files use process isolation, so same-file environment mutations are not automatically
cross-file leaks. The reported Rust catalog-cancellation test was inspected, but no causal defect
was demonstrated; no Rust code was changed. The reported sandbox `kill EPERM` was not reproduced.

## Verification

Focused checks:
- Matrix and producer fixtures: 10/10 passed, including injected fixture failure and Git-worktree removal.
- Two overlapping matrix invocations: 2/2 passed in each, with source bytes unchanged during mutations.
- Pip provision file: 51/51 passed; changed pip tests passed three additional targeted runs.
- Watchdog file after the review fixes: 81/81 passed, including forced group cleanup, spawn failure,
  and delayed PID readiness.
- Isolated pip mutations independently remove the new-download reset and the completed-transfer reset:
  each corresponding regression fails while the other remains green.
- Isolated watchdog mutation removes the runtime sleep cap: requested sleep changes from `[0.2]`
  to `[2.0]`, failing the new policy assertion. Production Python was not edited.

One independent review identified three test-harness gaps, repaired in one fix pass:
forced shutdown needs identity-verified owned-group cleanup (the sentinel does not reap a group
merely because its monitor dies); spawn rejection needs an immediate handler before asynchronous
readiness IO; and PID readiness must explicitly fail at its harness backstop rather than silently
proceed. Regressions cover a stopped guard/root, a missing interpreter, a child exiting inside the
action, a throwing action, and delayed PID creation beyond the original short readiness window.

The test-only hang backstops follow the existing 60-second harness convention; policy clocks,
heartbeat observations, and explicit readiness barriers—not longer waits—provide determinism.
Production watchdog and acquisition deadlines are unchanged.

Five full post-fix invocations, with `STARVECTOR_TERMINAL_INFERENCE_TEST_ROOT` pointing to the
available inference clone and no other test/concurrency overrides:

| Run | Total tests (both segments) | Passed | Failed | Skipped | Combined duration (s) | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1355 | 1349 | 0 | 6 | 168.138 | 0 |
| 2 | 1355 | 1349 | 0 | 6 | 166.519 | 0 |
| 3 | 1355 | 1349 | 0 | 6 | 166.058 | 0 |
| 4 | 1355 | 1349 | 0 | 6 | 167.677 | 0 |
| 5 | 1355 | 1349 | 0 | 6 | 167.405 | 0 |

The parallel segment reports 1353 tests (1347 passed, 6 skipped); the matrix segment reports
2 passed. The six skips are unchanged from baseline. Eight additional regressions run in the
parallel segment. Logs are retained locally under `.tmp/check-determinism/` as `baseline-N.log`
and `fixed-N.log`. Syntax checks on all four changed test files and `git diff --check` passed.

### Requested preflight

`npm run preflight -- --base main` was attempted and exited 1:

```text
npm error Missing script: "preflight"
```

Commit `b60d1a142` deliberately removed the command and script on 2026-10-21.
This fix does not restore that gate. The applicable scaffold suite is run directly;
no Rust or web application code is changed.
