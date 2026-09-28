# Film timeline delivery and editing

The saved timeline owns the cut; the film run owns attempts, assets and review history. Film generation delivers after each selected shot and does not request an export. A run read reconciles its timeline summary against the saved cut.

## Revisioned saves

`PUT /api/v1/projects/:project/timelines/:timeline` accepts `{ timeline, expectedRevision }`. Existing CLI callers may instead send the revision already on their document. An omitted revision means legacy revision zero, never an unconditional overwrite. New timelines start at one; a successful save increments the stored revision under the project lock. A stale save returns HTTP 409, `code: timeline_revision_conflict`, and `context: { timelineId, expectedRevision, currentRevision }` without changing the cut.

The editor rebases edits by stable track/item IDs. Disjoint edits merge automatically; conflicting item edits offer the local or stored version. Undo/redo applies only its user operation over the current timeline, preserving later unrelated deliveries. CLI editing rereads a conflict and refuses the write with instructions to review and retry.

## Atomic delivery

The shared harness calls `POST .../timelines/:timeline/film-deliveries` with `{ runId, shotOrder, timeline }`. The supplied timeline is an incoming take/audio description, not a replacement document. Under the same project lock, the store reads the current cut, resolves incoming assets' measured durations, merges only run-owned deliveries, validates, and saves. Existing picture items are identified by `filmHarness.runId` and `shotId`; new IDs digest both identities. Delivery receipts use `runId:shotId:a<attempt>`.

`filmAssembly.runs[runId]` stores `shotOrder`, `deletedShots`, `appliedDeliveries`, `audioDelivered`, and any `trimConflicts`. Shot order includes pending and failed slots; insertion follows the saved order without rebuilding existing clips. Ordinary saves maintain deletion tombstones and explicit restoration clears them. Replayed delivery cannot recreate a deleted shot. Existing audio edits survive delivery; a newly inserted shot shifts only the relevant downstream picture positions and run-owned shot-relative audio.

A fitting replacement preserves the existing source range, position, speed, volume, transitions and identity. An incompatible replacement is retained as a durable pending choice, with `code: timeline_replacement_trim_conflict`, the item/shot, old source range, measured duration, choices and incoming take. The delivery returns the saved timeline containing that conflict so the run can continue and reopening retains the resolver.

Resolve with `POST .../film-deliveries` and `{ runId, resolveShotId, resolution, expectedRevision }`. Choices are `reset`, `keepCurrent`, and `clamp` when at least 0.1 seconds remains after the in-point. Resolution validates against the latest cut under the lock. Keeping the current take records the delivery receipt while preserving its asset, so replay cannot undo the choice. The generated attempt remains in the run's history.

Ordinary editor replacement jobs also retain incompatible trims in `generationTrimConflicts`, with the same explicit choices and revisioned saves. Their completion applications use deterministic job item identities and CAS retries.

## Export identity

Explicit export freezes an immutable document under the project's `timeline-exports` directory and records `timelineRevision` in the job payload. The worker carries that identity into the result and render asset metadata. A later successful edit or delivery makes that export stale by revision comparison; it never automatically dispatches another export. Legacy exports without a source revision are conservatively stale.

Local validation covers store CAS/concurrent winners, ordered delivery/replay/tombstones, valid and incompatible replacements, fake-worker interleaving, CLI edits/resume, and Vitest/jsdom save/history resolvers. Real browser and media acceptance belong to the integrated film workflow campaign.
