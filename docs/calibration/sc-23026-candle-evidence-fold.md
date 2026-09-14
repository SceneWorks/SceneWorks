# Candle campaign evidence fold (sc-23026)

Retain the 47 successful captures from
[run 34493340342](https://github.com/SceneWorks/SceneWorks/actions/runs/34493340342)
by merging `story/sc-22738-candle-campaign-34493340342` at
`76807148c1ff30889764adeaeaa0810566c3fa02` into current main. The evidence bundles
are byte-identical to that branch; their measurement revision remains
`e11fd9f0fd26a0eee3a0eb1f4ca7f81c32b5aeb8`. The integration retains main's inference
pin `8fb95eb05cee4cf1e482aa050874fae98ccb3f78` and all subsequent capture repairs.

The packaged source list includes all 47 bundles. Regeneration increases the
anchor store from 146 to 192 rows (71 Candle, 121 MLX), reduces analytic-only
cells from 202 to 156, and retains all 18 exceeded bounds. No new capture runs
are performed by this fold.

Of the 46 additional representative anchors, 44 have current loader-closure
digests. SCAIL2 q4/q8 retain their original digests and are stale against the new
loader closure; no currency attestation is invented for them. The remaining
bundle, Qwen Image q4, is retained and packaged, but the extractor selects the
older `docs/generated/qwen-candle-five-rung-sc-15817.json` representative under
its existing largest-envelope rule (42,048,946,176 versus 33,661,386,752 bytes).
This fold does not change representative selection or promote stale evidence.

The catalog planner classifies all 47 keys as `already_captured` with
`campaign=sc-22738`. That campaign-directory check precedes matrix currency, so
reusing the campaign skips even the two stale SCAIL2 captures and Qwen q4.
To deliberately refresh those records, use a new campaign id and explicitly
select their anchor keys. A new full campaign with `skip_current=true` only
skips records recognized as current by the matrix.

The completed run reported 47 committed captures, 85 capture failures and one
runtime-budget stop. The failure classes, host repairs, engine/adapter fixes,
and remaining full-render verification are recorded in
[the capture repair report](sc-23026-capture-repairs.md). Those historical
failures are not recast as successful measurements by this fold. Story closeout
and another full campaign are outside this PR's publication scope.
