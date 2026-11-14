# SW-7037 historical option carriers

`historical-builders.json` records outputs from SceneWorks
`d3f7b05bf22826e44fd43da19cff0985fb17d68b`, before the options-core work.
Run `node scripts/capture-option-upgrade-corpus.mjs` from the repository root to
reproduce them with that commit's real JavaScript modules (Vite SSR only; no
network requests or model execution). The capture script records the invocation.

The Image/Simple/Audio/Video/training/regeneration rows execute historical
builders; settings rows execute the historical persistence writers. YuE2 rows
come from the historical checked-in builder-output fixture (whose original
NOTICE documents `UPDATE_YUE2_WEB_FIXTURE=1`). Job/recipe/training-plan rows
preserve the historical Rust migration contract snapshots. The film input is
the historical courier/workshop plan. Core tests additionally exercise a frozen
compiled plan and dispatch body produced by the film compiler, whose production
source was unchanged between the baseline and this item; the capture is in
`film_compile::tests::historical_film_builder_output_upgrades_without_changing_dispatch_intent`.
MCP tests execute extracted, unchanged historical argument structs and real body
builders. Core tests compile the archived real v1 share reader.

Ordinary CI only reads this bounded corpus and runs pure serializers/upgraders;
it neither refreshes goldens nor runs inference. Keep provenance when adding rows.
