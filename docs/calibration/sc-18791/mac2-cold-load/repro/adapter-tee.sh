#!/bin/zsh
# Tee the harness's provider-protocol stdin to a replay file, then run the real adapter.
N=$(date +%s)
tee "/Volumes/Data/calibration/sc-18791/diagnostic/repro/stdin-$N.json" | /Volumes/Data/calibration/sc-18791/diagnostic/SceneWorks/target/release/memory-mlx-adapter
