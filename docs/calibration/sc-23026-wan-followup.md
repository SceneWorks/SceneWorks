# Wan follow-up from campaign 34616864537

The rerun reached two remaining Wan defects:

- `wan_2_2:{bf16,q4,q8}:candle` loaded weights but failed while applying its
  calibrated request. The TI2V provider rejects any explicit `video_mode`; its
  calibrated text-to-video route uses the default selector. The capture adapter
  now leaves that field unset only for TI2V. A14B and SCAIL2 retain their explicit
  receipt-bound modes. Regression coverage enters the provider's real CPU request
  scope at all three tiers and rejects the old explicit selector.
- `wan_2_2_t2v_14b:bf16:candle` had a partial upstream snapshot. The first shard
  index reported four missing files in `transformer`; a full publisher inventory
  audit also found all twelve `transformer_2` weight shards absent and its index
  missing. The manifest left this upstream download unpinned, so
  `download_missing=true` correctly refused to fetch it. The manifest now pins
  `Wan-AI/Wan2.2-T2V-A14B-Diffusers` to the existing host snapshot revision
  `5be7df9619b54f4e2667b2755bc6a756675b5cd7`, verified against the publisher API.
  The normal downloader can therefore restore both experts at that exact revision.

The host repair fetches only the missing files at this revision and verifies
replacement weight hashes against the publisher's LFS SHA-256 values. It does not
modify the active campaign checkout or its loaded I2V model. The current run's
already-planned skip/failure results still require a later targeted retry.

These changes produce no new GPU measurements or currency attestations. Native
scope tests verify the repaired request handoff, not a complete TI2V render.
