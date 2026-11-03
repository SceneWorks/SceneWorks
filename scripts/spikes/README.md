# Archived feasibility spikes

The completed sc-3487 DWPose, sc-3489 Real-ESRGAN, sc-3668 AuraSR and sc-4387 DWPose
diagnostic payloads are preserved in the [immutable artifact archive](../../docs/archive/historical-artifacts.md).
Their original paths, bytes, modes and digests are pinned in the manifest. Restore **all groups**
outside the checkout to preserve experiment-image and root-pose dependencies before following
the historical commands. Retrieval never runs a spike or regenerates evidence.

Retained findings and reproduction requirements:

- [DWPose / sc-3487](../../docs/sc-3487/spike-findings.md)
- [Real-ESRGAN export / sc-3489](../../docs/sc-3489/spike-findings.md)
- [AuraSR / sc-3668](../../docs/sc-3668/spike-findings.md)
- [DWPose diagnostic / sc-4387](../../docs/sc-4387/spike-findings.md)

The Real-ESRGAN exporter remains recoverable for model provenance; today's Rust runtime uses the
published ONNX artifacts and does not invoke these historical scripts. Archived code is unchanged,
including references to its original environments and retired workers.
