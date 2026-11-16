# Continue a completed LoRA run

Completed LoRA jobs offer **Continue training** in the Queue and in Training Studio's
completed runs. Enter additional steps; the form shows the cumulative total. For
example, 3,000 completed steps plus 1,000 additional steps trains through step 4,000.

`POST /api/v1/training/jobs/:job_id/continue` accepts only
`{"additionalSteps": 1000}`. The server copies the original dataset and training
settings, validates the new total against the target's bounds, and allocates a new
job, LoRA id, output directory and manifest entry. `sourceJobId` links the job to its
parent; `payload.continuation` and both training and adapter provenance record the
parent LoRA and job, initial step, additional steps and total steps. Completion
registers a separate adapter through the normal training registrar.

Generic checkpoint continuation copies the terminal factors and optimizer state
into the new namespace. Only the copied training identity's total-step horizon is
extended. Dataset identity, optimizer moments and update counters are retained.
The learning-rate schedule is evaluated at the saved update against the extended
total, including the original warmup. It can therefore rise from the old schedule's
final rate; the already-completed steps are never replayed. This is an extension of
a finished schedule, not numerical equivalence to originally planning the longer run.

The shared MLX trainer loop and Candle flow-match driver retain recovery state at
successful completion even when intermediate checkpoints are disabled. Iris uses
its existing cross-run checkpoint resume and a separate output namespace.

Continuation requires a checkpoint at the completed step, not merely the most
recent intermediate checkpoint. Older exports without terminal recovery state,
unsupported native recovery formats (including MOSS and FLUX.3), and runs ending
inside a gradient accumulation window are refused with an explanation. There is
no fallback to training from scratch or to an earlier checkpoint. Importing an
export with a fresh optimizer is outside this action's contract.

The source job, its exported adapter, recovery files and registered manifest entry
are unchanged. Sibling continuations can queue independently. A failed/canceled
generic continuation resumes through the existing retry action into its own output
directory; Iris uses its existing Training Studio resume flow.
