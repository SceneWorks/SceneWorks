# sc-8253 / sc-8278 — klein angle-set identity A/B, 2026-07-27

Driver: `sc_8253_8278_identity_angle_ab` (PR #1921). Scoring: `lora_eval_harness`.
Reference: `~/Datasets/Uhura/Uhura.jpg` (1366×880, ar 1.55, face fraction 0.167).
11 canonical angles × seeds 8001/8002 per arm; `square_on` carries 8003/8004 too
as a noise probe. klein-9b Q8, 4 steps, 1024², guidance 1.0. Face detection was
**100% in every arm**, so no detection failures confound any of this.

## Arm means (primary seeds only — like-for-like)

| arm | reference | image_guidance | identity_cosine | n |
|---|---|---|---|---|
| `squish_off` | stretched to square | off | 0.3138 | 22 |
| `square_off` | production `fit_engine_image` | off | 0.4784 | 22 |
| `square_on` | production `fit_engine_image` | 1.5 | 0.5952 | 22 |

## Effects

| | delta | paired positive | min delta | max delta |
|---|---|---|---|---|
| **sc-8253** squish → square | **+0.1646** | **22/22** | +0.0681 | +0.2942 |
| **sc-8278** guidance off → 1.5 | **+0.1168** | **22/22** | +0.0236 | +0.1979 |
| combined | **+0.2814** | | | |

## Noise floor (the sc-6541 discipline)

Measured from `square_on` at 11 angles × 4 seeds — one condition, seed varying only:

- within-angle stdev, mean **0.0442**
- within-angle range, mean 0.0999 (max 0.2280)

**Both effects clear it.** sc-8253's mean delta is 3.7× that stdev and its
*smallest* single delta (+0.068) still exceeds it. sc-8278's is 2.6×; its
smallest delta (+0.024) sits inside the noise, so individual angles are not
separable — but every one of 22 pairs moved the same way.

A sign test on 22/22 gives p ≈ 2.4e-7. Treating the 11 angles as the unit
(the conservative reading, since two seeds of one angle share a reference) still
gives 11/11 in both seeds, p ≈ 5e-4. Decisive either way.

This is the bar sc-6541 could not reach — it got 11/14 (p≈0.057) and 12/16
(p≈0.077) and had to retract them as pseudoreplicated. That retraction does
**not** apply here: sc-6541 subsampled a single *trained checkpoint* per
condition, whereas no training happens in this A/B. Each generation is an
independent draw from the same generator, so the pairs are genuinely paired.

## Scope limits — read before quoting

1. **Absolute numbers do not reproduce the stories' figures** (sc-8253's
   0.469/0.597, sc-8278's 0.38/0.73). Those used the sc-6599 harness reference,
   which is not present locally. Different subject ⇒ different absolute scale.
   What transfers is the *effect*, which is what both fixes were asserted to
   deliver. sc-8253 originally measured +0.128; this run gives +0.165 on a
   somewhat wider reference (ar 1.55 vs 1.38), so a larger squish and a larger
   recovery is the expected direction.
2. **One reference subject, one model.** Direction and magnitude, not
   cross-subject generalization.
3. `same_prompt_spread` from the harness stayed null — stems are
   `{angle}_s{seed}`, so every prompt_id is unique and the harness had no
   same-prompt group to measure. The noise floor above was computed directly
   from per-image output instead. A future driver revision should set
   prompt_id = angle so the harness can do this itself.
4. Prompt adherence is flat across arms (0.193 / 0.188 / 0.188), i.e. neither
   fix bought identity by drifting from the prompt. Sharpness rises slightly
   with each fix (992 → 1092 → 1137), so no softening penalty either.

## Verdict

Both fixes work, measured, with the effect clearly separated from generation
noise. sc-8253's aspect correction is the larger single lever (+0.165);
identity-strength adds a further +0.117 on top of it, and they compound to
+0.281 — 0.314 → 0.595, nearly doubling measured identity.

## Reproduce

Driver `sc_8253_8278_identity_angle_ab` (`crates/sceneworks-worker/src/image_jobs/tests.rs`),
scored by `eval_lora_outputs` (`crates/sceneworks-worker/src/lora_eval_harness.rs`).

```
AB_OUT=~/identity-ab \
  cargo test -p sceneworks-worker --lib -- --ignored --nocapture sc_8253_8278_identity_angle_ab

# per arm — prompts.json is angle-keyed so same_prompt_spread has a group to measure
REF_DIR=~/identity-ab/ref GEN_DIR=~/identity-ab/<arm> PROMPTS_JSON=~/identity-ab/prompts.json \
  EVAL_LABEL=<arm> cargo test -p sceneworks-worker --lib -- --ignored --nocapture eval_lora_outputs
```

Env: `AB_REF` (reference image), `AB_KLEIN_WEIGHTS`, `AB_SEEDS`, `AB_SPREAD_SEEDS`,
`AB_ANGLES` (subset for a cheap smoke), `AB_GUIDANCE`. Re-running skips existing
PNGs, so an interrupted sweep resumes. ~56 s/generation on Apple Silicon; the
full 88-generation sweep is ~80 min.
