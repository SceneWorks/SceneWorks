# Face likeness (identity-likeness score)

The **face-likeness score** is the identity-confidence signal SceneWorks attaches to character
generations across Character Studio (Angles / Poses) and Image Studio ("With Character"). It
answers one narrow question: *does the frontal face in this generated image match the reference
person?* It is **not** a general image-quality, aesthetics, or prompt-adherence score, and must
never be presented as one.

- **Epic:** 4406 (generator-agnostic native ArcFace identity score).
- **Backend scorer:** `crates/sceneworks-worker/src/face_likeness.rs` (sc-4407).
- **Persisted shape:** `recipe.rawAdapterSettings.faceLikeness` (sc-4408).
- **Shared band constants + classifier (single source of truth):** `apps/web/src/faceLikeness.js`
  (sc-4414). The frontend badge (sc-4413) imports `classifyLikeness` / `LIKENESS_BANDS` from it —
  the cut-points are defined once, there.

## What the metric is

The score is the **cosine similarity between two ArcFace face embeddings** — one of the reference
("source") face and one of the largest detected face in the generated image. Both faces are found
with SCRFD and embedded with ArcFace from the InsightFace **antelopev2** pack (`glintr100`), the
same SCRFD + ArcFace stack the InstantID / keypoint / dataset-face paths already provision
(`mlx-gen-face` on macOS, `candle-gen-face` off-Mac — no new weights). The embeddings are
L2-normalized, so the cosine lives in `[-1, 1]`; for real face pairs it is effectively `[0, 1]`.

The source face is embedded **once per job** and cached; each generated image re-embeds only
itself and dots against the cached source. Scoring is **non-fatal** — a failure is logged and
recorded as `null`, never blocking a generation.

The persisted block:

```json
{
  "score": 0.876,
  "detected": true,
  "method": "arcface_antelopev2",
  "sourceAssetId": "asset_…"
}
```

…or, when there is no measurable frontal identity:

```json
{
  "score": null,
  "detected": false,
  "method": "arcface_antelopev2",
  "sourceAssetId": "asset_…",
  "reason": "no_face"
}
```

## Frontal-identity only — the load-bearing limitation

ArcFace is a **frontal face-recognition** model. The embedding is only meaningful when SCRFD
detects a reasonably frontal face with enough confidence (the scorer's `MIN_DET_SCORE = 0.65`,
mirroring the keypoint-extract confidence floor). The score therefore measures *frontal-identity
match*, not "how good the picture is".

Consequences a reader must keep in mind:

- A **high** score means the frontal face is recognizably the reference person. It says nothing
  about pose accuracy, composition, hands, background, or style.
- A **low** score (a real, detected cosine below the bands) means a frontal face *was* found but
  it is **drifting toward a different person**.
- A **missing** score (N/A) means there was **no detectable frontal face to compare** — see below.

### Why profiles, up/down, and full-body poses return N/A

A left/right profile, a strong up/down tilt, an occluded face, or a full-body / turned pose
legitimately has **no detectable frontal face**. In that case the scorer returns an explicit
**`detected: false`, `score: null`** result with a `reason`
(`no_face` | `low_confidence` | `no_source_face` | `embedding_error`) — **not** a low number.

This is deliberate and is the single most important thing not to misread:

> A profile shot is **N/A**, not "0.2". The absence of a frontal-identity signal is *not* evidence
> of poor likeness. Rendering N/A as a low score would punish exactly the angles a character set is
> *supposed* to produce (profiles, looking up/down), and would be dishonest.

The UI must surface N/A as its own neutral state ("No frontal face to score" / "—"), distinct
from a weak likeness. The classifier in `apps/web/src/faceLikeness.js` enforces this: any
`detected: false` or non-finite score maps to the `na` band, never to `weak`.

## The bands

`classifyLikeness(scoreOrBlock)` maps a cosine to one of four states:

| Band         | Condition                       | Meaning                                                  |
| ------------ | ------------------------------- | -------------------------------------------------------- |
| **strong**   | `score >= 0.80`                 | Frontal identity solidly matches the reference.          |
| **moderate** | `0.55 <= score < 0.80`          | Recognizable as the same person, with some drift.        |
| **weak**     | `score < 0.55`                  | Clear identity drift — likely not a confident match.     |
| **na**       | `detected:false` / `score:null` | No detectable frontal face to score (not a low number).  |

Cut-points: `STRONG_MIN = 0.80`, `MODERATE_MIN = 0.55` (defined in `apps/web/src/faceLikeness.js`).

### Rationale — why these edges (anchored to recorded baselines)

The edges are calibrated against **recorded antelopev2 ArcFace cosines** measured on this stack
(InstantID and PuLID-FLUX spikes, epic 4406 surfaces), not invented round numbers:

| Generation / condition                              | Recorded cosine | Source                |
| --------------------------------------------------- | --------------- | --------------------- |
| InstantID frontal identity (E2E, "STRONG")          | ~0.876          | sc-2009               |
| InstantID 9-angle pack (all angles hold)            | 0.81 – 0.89     | sc-2009               |
| InstantID three-quarter                             | ~0.875          | sc-2009               |
| InstantID with face-restore pass                    | ~0.74 → 0.83    | epic 4406 baselines   |
| InstantID pose tier                                 | ~0.71           | epic 4406 baselines   |
| InstantID, landmark ControlNet disabled (collapse)  | ~0.15           | sc-2009               |
| PuLID-FLUX photoreal preset (iw=1.0, sc=4)          | 0.8016          | sc-2012               |
| PuLID-FLUX iw=0.8 (visually drifts)                 | 0.7422          | sc-2012               |
| PuLID-FLUX iw=0.6 (drifts further)                  | 0.5689          | sc-2012               |
| Qwen-Edit Lightning angle-set mean                  | ~0.62           | sc-2003               |
| FLUX.2-klein angle-set mean                         | ~0.52           | sc-2003               |

**`STRONG_MIN = 0.80`** — the floor that captures every result a viewer reads as unmistakably the
same person: InstantID frontal (~0.876) and its full angle pack (0.81–0.89), PuLID-FLUX photoreal
(0.8016), and the top of the InstantID face-restore range (~0.83). It deliberately sits **just
above** the PuLID iw=0.8 case (0.7422) that the spike described as *visually drifting* — that
result should read as **moderate**, not strong. Confidence: **high.**

**`MODERATE_MIN = 0.55`** — the floor below which results are *clear* drift. It sits just below
the closest anchor above it, PuLID iw=0.6 (0.5689, "drifts further") — that borderline result
stays **moderate** — while the soft-identity backbones (FLUX.2-klein ~0.52) and the
landmark-disabled collapse (~0.15) fall into **weak**. Everything in `[0.55, 0.80)` — the pose
tier (~0.71), the lower face-restore edge (~0.74), PuLID iw=0.8 drift (0.7422), Qwen-Edit
Lightning (~0.62), PuLID iw=0.6 (0.5689) — is "recognizable but drifting", i.e. **moderate**.
Confidence: **medium-high**;
the 0.50–0.60 region is genuinely fuzzy across generators, so this edge is the more debatable of
the two. If the badge later proves too lenient or too harsh in this band, this is the knob to
revisit (and it lives in one place).

## Licensing note — antelopev2 (ArcFace / SCRFD) is non-commercial

The face detection + recognition models here come from the InsightFace **antelopev2** pack, which
is licensed for **non-commercial / research use only**. SceneWorks is an **open-source,
non-commercial project** (PolyForm Noncommercial for its own code), so this is in-scope and
acceptable — the same posture under which the project already ships non-commercial model weights.

Precedent and posture:

- SceneWorks downloads weights under the **user's own license acceptance** and does not redistribute
  the antelopev2 pack; the non-commercial obligation falls on the end user.
- The project already ships strictly more-restrictive non-commercial weights (e.g. FLUX.1-dev under
  the BFL Non-Commercial License), and accepted the revenue-gated Stability AI Community License as
  non-blocking — so a non-commercial face pack is well within the established licensing posture.
- This is **not legal advice**. It is recorded here so that anyone evaluating a *commercial* fork of
  SceneWorks knows antelopev2 is a commercial blocker (InsightFace sells a separate commercial
  license) and that the likeness feature depends on it.
