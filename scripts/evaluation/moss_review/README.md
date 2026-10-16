# Independent review support for sc-26322

This CPU-only postprocessor is separate from the frozen `moss/` runtime closure.
It does not generate images, score pixels, contact a model service, or manufacture
ratings. It imports the existing harness's receipt/PNG validation and translates
real independent observations into its existing accepted report formats.

Freeze review instructions and evaluator identities **before scored execution**:

```text
python scripts/evaluation/moss_review/review.py freeze /ABS/lock.json /ABS/run /ABS/evaluators.json /ABS/review-protocol.json
```

For Windows generation reviewed on macOS, use this order. First download the
Windows generation `protocol.lock.json` and preserve its bytes. Choose and create
an empty persistent local run directory, then run the review `freeze` command
above against that generation lock and the empty local directory **before the
coordinator dispatches scored execution**. This is a separate review protocol;
the Windows bootstrap generation freeze remains unchanged. After generation,
copy the untouched receipts, logs, parity proofs and PNGs into that **same** local
run directory, preserving their relative directory structure. For an explicitly
approved mixed-availability campaign, include all six original calibration
receipts/logs and the private `availability-authorization.json` run mirror; do not
put that unblinded evidence in evaluator packets. Then export and
translate there. Keep originals private and retain the pre-dispatch review freeze
and custody evidence. Do not freeze review after importing scored files, move the
bound local root, edit Windows receipt paths, or reseal copied receipts. The
reader accepts relative Windows backslashes and POSIX slashes while rejecting
absolute/drive/UNC/traversal paths; new receipts use forward slashes. Review loads
the generation lock without running its Windows execution/environment guards.

`evaluators.json` declares exactly `annotator`, `judge`, and `custodian`. Each has:

```json
{
  "identity": {
    "reviewer": "EXACT independent reviewer or context identity",
    "reviewer_kind": "machine_assisted",
    "model": "EXACT evaluator model identifier",
    "version": "EXACT evaluator version or immutable release",
    "tool_revision": "EXACT visual tool/client revision"
  },
  "task_contexts": ["EXACT actual pre-created role context identity"],
  "additional_instructions": "Reviewed role-specific instructions, including ambiguity handling."
}
```

Use actual values, never placeholders in a real freeze. For a human, use
`reviewer_kind: human`, `model: not-applicable`, and record the actual review
procedure version. The three reviewer identities must differ. Pre-create the actual isolated review
contexts before freezing and declare their exact IDs in each role's `task_contexts`
list; these lists must be pairwise disjoint. Records from undeclared contexts or
contexts reused across roles are rejected. The same role context may process
several bounded batches. Fresh vision contexts can share a model but receive only their own
role's packet. Never expose mappings, run receipts, source model names, expected
answers, or custodian output to annotators. Do not put credentials in specifications.

The protocol incorporates fixed pixel-first/judging/custodian instructions, exact
instruction hashes, frozen rubric anchors/thresholds, lock hash and tool/harness
hashes. It binds the exact canonical run root and refuses any existing `scored.*`
artifact anywhere beneath it, including orphan outputs without receipts. Export
and translation cannot switch to another run root. It is not a signature, a trusted
clock, or proof of independence; retain custody and pre-run freeze evidence.
Changing review code/instructions/identities requires an explicit new freeze;
never relabel completed ratings under new provenance.

After all 288 scored receipts exist (success or reproducible failures):

```text
python scripts/evaluation/moss_review/review.py export /ABS/lock.json /ABS/run /ABS/review-protocol.json /ABS/public-review /ABS/private-review
```

The fresh public/private directories must be disjoint. Only the custodian gets
`private-review/mapping.json`. Public PNGs use random opaque names and are hash-checked. Registered native
outputs can embed the workflow (including model/prompt) through
`image_jobs.rs::write_image_asset` / `write_workflow_chunk`. Export therefore
creates review-only derivatives removing PNG tEXt/zTXt/iTXt/eXIf chunks. Original
run files remain untouched/private. The private mapping records original and review
byte hashes, removed chunk kinds, and matching independently decoded RGB/RGBA pixel
hashes (IHDR plus unfiltered row bytes, without color/orientation transforms).
Translation rechecks pixel equality against originals. Unsupported PNG encodings
or unequal pixels fail closed; retain originals and diagnose before another export.
There is no silent metadata drop or claim that derivative bytes equal original bytes.

Two independently sealed public packets are emitted:

- `annotations.packet.json`: up to 288 unique image tasks, dimensions and hashes;
  no scene prompts, expected text, expected box coordinates or model/cell IDs.
- `judgments.packet.json`: exactly 432 randomized pairs, scene prompts/requirements,
  frozen rubric anchors and A/B opaque image references. Failed arms are unavailable.

`annotation-batches/` and `judge-batches/` contain at most 24 tasks each. Give a
fresh `fork_turns=none` visual evaluator only a batch's instructions/opaque paths
and image files. Read images with the visual tool at useful resolution; filenames
and receipts are never evidence of pixel quality. Keep annotation contexts separate
from prompt-aware judging contexts. Store each completed response immediately as
an append-only JSON list outside the public directory. Hash the actual tool/task
transcript, retain its bytes, and record the exact task/context identity. With no
external scoring endpoint, these are actual tool-assisted reviews, not automated
model API calls. Machine-assisted records never become claimed human ratings.

Each annotation record contains:

- `image_id`, `output_sha256` (the review derivative hash from its opaque task);
- `text`: **all** independently observed strings, including unexpected text;
- `boxes`: a list of `{observation_id, label, box}`, with literal observed labels
  and normalized `[x1,y1,x2,y2]` coordinates. Label illegible text honestly;
- common provenance fields below. Do not consult expected strings/boxes.

Each judgment contains `pair_id`, `winner` (`A`, `B`, `tie`, `unavailable`),
`scores: {A: {visual_quality, instruction_following, composition}, B: {...}}`
with integer 1–5 absolute scores per available arm, and
`viewed_image_hashes: {A: SHA256, B: SHA256}`. For an unavailable arm its score/hash
are null. If either arm is unavailable, winner must be unavailable, even when the
other arm is scored. Explain the observed evidence in notes.

All annotation, judgment and custodian-alignment records require:

- `evaluator`: the exact frozen identity object for that role;
- `instructions_sha256`: its frozen instruction hash;
- `task_context`: actual evaluator task/context identity from that role's frozen
  `task_contexts` list; all three roles' actual context sets must remain disjoint;
- `transcript: {path, sha256}`: a nonempty regular file relative to the private
  evidence root, retaining actual visual tool/task messages;
- `notes` and `uncertainty` (`low`, `medium`, `high`).

After independent annotations are complete, the custodian creates one alignment
record per opaque image: `image_id`, `text_indices` (one observed-text index or
null per expected line), `box_ids` (expected element ID to observed box ID or null),
and custodian provenance. Multiple expected elements cannot reuse one observation.
Null produces empty/missing legacy observations, never requested text. Translated legacy annotations bind the original output hash and retain
`review_image_sha256` alongside it. Legacy pair files are private and reference
original run outputs, so the existing harness validates its original byte hashes;
never give these unblinded legacy files to judges. Original
raw records retain extra words and unmatched objects even though the legacy text
score does not measure them. No-other-text judgments must consider those extras.

Merge bounded checkpoints without silently replacing duplicate records:

```text
python scripts/evaluation/moss_review/review.py merge image_id /ABS/observations.json /ABS/annotation-batch-*.json
python scripts/evaluation/moss_review/review.py merge pair_id /ABS/ratings.json /ABS/judge-batch-*.json
python scripts/evaluation/moss_review/review.py merge image_id /ABS/alignments.json /ABS/alignment-batch-*.json
python scripts/evaluation/moss_review/review.py translate /ABS/lock.json /ABS/run /ABS/review-protocol.json /ABS/public-review /ABS/private-review /ABS/observations.json /ABS/ratings.json /ABS/alignments.json /ABS/private-evidence /ABS/translated-review
```

Translation validates exact independent-image coverage, all 432 unique matchups,
image/transcript/instruction/evaluator hashes, per-arm scores, availability and
private alignments. It writes existing-harness-compatible `annotations.json`,
`judgments.json`, `pairs.json`, and `pairs.json.private.json` to a **private** output
directory. The legacy single rubric explicitly means the componentwise minimum
of A/B absolute scores; it is conservative pair usability, not a per-model score.
Actual per-arm scores remain in the original hash-bound rating records and analyst
tables. Then run the original `moss/evaluate.py report` on these translated files.

`analyst-tables.json` contains all nine model matchups × twelve scenarios, four
pair judgments per row, wins/ties/unavailable counts, per-arm absolute quality
means/sample counts, independent text/box scores, source statuses and cost records.
Actual runtime failure rows and remaining unattempted `runtime_censored` rows
remain distinct in failure-kind counts and observation provenance; neither is
retroactively labeled unavailable. Cold/warm latency summaries remain separate. GPU peaks remain grouped by their
measurement method; native device sampling and PyTorch allocator peaks are not
combined. Missing measurements remain null; preparation acquisition/disk evidence
must be read from the separate bootstrap receipts. Repeated image exposures are
not independent samples. These tables support an analyst's actual use-case
shortlist/tradeoffs; they do not infer model superiority or decide inclusion.

`review-receipt.json` binds input and translated output hashes. Original observation,
alignment, rating records and their referenced transcripts must remain immutable
and available. A hash match cannot establish that a reviewer really viewed pixels;
actual task transcripts and custody are necessary. No real ratings exist merely
because CPU fixture tests pass.

```text
python3 -m unittest discover -s scripts/evaluation/moss_review -p 'test_*.py' -v
```
