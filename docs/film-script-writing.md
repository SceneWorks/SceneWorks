# Writing your first film script

Start with a small silent scene: one person, one location, two visible actions. Your first test is
successful when the editor turns your text into an editable plan and delivers separate playable
clips. It does not need to look like a finished film.

This guide uses the current **Editor → Film** workspace: **Script**, **References**, **Plan**,
**Shots**, **Review**, and **Sound**. For installation, provider settings, recovery, and the full
workflow, see the [Film Editor operator guide](film-editor.md).

## What to write

You can paste ordinary prose. You do not need JSON, a special screenplay file, or reference images.
Write what a viewer should see, in the order it happens:

- Name the subject consistently: “Mara” throughout, rather than switching between several names.
- Give each paragraph one main visible action. A blank line starts another extracted beat.
- Describe the setting, important objects, and how the action ends.
- Make emotion visible: “Mara pauses with her hand on the box” is more concrete than “Mara remembers
  her childhood.”
- Keep persistent details consistent across paragraphs: clothing, object color, location, and time
  of day. References can help later, but do not guarantee identity or continuity.
- Put overall lighting, visual style, and camera preferences in **Visual direction** after extracting
  the script. Exact shot framing and duration can be edited in **Shots**.

A **beat** is something that happens in the story. A **shot** is one planned video clip. They are not
necessarily one-to-one: the planner may split a beat into several shots. A target duration is a
planning target, not a command to make every shot that length. Shot durations must fit the selected
video model's supported settings.

## Copy-and-paste first test

Create a fresh test project, open **Editor**, and choose **Start from a script**. If a Film workspace
is already open, use **New film draft**. Set **Film title** to `The red box — first test`.

Paste only the following two paragraphs into **Original prose or screenplay**:

```text
Mara, wearing a plain blue jacket, stands beside a wooden table in a quiet workshop. A small red box rests on the table. Mara places her right hand on the closed box and pauses.

In the same workshop, Mara wears the same blue jacket beside the wooden table. The small red box is still closed. Mara slides the box a short distance toward the center of the table, then lifts her hand away. The box remains on the table.
```

Choose **Extract editable beats and dialogue**. Before using a planning model, check:

- There are two beats, `B001` and `B002`, matching the two paragraphs.
- There are no dialogue lines.
- The synopsis describes Mara and the red box. Edit it if needed.

Then put this in **Visual direction**:

```text
Naturalistic live-action look. Soft daylight from a workshop window. Keep Mara's blue jacket, the wooden table, and the small red box consistent. Simple stable framing, one clear action per shot. No dialogue, music, titles, or on-screen text. Prefer two shots covering the two beats.
```

The last sentence is a planning preference, not a guaranteed shot count. Inspect the result.
For this first test, leave the extracted target duration initially; adjust it if planning or preflight
reports that it cannot be covered by the selected model's legal shot durations. Do not copy another
model's resolution, duration, or memory limits.

**Extract first, then edit.** Extracting again replaces the structured brief, including your edited
synopsis, visual direction, target duration, beats, and dialogue. It does not just append new lines.
Use **Save draft** after reviewing your edits.

## Test in small stages

### 1. Check authoring without generating media

Leave **References** empty. Move between Script and the other steps, save, and reload the draft.
Your script and edited brief should remain. This stage needs no model inference.

If extraction does not produce the two expected beats, fix the text or extracted summaries before
planning. There is no value in rendering a misunderstood story.

### 2. Generate and inspect a plan

On **Plan**, keep **Built-in prompt refiner (default)** and choose an installed, compatible
**Video model**. The local prompt-refiner also needs its weights and a compatible worker; Qwen3.6-27B
is not required. Check Models and Workers if availability is unclear.

Choose **Generate candidate plan**. This uses the planning model but does not render video. Inspect
its findings and candidate shots before choosing **Replace current edited plan with this candidate**.
The candidate does not replace your edited plan automatically.

For the example, look for:

- Both beats are covered in order: hand placed on box, then box slid and hand removed.
- The same person, jacket, room, table, and closed red box appear in the relevant descriptions.
- Shots use `text_to_video`, with no invented reference bindings or reference-backed continuity roles.
- There is no dialogue or added sound plan you did not request.
- Each shot has one manageable action, a clear starting situation, and a clear ending situation.

A different shot count is not automatically a failure. Correct the plan in **Shots** if it invents
an action or makes the simple sequence unnecessarily complicated. If bounded planning fails, read
its findings and edit the draft or manual shot plan; repeatedly rendering will not repair a bad plan.

### 3. Check preflight, then render

On **Shots**, inspect **Shots and render controls**, the selected shots, and their durations. Use an
available render regime and settings supported by the chosen model. Turbo is useful when the UI
reports a compatible installed recipe; it is not available for every configuration.

On **Sound**, confirm generated picture audio is **Mute** and leave dialogue and beds unconfigured
for this silent test. Writing “silent” in the script is not a substitute for the audio policy.

Save the draft and choose **Run preflight**. This does not render video. Correct blocking findings
and inspect **Effective requests**, particularly model, conditioning, duration, geometry, adapters,
and steps. Keep finite run/shot/attempt budgets and heed memory admission; this guide does not
prescribe a model-independent budget or completion time.

Choose **Render selected shots** only when you are ready to spend inference time. For the simple
example, render the short plan together. Completed shots should become individual clips in the
**Cut** strip and the film timeline; an export should not start automatically.

### 4. Inspect the clips and save a cut

Use **Open in Timeline**. Play each available clip and check both the product behavior and the image:

| Check | What success looks like |
| --- | --- |
| Delivery | Each completed shot is a separate selectable, playable clip. |
| Action | The intended hand/box action is visible; note missed or invented actions. |
| Continuity | Compare jacket, box, table, and room across shots; record visible changes. |
| Editing | Trim or reorder a clip, save, and reload; the saved cut retains your edit. |
| Decisions | Generated takes remain unreviewed until you explicitly accept or reject them. |
| Export | No export exists merely because rendering finished. |

If a later shot is still rendering, try trimming the earlier one before it arrives. The later clip
should appear without resetting your saved trim. Do not treat different model output as a UI failure:
“clip never arrived” and “clip arrived but the box changed color” are different observations.

Save the cut, choose **Export current cut**, and play the resulting file. Compare its order, trims,
and silence with the saved timeline. Edit and save the timeline once more: the earlier export should
be marked stale, and a new export should require another explicit action.

## Add dialogue after the silent test works

The initial extraction is a simple text parser, not a complete screenplay interpreter. Its currently
recognized dialogue form is an uppercase speaker name immediately followed by one non-empty line:

```text
Mara rests her hand on the closed red box in the workshop.
MARA
This stays here.

Mara slides the red box toward the center of the wooden table and lifts her hand away.
```

This produces two action beats and one dialogue entry for Mara, attached to the preceding beat.
Keep the whole spoken utterance on the line immediately after the speaker. Do not put a blank line
or a parenthetical such as `(whispering)` between the name and the words: the parser reads only that
next line as dialogue. Put delivery direction in prose or edit the extracted fields afterward.

Other parsing details matter:

- `INT.` and `EXT.` scene headings are recognized, but become their own beats. They are not silently
  merged into the following action paragraph. Plain prose is simpler for a first test.
- Avoid standalone all-caps titles, action labels, or `FADE OUT` followed by text; short uppercase
  lines can be mistaken for speaker names. Use the separate Film title field.
- Markdown headings, timestamps, bullets, and camera commands are not a formal scripting language.
  They may become ordinary beat text. Write plain paragraphs and review the extraction.
- Pasting dialogue does not by itself prove speech synthesis is configured, audible, or lip-synced.
  Configure supported dialogue text, voice/model or prerecorded audio in **Sound**, inspect its shot
  placement, and verify the actual timeline preview and exported audio.

## Add references and complexity one change at a time

After the silent example works, duplicate the experiment as a new draft so you can compare results:

1. Add one character reference on **References**, give it a clear role name and description, and
   approve it after checking the image. Bind it to supported shots and inspect preflight. Mention
   the same character consistently in the script; a name in prose does not create an image binding.
2. Add one short dialogue line and verify its placement and audibility before adding a music bed.
3. Try a longer scene or more complex camera motion only after these simpler runs are understood.
4. Compare a different planner only after keeping the script, video model, and settings stable.
   Qwen3.6-27B is an optional large local model; an external connection is an explicit separate choice.

For a reusable starting structure, write two to four short paragraphs describing setting, subject,
visible action, and the state left for the next beat. Add visual direction in its own field. Avoid
starting with a long screenplay, rapid location changes, crowds, many simultaneous actions, or a
critical requirement for readable text: they make it harder to identify why a first test failed.

## When a test fails

| Symptom | First thing to check |
| --- | --- |
| Wrong beats or dialogue | Paragraph breaks, uppercase labels, and the extracted editable brief. |
| Planning refuses to start | Selected planner installation and worker availability; original script is non-empty. |
| Candidate fails validation | Exact findings, beat coverage, legal model settings, and reference bindings. |
| Preflight blocks rendering | Named model/worker, installation, license, geometry, memory, or budget finding. |
| A clip is missing or failed | Operations and shot status, job details, and retained error—not just the preview. |
| Clip plays but looks wrong | Prompt/action complexity, selected take, references, and actual model output. |
| Dialogue is absent or doubled | Sound configuration, mute policy, clip placement, and the saved timeline. |
| Export differs from the edit | Save state, exported timeline revision, stale indicator, and dropped-audio reasons. |

Record the project/draft/run and shot IDs, script, planner/video model, relevant settings, exact
error, expected behavior, actual behavior, and a screenshot or clip. Preserve the failed take and
run record. Use the operator guide for bounded **Cancel**/**Resume** and take replacement; do not
restart an entire experiment merely to hide its first failure.

A good first result is a reproducible small workflow with understandable findings. Visual quality
and cross-shot consistency are separate things to evaluate, not guarantees made by the planner.
