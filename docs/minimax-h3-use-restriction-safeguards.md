# MiniMax-H3 use-restriction safeguards and reporting plan (sc-17227)

**Status: written and in force for everything SceneWorks controls today. Two items are open and
belong to Michael, not to this document — see [Open, and not resolvable here](#open-and-not-resolvable-here).**

This is SceneWorks' §V.5 plan under the **MiniMax H3 Community License Agreement**
(`apps/desktop/licenses/minimax-h3/MiniMax-H3-Community-License.txt`, shown in-app under
**About → Licenses**). §V.5 applies because SceneWorks makes available a product that "permits the
generation of Outputs using MiniMax H3":

> you must, before making that product or service available and throughout its operation,
> implement, maintain, test, and periodically review reasonable and proportionate technical and
> organizational safeguards designed to prevent and mitigate access, uses, and Outputs that violate
> this Section V or Exhibit A […] You must maintain a reasonably accessible mechanism for reporting
> suspected violations.

"Reasonable and proportionate" is the operative standard, and proportionality here is unusual, so
the shape of the product is stated first rather than assumed.

## What SceneWorks is, and what that makes possible

SceneWorks is an **AGPL-3.0, locally-run application**. Users clone or install it and run it on
their own hardware; H3 weights are fetched from Hugging Face **onto the user's machine**, and
generation runs **on that machine**. There is no SceneWorks-operated inference service, no
server-side content pipeline, no account system tied to generation, and no telemetry that could
observe a prompt or an output.

Three consequences follow, and they are the reason this plan looks the way it does:

1. **Prevention has to happen before the weights land, not during generation.** SceneWorks cannot
   inspect, score, or refuse a render it never sees. Every safeguard with real force is therefore a
   *pre-download* one.
2. **"Terminate repeat violators" has no account to terminate.** There is no login, no licence key,
   no server-side kill switch, and adding one would mean building surveillance into an offline app.
   What SceneWorks can actually do is remove or disable *distribution* — the catalog entry, the
   re-hosted weights, and the release — which it can do quickly and which is recorded below.
3. **Everything about the app is inspectable.** The licence text, the manifest declarations and the
   gate are all in the repository under AGPL-3.0, so a reviewer can verify every claim in this
   document against source rather than taking it on trust. Each claim below cites its file.

## Safeguards in force

### S1 — The model is not downloadable until the user accepts the licence (technical, enforced)

`config/manifests/builtin.models.jsonc` declares `requiresLicenseAcknowledgment: true` on both
`minimax_h3` and `minimax_h3_ref`. The Models screen renders a licence gate on any uninstalled
model carrying that flag and **keeps the Download button disabled until the user checks the
acknowledgment box** (`apps/web/src/screens/ModelManagerScreen.jsx`,
`requiresLicenseAcknowledgment` → `licenseAckRequired`). The acceptance persists per model id.

This flag is deliberately **decoupled from `gated`** (sc-17227). `gated` means "the download needs a
Hugging Face credential"; `MiniMaxAI/MiniMax-H3` is a **public** repo, so before the decoupling the
gate could only either fail open or demand a token that does not exist.

Covered by `apps/web/src/screens/ModelManagerScreen.test.jsx`
("blocks a license-acknowledgment download until the box is checked (sc-17227)") and by
`tests/test_builtin_manifest_audit.py::test_minimax_h3_requires_license_acknowledgment_without_a_credential`.

### S2 — The user is told which restrictions apply, in the gate itself (§V.2 notification)

§V.2 requires notifying each user *that the restrictions apply*, which a bare "accept the license"
checkbox does not do. The manifest's `licenseNotice` field carries the text, rendered inside the gate
box above the checkbox. It names, with section citations:

* the **Applicable Territory** (§I.3, §I.5, §V.4) and every Excluded Territory by name — the
  European Union, the United Kingdom, the Republic of Korea, the United States of America;
* the **Acceptable Use Policy** (§V.1, Exhibit A), calling out **item 12**, the
  machine-generated-content disclosure, and stating plainly that SceneWorks does not make that
  disclosure on the user's behalf;
* the **$20 M revenue ceiling** (§IV.1) above which separate written authorization is required;
* that the licence is **non-transferable** (§II) and that a "Licensee" is whoever uses the Works
  (§I.9) — i.e. the user, not only SceneWorks.

`tests/test_builtin_manifest_audit.py::test_minimax_h3_license_notice_names_the_restrictions_it_notifies_of`
asserts each of those substantively; dropping any one territory or the disclosure sentence fails it.

### S3 — Full licence text ships with the app and is not reachable only online

The verbatim agreement, the Apache-2.0 text for the Qwen3-VL-32B encoder, and the §III.4 NOTICE are
tracked under `apps/desktop/licenses/minimax-h3/` and rendered by **About → Licenses**
(`apps/web/src/data/bundledLicenses.js`). `scripts/check-license-coverage.mjs` fails the build if a
shipped model has no wired licence entry, so this cannot silently rot.

### S4 — Attribution on the user interface (§IV.2)

`ui.attribution` on both entries carries **"Powered by MiniMax H3"**, rendered as its own line on
the model card (`.model-card-attribution`). §IV.2 requires the exact string *MiniMax H3*; the
hyphenated product name `MiniMax-H3` does not contain it, which is why the attribution is a separate
field rather than an assumption about the model's display name.

### S5 — Upstream safety behaviour is not weakened

§V.5 forbids knowingly disabling or materially weakening safeguards. SceneWorks does not strip,
disable, or bypass any upstream safety component. The nearest comparable case in the codebase runs
the other way: the Chatterbox PerTh provenance watermarker is a **hard-required** co-requisite whose
absence fails the job outright (`config/manifests/builtin.models.jsonc`, the `perth` component;
`crates/sceneworks-worker/src/audio_jobs.rs`). There is no `disable_watermark`-style flag anywhere in
the repository.

### S6 — Reporting mechanism (§V.5, "reasonably accessible mechanism")

Suspected violations of §V or Exhibit A involving SceneWorks — including SceneWorks' own conduct —
are reported through the channels in [`SECURITY.md`](../SECURITY.md):

* **GitHub private vulnerability reporting** — repository **Security** tab → *Report a
  vulnerability* (preferred; keeps the report attached to the repo), or a public issue at
  <https://github.com/SceneWorks/SceneWorks/issues> where privacy is not needed;
* **email — michael@trefry.net**.

The same address is reachable from the shipped NOTICE
(`apps/desktop/licenses/minimax-h3/NOTICE.txt`), so a user who has only the built application can
still find it. MiniMax's own contact for licensing questions is `api@minimax.io` (§II, §IV.1).

Reports are triaged on the same path as security reports. No separate SLA is invented here that
would not be honoured; the honest commitment is the one in `SECURITY.md`.

### S7 — Investigate, mitigate, and the limits of "terminate"

On a good-faith report or actual knowledge of a violation, in order:

1. **Investigate** — reproduce against the current release or `main`, and determine whether the
   cause is (a) something SceneWorks distributes, (b) SceneWorks' own use, or (c) a third party's
   use of a local install.
2. **Mitigate what is in our control.** For (a) and (b) that includes: removing or disabling the
   `minimax_h3` / `minimax_h3_ref` catalog entries so no further install can occur; removing the
   converted weights from `SceneWorks/minimax-h3-mlx`; and shipping the change in a release. These
   are ordinary, fast operations — a manifest edit and a Hugging Face deletion.
3. **Where the violation is a third party's local use, say so rather than pretend.** SceneWorks has
   no account to suspend and no remote control over an installed copy. The available actions are to
   stop distributing to that party where a distribution channel exists, to notify MiniMax at
   `api@minimax.io`, and to cooperate with MiniMax's own enforcement.
4. **Record the outcome** on the tracking story so the review in S8 has something to review.

Claiming a termination capability that does not exist would be worse than stating the limit.

### S8 — Periodic review (§V.5, "test, and periodically review")

This document, the licence corpus, and the gate are reviewed **whenever any of these changes**: the
upstream licence or Acceptable Use Policy is revised (Exhibit A carries its own "Last revised" date —
currently **August 2, 2026**); the H3 manifest entries change; the gate code changes; or a report
under S6 is received. Absent any of those, at each release that ships the family. The mechanical part
of the review is already automated — `npm run check:license-coverage`, the manifest audits in
`tests/test_builtin_manifest_audit.py`, and the gate tests in `ModelManagerScreen.test.jsx` run in CI
and fail on drift.

## Exhibit A item 12 — measured, not asserted

Item 12 prohibits publishing generated content to a public environment "without clearly and
prominently disclosing that such information and/or content is machine-generated". This is a
restriction **on the person publishing**, and the honest finding is that SceneWorks does not
discharge it for them. What the application actually does today, measured against the source:

| Mechanism | Present? | What it actually does |
| --- | --- | --- |
| C2PA / Content Credentials | **No** | No `c2pa` dependency, signer, or manifest anywhere in the repo or `Cargo.lock`. No IPTC `DigitalSourceType` marker. |
| IPTC / XMP / EXIF on output | **No** | `crates/sceneworks-core/src/workflow_png.rs` — no EXIF, no XMP is written. The only EXIF code in the repo *strips* it from **input** training images. |
| PNG text chunks on generated images | **Yes**, but producer metadata | Two chunks: `sceneworks:workflow` (a JSON recipe envelope with a `producer` block naming SceneWorks) and the A1111-style `parameters` chunk ending `software: SceneWorks` (`workflow_png.rs`, `workflow_parameters.rs`). Neither contains any statement that the content is AI-generated. |
| MP4 metadata on generated clips | **Yes**, but producer metadata | The same JSON envelope in the standard `comment` tag (`crates/sceneworks-core/src/workflow_mp4.rs`). No `encoder`/`title`/`artist` tags. Applies to MiniMax-H3 renders like any other video. |
| WAV metadata | **No** | `write_wav_pcm16` emits a bare 44-byte RIFF/WAVE header — no `LIST`/`INFO`, no `ISFT`, no ID3. |
| Video-editor timeline export | **Strips everything** | `crates/sceneworks-worker/src/media_jobs.rs` passes `-map_metadata -1` on both the concat and crossfade paths, deliberately, so an export carries **no** provenance at all. This is the surface that produces the file a user actually publishes. |
| Watermarking of image/video output | **No** | The only watermarker in the product is PerTh, on Chatterbox cloned-voice **audio** only. No image or video watermark, visible or invisible. (An MLX Gaussian-Shading implementation exists for **Mage** in the separate `inference` repository; it is unrelated to H3 and unverified from here.) |
| User-facing "this is AI-generated" disclosure | **No** | The one "AI generated" string in the product is an in-app thumbnail badge in the Video Editor media bin (`apps/web/src/components/editor/MediaBin.jsx`), derived from local database state. It is never written into a file and never seen by a recipient. |
| Metadata embedding toggle | **Yes**, user-controlled | `embedWorkflowInImages` (default on) and "Save a copy without the workflow" let the user remove even the producer metadata. Framed throughout the UI as a **privacy/recipe-sharing** choice, not a disclosure. |

**Conclusion.** Producer identification exists and is partial; a machine-generated-content
*disclosure* does not exist anywhere in SceneWorks' outputs, and the timeline export — the most
likely publication path — strips what little there is. Item 12 therefore rests entirely on the user,
so the only honest safeguard is to **tell them so before they download**. That is what S2 does, in
the licence gate, in the manifest's own words.

Building an actual disclosure mechanism (C2PA Content Credentials, an IPTC `DigitalSourceType`
marker, or a disclosure step at export) is a real and worthwhile piece of product work, but it is
**not** an H3 obligation and it would apply to every model in the catalog, not this one. It is out of
scope here and tracked separately rather than half-built.

## Open, and not resolvable here

Two things remain, and neither is an engineering task:

1. **The request email text.** MiniMax's authorization (recorded verbatim on sc-17227, activity
   18709) is expressly "conditioned upon […] continued compliance with the commitments and
   representations set forth in its request email". Those commitments are therefore binding product
   requirements, and this plan cannot claim to cover them while the text is unrecorded.
2. **The territory decision.** §V.4 bars use of the model *and its outputs* outside the Applicable
   Territory. **An acknowledgment gate does not cure this** — a user in an Excluded Territory who
   ticks the box is still not authorized. Whether SceneWorks ships the family publicly, ships it
   disabled by default, or does not ship it, is a ship/no-ship call that depends on item 1 and on
   any further answer from `api@minimax.io`.

Until item 2 is settled, nothing in the user-facing lane should advertise the family; that
constraint is carried by sc-17227's blocking links to sc-17161, sc-17162 and sc-17240.
