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

### S1 — No download starts without an acknowledgment, on the routes named below (technical, enforced)

`config/manifests/builtin.models.jsonc` declares `requiresLicenseAcknowledgment: true` on both
`minimax_h3` and `minimax_h3_ref`. Enforcement is stated here as **the specific routes actually
gated**, not as a universal — an earlier draft of this section claimed "every surface", and a review
found two unlisted doors that fetched the same weights unrefused. Those are now closed; the point of
enumerating rather than generalising is that the next such door should be visible as an addition to
this list.

**Server side — the enforcement boundary.** These are the doors that can fetch H3 weights, and what
each answers. Enforcement is on the SERVER because no client-side check binds a browser on another
machine (the remote-access lane, epic 4484), a workflow envelope's suggested action, or `curl`.

| Route | Gated? | Where |
| --- | --- | --- |
| `POST /api/v1/models/:id/download` | **Yes** — 403 `license_acknowledgment_required` unless the body carries `licenseAcknowledged: true`. Keyed on the catalog **id** *and* on the **repos the job would queue** (the selected download plus its co-requisites), so an entry that names a flagged entry's repo without carrying the flag itself is refused here exactly as it is on `/api/v1/jobs`. | `create_model_download_job`, `apps/rust-api/src/models.rs` |
| `POST /api/v1/jobs` with `model_download` / `model_import` / `model_convert` / `lora_download` / `lora_import` | **Yes** — same 403 and same code, keyed on the payload's **repo** (and on a huggingface.co `sourceUrl`). | `validate_raw_job_payload`, `apps/rust-api/src/jobs.rs` → `ensure_job_payload_license_acknowledged` |
| `POST /api/v1/models/import` (JSON) | **Yes** — same 403, same repo-keyed predicate. | `queue_model_import_job`, `apps/rust-api/src/models.rs` |
| `POST /api/v1/models/import` (multipart) | **Yes** — the same gate, on the same predicate. The form is *not* usable as a bare remote fetch: without an upload `file` the parser answers 400 before the gate is reached. But it parses `repo`/`sourceUrl` alongside the file, and the worker prefers `repo` over the uploaded `sourcePath`, so a file-plus-`repo` request does reach the restricted weights — and is refused identically. | `model_import_request_from_multipart`, `apps/rust-api/src/models.rs` |
| `POST /api/v1/loras/import` | **Yes** — same 403, same repo-keyed predicate. This route takes a caller-supplied `repo`/`sourceUrl` and **never consults the LoRA catalog** for it, so what the LoRA catalog declares has no bearing on what it can reach. | `queue_lora_import_job`, `apps/rust-api/src/loras.rs` |
| `POST /api/v1/loras/:id/download` | **Yes** — same 403, same repo-keyed predicate, on the repo it resolves **from the catalog entry** named by the path id. A caller cannot point this route at a repo (an unknown id 404s, a body `repo` is inert), but that is a claim about who *chooses* the repo, not about whether the chosen repo is restricted — so it was gated. | `create_lora_download_job`, `apps/rust-api/src/loras.rs` |

`model_convert` is on the list for a reason that is not obvious from its payload: it names no `repo`
at all. Its LTX arm hands the payload's **`baseRepo`** to `ensure_ltx_upscaler_cached` →
`ensure_hf_files_cached` — a real download — and `upscalerFile` is a **glob**, so `"**"` pulls the
whole named repo. Adding the job type to the gate alone would have been **inert**: the shared
predicate read only `repo` and `sourceUrl`. It now reads every repo-bearing payload key
(`LICENSE_GATED_REPO_PAYLOAD_KEYS` — `repo`, `baseRepo`, `sourceRepo`), which is what makes the job
type's presence bite.

The repo-keyed half is what closes the generic queue route, which enqueues `job_type` + payload
**verbatim**: `run_model_download_job` reads `repo` / `files` / `revision` straight out of the
payload with no catalog lookup between the request and Hugging Face. Its index is built from
`downloads[].repo` across **every** entry declaring the flag, **including co-requisite rows** —
MiniMax-H3's shared text encoder and both VAEs come from `MiniMaxAI/MiniMax-H3` itself — and from
the **unfiltered** manifest, because every H3 download row is `platforms: ["macos"]` and an
OS-filtered index would leave the gate absent on Linux and Windows. Repo keys are compared
case-insensitively, and a trailing `.git` — the git-remote spelling of the same repository, which
passes the worker's `validate_hf_repo_id` — is stripped before the comparison.

**The limit of a repo-keyed index, stated plainly.** It recognises Hugging Face repos. A `sourceUrl`
pointing at a **non-huggingface.co host** — a third-party mirror, or `cdn-lfs.huggingface.co` and the
other CDN/LFS hostnames that serve the same bytes from a different name — is **not** matched, and
`/models/import` answers 201. That is inherent to keying on the repo rather than on the bytes: the
same weights republished anywhere else are, to this index, a different source. It is not a defect in
the enforcement above and the table does not claim otherwise, but it is the boundary, and this
section has twice been read as claiming more than it does. Closing it would mean gating by content
rather than by name, which nothing here attempts.

**Client side — where the terms are shown.** The acknowledgment UI lives on the Models screen
(`apps/web/src/screens/ModelManagerScreen.jsx`). The gate renders when the card renders **any
download affordance**: the model is not installed, its cache is incomplete, an update is offered,
or — for a quant matrix — **any tier is still missing**. That last clause is not a refinement but a
correction: the API marks a matrix model `installed` when *any* tier is present
(`install_state_for`), so a MiniMax-H3 with q4 installed and q8/bf16 missing rendered no notice and
no checkbox while its tier panel was still offering downloads, and the choke point then refused the
click by naming a checkbox that was not on screen. While the gate applies, the Download button,
every quant-tier checkbox, the tier panel's download button and the Update button are disabled.
Acceptance persists per model id in localStorage — which does **not** survive a desktop relaunch, so
the gate re-raising itself on a partially-installed model is the recovery path, not an edge case.

`createModelDownloadJob` (`apps/web/src/hooks/useModelsAndLoras.js`) is the client choke point: it
refuses when `licenseAcknowledgmentBlocked(model)` and attaches `licenseAcknowledged` otherwise. The
Models screen, the Simple UI's model manager, the first-run Setup Wizard, the studio availability
gates, the workflow drop and the Update button all call it. It reads the **catalog entry's own
flags**, so every caller must hand it the real entry — the workflow drop resolves the requirement id
against the full catalog for exactly that reason; passing a `{ id }` stub silently disabled the gate.
Two surfaces additionally decline to *offer* what they cannot gate: the Setup Wizard does not list a
model requiring acknowledgment at all (it has no licence UI, and first-run bulk-queues what is
ticked), and the Simple UI hands off to the Models screen rather than toasting a refusal the user
could not act on.

This flag is deliberately **decoupled from `gated`** (sc-17227). `gated` means "the download needs a
Hugging Face credential"; `MiniMaxAI/MiniMax-H3` is a **public** repo, so before the decoupling the
gate could only either fail open or demand a token that does not exist. One consequence is worth
stating plainly: the server gate is scoped to `requiresLicenseAcknowledgment` and **not** to `gated`,
so a merely-`gated` model is still backstopped only by Hugging Face's own 401.

**What this does not claim.** This paragraph scopes *authorization*; it does not scope
*reachability* — which routes are covered is the table above, and that is the part a reviewer should
re-derive from source rather than from this sentence. The gate stops an *unacknowledged* download; it
is not an authorization check and cannot be. A request that sets `licenseAcknowledged: true` by hand
is allowed — but that request is itself an affirmative assertion that the user has accepted, which is
what §V.2 asks us to obtain. Nor does accepting make an otherwise unauthorized use authorized: a
user in an Excluded Territory who ticks the box is still not licensed (§V.4), which is the open
question in [Open, and not resolvable here](#open-and-not-resolvable-here), not something this
safeguard resolves.

Covered by `apps/web/src/screens/ModelManagerScreen.test.jsx` (Models screen, tier panel, Update
button, and the partially-installed matrix), `apps/web/src/simple/SimpleModelManager.test.jsx`,
`apps/web/src/screens/SetupWizard.licenseGate.test.jsx`,
`apps/web/src/hooks/useModelsAndLoras.licenseGate.test.jsx` (the choke point),
`apps/web/src/hooks/useWorkflowDrop.test.jsx` (the workflow drop's entry resolution), and
`apps/rust-api/src/tests/catalog.rs` — where
`license_acknowledgment_generic_jobs_route_refuses_what_the_typed_route_refuses` runs the typed and
generic requests against one app instance,
`license_acknowledgment_model_import_is_refused_for_a_restricted_repo` covers the import route,
`license_acknowledgment_lora_import_is_refused_for_a_restricted_repo` covers `/loras/import`,
`license_acknowledgment_lora_download_refuses_a_catalog_named_restricted_repo` covers
`/loras/:id/download` (a catalog LoRA whose `source.repo` is restricted, plus the weaker
caller-cannot-supply-a-repo claim it used to pin alone),
`license_acknowledgment_typed_download_refuses_a_repo_another_entry_declares` constructs the
entry-without-the-flag case on `/models/:id/download`,
`license_acknowledgment_model_convert_is_refused_for_a_restricted_base_repo` covers `baseRepo`, and
`license_acknowledgment_is_not_bypassed_by_a_git_suffix` covers the `.git` spelling. The
manifest declarations are pinned by
`tests/test_builtin_manifest_audit.py::test_minimax_h3_requires_license_acknowledgment_without_a_credential`,
over a set of ids **derived** from the flag rather than listed, and
`::test_every_entry_naming_a_license_gated_repo_carries_the_flag_itself` keeps the shipped catalog
free of the entry-without-the-flag shape.

### S2 — The user is told which restrictions apply, in the gate itself (§V.2 notification)

§V.2 requires notifying each user *that the restrictions apply*, which a bare "accept the license"
checkbox does not do. The manifest's `licenseNotice` field carries the text, rendered inside the gate
box above the checkbox. It names, with section citations:

* the **Applicable Territory** (§I.3, §I.5, §V.4) and every Excluded Territory by name — the
  European Union, the United Kingdom, the Republic of Korea, the United States of America;
* the **Acceptable Use Policy** (§V.1, Exhibit A), calling out **item 12**, the
  machine-generated-content disclosure, and stating plainly that SceneWorks does not make that
  disclosure on the user's behalf;
* the **$20 M yearly-revenue ceiling** (§IV.1) above which separate written authorization is
  required — the licence's own measure is revenue, not earnings, and the notice says so;
* the **bar on improving other AI models** (§V.3): the Works and their Outputs may not be used to
  improve any other artificial intelligence model, other than MiniMax H3 or its Model Derivatives.
  This one is called out because SceneWorks ships a **LoRA trainer, dataset captioning and a
  training studio** — it is the §V restriction the product's own feature set is most likely to
  reach, and a user reading a three-item list would reasonably have concluded training on H3 output
  was unrestricted;
* that the licence is **non-transferable** (§II) and that a "Licensee" is whoever uses the Works
  (§I.9) — i.e. the user, not only SceneWorks;
* that the Excluded-Territory scope is **"not yet", not "not ever"** — §II records that MiniMax
  continues to evaluate those territories and invites anyone in them to ask about a licence, so the
  notice gives the contact address the agreement names (`api@minimax.io`, §IV.1) rather than
  leaving the territory line reading as a permanent bar.

`tests/test_builtin_manifest_audit.py::test_minimax_h3_license_notice_names_the_restrictions_it_notifies_of`
asserts each of those substantively; dropping any one territory, the disclosure sentence, the §V.3
item or the revenue wording fails it.
`…::test_minimax_h3_shipped_notice_names_the_same_restrictions` pins the same set in the §III.4
NOTICE that ships in the application, which is the only copy a user without a checkout has.

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

**This is not the whole obligation.** §IV.2 says "prominently display … on the user interface", and
the Models card is one screen a user may never revisit after installing. The **generation
surfaces** — Video Studio, the Simple UI's video studio, the queue and the asset detail view — are
where the model is actually used, and they do not carry the attribution today. Landing it there is
**sc-17161**'s work (the user-facing lane for the family), and it is a precondition for shipping,
not an optional extra: until it lands, §IV.2 is discharged only on the Models screen. The manifest
field is already the single source, so those surfaces read `model.ui.attribution` rather than
hard-coding a second copy.

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
of the review is already automated — `npm run check:license-coverage`, the manifest and shipped-NOTICE
audits in `tests/test_builtin_manifest_audit.py`, the per-surface gate tests listed under S1, and the
server-side refusal tests in `apps/rust-api/src/tests/catalog.rs` all run in CI and fail on drift.
Each of those guards has been mutation-checked individually: removing the choke-point refusal, the
typed-route rejection, the raw-jobs-route rejection, the import-route rejection, the co-requisite
rows or the unfiltered read from the repo index, the case-normalisation of a repo key, the
acknowledgment stamp that keeps a retry authorized, the partially-installed-matrix clause in the UI
predicate, the workflow drop's entry resolution, the Setup Wizard exclusion, the Simple UI hand-off,
the Update-button guard, or any one named restriction from either notice fails a test rather than
passing quietly.

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
