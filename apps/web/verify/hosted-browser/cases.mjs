// sc-26687 S8 browser cases. Every case drives the REAL production frontend through
// its rendered controls and asserts the request the app actually serialized. Nothing
// here calls a component helper, an oracle, or injects UI state.
import { IMPORTED_MODEL_REF, MINIMAX_REF, MANAGED_REPAIR_REQUEST } from "../../verify-fixtures.mjs";

const SHORT = 5000;
const LONG = 15000;

export class CaseAssertionError extends Error {}

function assert(condition, message) {
  if (!condition) throw new CaseAssertionError(message);
}

const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])):value;
function equal(actual, expected, message) {
  if (JSON.stringify(canonical(actual)) !== JSON.stringify(canonical(expected))) {
    throw new CaseAssertionError(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

export function sameRequest(entry, method, pathname) {
  if (entry.method !== method) return false;
  try {
    return new URL(entry.url).pathname === pathname;
  } catch {
    return false;
  }
}

export function matchingRequests(recorder, method, pathname) {
  return recorder.entries.filter((entry) => sameRequest(entry, method, pathname));
}

async function waitFor(predicate, { timeout = LONG, interval = 75, label = "condition" } = {}) {
  const started = Date.now();
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() - started > timeout) throw new CaseAssertionError(`timed out waiting for ${label}`);
    await new Promise((done) => setTimeout(done, interval));
  }
}

export async function waitForRequest(recorder, method, pathname, options = {}) {
  return waitFor(() => matchingRequests(recorder, method, pathname)[0] ?? null, {
    ...options,
    label: `${method} ${pathname}`,
  });
}

// Awaiting actual DOM state instead of a one-shot assert: React state updates (e.g. the
// repair-request effect that opens the import panel) land after the click returns.
async function waitForCondition(predicate, { timeout = LONG, interval = 75, label = "condition" } = {}) {
  const started = Date.now();
  for (;;) {
    const value = await predicate();
    if (value) return value;
    if (Date.now() - started > timeout) throw new CaseAssertionError(`timed out waiting for ${label}`);
    await new Promise((done) => setTimeout(done, interval));
  }
}

async function navTo(page, label) {
  await page.getByRole("button", { name: label, exact: true }).first().click();
}

async function waitForApp(page) {
  await page.locator("nav .nav-item").first().waitFor({ state: "visible", timeout: LONG });
}

async function modelsTab(page) {
  await page.locator(".models-surface").waitFor({ state: "visible", timeout: LONG });
  await page.getByRole("tab", { name: /Video/ }).first().click();
  await page.locator(".models-card-grid").first().waitFor({ state: "visible", timeout: LONG });
}

function cardFor(page, name) {
  return page.locator(".model-card").filter({ hasText: name }).first();
}

async function openImportedSurface(page, name) {
  const card = cardFor(page, name);
  await card.waitFor({ state: "visible", timeout: LONG });
  await card.locator("summary").filter({ hasText: "Imported checkpoint" }).click();
  return card;
}

// A rendered form field is a <label> that wraps its own control. Playwright's getByLabel matches a
// label's FULL text, and a label that wraps a <select> also carries that select's OPTION text
// ("Family" + "Auto-detect" + every family name), so an anchored label regex can never resolve a
// select field: it resolves nothing, and the case times out on the product's markup instead of
// failing on it. Read the field through its label element — the field name is the label's own
// leading text and the control is the label's own element — which is the same lookup the panel's
// own unit tests use.
function labelledField(scope, name) {
  // The field name is interpolated into a RegExp, so it is escaped: a name carrying regex
  // metacharacters must match itself rather than a pattern.
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return scope
    .locator("label")
    .filter({ hasText: new RegExp(`^${escaped}`) })
    .locator("input, select")
    .first();
}

async function pickAsset(page, fieldLabel, assetName) {
  const field = page
    .locator(".asset-picker-field")
    .filter({ has: page.locator(".asset-picker-label", { hasText: fieldLabel }) })
    .first();
  await field.waitFor({ state: "visible", timeout: LONG });
  // The trigger is found STRUCTURALLY, not by its name: the button's text changes with the
  // field's state ("Select image" when the field is empty, "Change" when it already holds a
  // value), so a name pattern silently stops matching a field that has a selection — and the
  // case then times out on the product's MARKUP instead of failing on its behaviour. The trigger
  // is the field's own dialog-opening button; it sits directly in the field's head for the
  // reference pickers and inside the head's actions group for the frame pickers, so it is
  // located by the behaviour that identifies it rather than by where it is nested.
  await field.locator('button[aria-haspopup="dialog"]').first().click();
  const modal = page.locator(".asset-picker-modal").last();
  await modal.waitFor({ state: "visible", timeout: LONG });
  await modal.locator(".asset-picker-card").filter({ hasText: assetName }).first().click();
  await modal.getByRole("button", { name: "Use Selection" }).click();
  await modal.waitFor({ state: "hidden", timeout: LONG });
}

async function replayAsset(page, assetName) {
  await navTo(page, "Assets");
  await page.locator(".asset-tile").first().waitFor({ state: "visible", timeout: LONG });
  await page.locator(".asset-tile").filter({ hasText: assetName }).first().dblclick();
  const cta = page.getByRole("button", { name: "Use this recipe" });
  await cta.waitFor({ state: "visible", timeout: LONG });
  await cta.click();
}

// ---------------------------------------------------------------------------
// Case 1 — Model Manager repairs the named resident component, never a stock DiT.
// ---------------------------------------------------------------------------
async function caseModelManagerRepair(page, recorder) {
  await page.goto("/");
  await waitForApp(page);
  await navTo(page, "Models");
  await modelsTab(page);
  const card = await openImportedSurface(page, "My H3");
  await card.getByRole("button", { name: "Repair audio", exact: true }).click();

  const request = await waitForRequest(recorder, "POST", "/api/v1/models/my_h3/download");
  equal(
    request.body,
    { requestedGpu: "auto", componentId: "audio_vae" },
    "repair request body must bind the named resident component",
  );
  assert(request.status == null || request.status === 200, `repair response was ${request.status}`);
  return { evidence: { request: { method: request.method, path: "/api/v1/models/my_h3/download", body: request.body } } };
}

// ---------------------------------------------------------------------------
// Case 2 — managed reimport opens the existing panel, preserves metadata/draft,
// and never submits by itself.
// ---------------------------------------------------------------------------
async function caseModelManagerReimport(page, recorder) {
  await page.goto("/");
  await waitForApp(page);
  await navTo(page, "Models");
  await modelsTab(page);
  const card = await openImportedSurface(page, "My H3");
  await card.getByRole("button", { name: "Reimport transformer", exact: true }).click();

  const toggle = page.locator(".checkpoint-import-toggle").first();
  await toggle.waitFor({ state: "visible", timeout: LONG });
  // The repair-request effect opens the panel in a follow-up commit; assert the REAL
  // rendered state rather than racing it.
  await waitForCondition(
    async () => (await toggle.getAttribute("aria-expanded")) === "true",
    { label: "reimport must open the existing import panel" },
  );
  assert((await toggle.getAttribute("aria-expanded")) === "true", "reimport must open the existing import panel");

  const form = page.locator('form[aria-label="Add to SceneWorks"]').first();
  await form.waitFor({ state: "visible", timeout: LONG });
  // Wait for the managed repair seeding effect, not just the panel container.
  await waitForCondition(
    async () => (await labelledField(form, "Family").inputValue()) === "minimax-h3",
    { label: "the reimport declaration to be seeded" },
  );

  // Scoped to the OWNERSHIP choices: the page carries other checked radios (the Simple/Advanced
  // tab), and the first checked radio on the page is not the ownership choice. The ownership
  // choice is the panel's FIRST decision, so it sits OUTSIDE the managed form — the form holds the
  // source-kind radios (Upload / Local copy / URL / Hugging Face / Civitai). The assertion then
  // still means something: among the OWNERSHIP radios, the selected one is the managed one.
  const ownership = page.locator('section.checkpoint-import [role="radio"].checkpoint-ownership-choice[aria-checked="true"]').first();
  assert(((await ownership.textContent()) ?? "").includes("Add to SceneWorks"), "managed ownership must stay selected");

  const typeValue = await labelledField(form, "Type").inputValue();
  const familyValue = await labelledField(form, "Family").inputValue();
  const baseValue = await labelledField(form, "Compatible base model").inputValue();
  const partitionValue = await labelledField(form, "MiniMax-H3 partition").inputValue();
  const seededName = await labelledField(form, "Name").inputValue();
  equal({ typeValue, familyValue, baseValue, partitionValue }, {
    typeValue: "video",
    familyValue: "minimax-h3",
    baseValue: "minimax_h3_ref",
    partitionValue: "ref2va",
  }, "reimport must preserve the imported model's declaration metadata");
  equal(seededName, "My H3", "reimport must seed the imported model's name");

  await labelledField(form, "Name").fill("Edited name");
  await form
    .locator('[aria-label="Where the checkpoint comes from"]')
    .getByRole("radio", { name: "URL", exact: true })
    .click();
  await labelledField(form, "Source URL").fill("https://example.test/new.safetensors");

  await toggle.click();
  await waitForCondition(
    async () => (await toggle.getAttribute("aria-expanded")) === "false",
    { label: "Hide options must collapse the panel" },
  );
  assert((await toggle.getAttribute("aria-expanded")) === "false", "Hide options must collapse the panel");

  await new Promise((done) => setTimeout(done, SHORT));
  equal(matchingRequests(recorder, "POST", "/api/v1/models/import"), [], "opening/editing a repair must not auto-submit an import");
  equal(matchingRequests(recorder, "POST", "/api/v1/models/my_h3/download"), [], "reimport must not trigger a download");

  await card.getByRole("button", { name: "Reimport transformer", exact: true }).click();
  await toggle.waitFor({ state: "visible", timeout: LONG });
  await waitForCondition(
    async () => (await toggle.getAttribute("aria-expanded")) === "true",
    { label: "reopening the repair must show the panel again" },
  );
  assert((await toggle.getAttribute("aria-expanded")) === "true", "reopening the repair must show the panel again");
  const reopenedName = await labelledField(form, "Name").inputValue();
  const reopenedUrl = await labelledField(form, "Source URL").inputValue();
  equal(reopenedName, "Edited name", "the edited import draft name must survive dismiss/reopen");
  equal(reopenedUrl, "https://example.test/new.safetensors", "the edited import draft URL must survive dismiss/reopen");
  // Interact with a real rendered control over explicitly SYNTHETIC fixture terms.
  // No storage injection and no acceptance of actual upstream terms or download.
  const baseCard=cardFor(page,MINIMAX_REF.name);
  await baseCard.getByText('Synthetic fixture terms only.',{exact:false}).waitFor({state:'visible',timeout:LONG});
  await baseCard.getByRole('checkbox',{name:/I have read and accept this model/}).check();
  await form.getByRole('button',{name:'Repair selected model',exact:true}).waitFor({state:'visible',timeout:LONG});
  const repairedResponse=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/v1/models/my_h3/repair'&&response.request().method()==='POST');
  await form.getByRole('button',{name:'Repair selected model',exact:true}).click();
  const response=await repairedResponse;equal(response.status(),201,'existing-selection repair must queue through the actual endpoint');
  const repairJob=await response.json();equal(repairJob.payload.modelId,'my_h3','repair preserves the original selection even after name edit');
  const repair=await waitForRequest(recorder,'POST','/api/v1/models/my_h3/repair');equal(repair.body,MANAGED_REPAIR_REQUEST,'repair serializes exact actual managed DTO without internal routing flags');
  equal(matchingRequests(recorder,'POST','/api/v1/models/import'),[],'repair must not create another same-name model');
  equal(matchingRequests(recorder,'POST','/api/v1/models/my_h3/download'),[],'repair must not silently substitute download');
  await page.getByText('Import queued for my_h3.',{exact:true}).waitFor({state:'visible',timeout:LONG});
  const catalogResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/v1/models'&&r.request().method()==='GET');
  await page.reload();const catalog=await(await catalogResponse).json();
  equal(catalog.filter(model=>model.id==='my_h3').length,1,'exactly the original selection remains');equal(catalog.find(model=>model.id==='my_h3').name,'Edited name','repair updated the original selection rather than creating a new ID');
  await waitForApp(page);await modelsTab(page);await cardFor(page,'Edited name').waitFor({state:'visible',timeout:LONG});

  return {
    evidence: {
      metadata: { typeValue, familyValue, baseValue, partitionValue, seededName },
      draft: { name: reopenedName, url: reopenedUrl },
      repairRequest:repair.body, repairJob, stableSelectionId:'my_h3',
      repairPosts:matchingRequests(recorder,'POST','/api/v1/models/my_h3/repair').length,
      importPosts: 0,
      downloadPosts: 0,
    },
  };
}

// ---------------------------------------------------------------------------
// Case 3 — imported FL2VA/hybrid last-only conditioning serializes without
// inventing a first frame.
// ---------------------------------------------------------------------------
async function caseVideoLastOnly(page, recorder) {
  const evidence = [];
  for (const partition of ["fl2va", "hybrid"]) {
    const modelId = `imported_${partition}`;
    await page.goto("/");
    await waitForApp(page);
    await navTo(page, "Video");
    await page.locator(".video-studio").waitFor({ state: "visible", timeout: LONG });
    await page.getByRole("tab", { name: /Image → Video/ }).first().click();
    await page.locator(".settings-field-model select").selectOption(modelId);
    await pickAsset(page, "Last frame", "last frame");

    const render = page.locator(".video-studio button.prompt-cta").filter({ hasText: "Render clip" }).first();
    await render.waitFor({ state: "visible", timeout: LONG });
    assert(!(await render.isDisabled()), `${modelId}: Render clip must be enabled for last-only conditioning`);

    const before = matchingRequests(recorder, "POST", "/api/v1/video/jobs").length;
    await render.click();
    const request = await waitFor(() => matchingRequests(recorder, "POST", "/api/v1/video/jobs")[before] ?? null, {
      label: `${modelId} video job POST`,
    });
    equal(request.body.model, modelId, `${partition}: selected imported model must be serialized`);
    equal(request.body.mode, "image_to_video", `${partition}: mode must be image_to_video`);
    equal(request.body.sourceAssetId, null, `${partition}: no first frame may be invented`);
    equal(request.body.lastFrameAssetId, "last", `${partition}: the picked last frame must be serialized`);
    evidence.push({
      partition,
      request: {
        model: request.body.model,
        mode: request.body.mode,
        sourceAssetId: request.body.sourceAssetId,
        lastFrameAssetId: request.body.lastFrameAssetId,
      },
    });
  }
  return { evidence };
}

// ---------------------------------------------------------------------------
// Case 4 — an explicit empty replayed reference list clears the current refs.
// ---------------------------------------------------------------------------
async function caseImageEmptyRefs(page, recorder) {
  await page.goto("/");
  await waitForApp(page);
  await navTo(page, "Image");
  await page.locator(".image-studio").waitFor({ state: "visible", timeout: LONG });
  await page.getByRole("tab", { name: /^Edit$/ }).first().click();
  await page.locator(".settings-field-model select").selectOption("flux2_dev");
  await pickAsset(page, "Reference images", "stale reference");

  const picker = page
    .locator(".asset-picker-field")
    .filter({ has: page.locator(".asset-picker-label", { hasText: "Reference images" }) })
    .first();
  await picker.getByText("stale reference").first().waitFor({ state: "visible", timeout: LONG });

  await replayAsset(page, "empty refs replay");
  await page.locator(".image-studio").waitFor({ state: "visible", timeout: LONG });

  const generate = page.locator(".image-studio button.prompt-cta").filter({ hasText: "Generate" }).first();
  await generate.waitFor({ state: "visible", timeout: LONG });
  assert(await generate.isDisabled(), "after an explicit empty replay the Generate button must require a reference");
  await page
    .locator(".image-studio")
    .getByText("No reference images selected", { exact: true })
    .first()
    .waitFor({ state: "visible", timeout: LONG });

  const imagePosts = matchingRequests(recorder, "POST", "/api/v1/image/jobs");
  equal(imagePosts, [], "a cleared reference set must never submit an image job");

  return {
    evidence: {
      generateDisabled: true,
      visibleEmptyReferenceLabel: "No reference images selected",
      imageJobPosts: imagePosts.length,
    },
  };
}

// ---------------------------------------------------------------------------
// Case 5 — video replay carries the imported checkpoint identity.
// ---------------------------------------------------------------------------
async function caseVideoReplayExpectedRef(page, recorder) {
  await page.goto("/");
  await waitForApp(page);
  await replayAsset(page, "cyclist");
  await page.locator(".video-studio").waitFor({ state: "visible", timeout: LONG });

  const render = page.locator(".video-studio button.prompt-cta").filter({ hasText: "Render clip" }).first();
  await render.waitFor({ state: "visible", timeout: LONG });
  assert(!(await render.isDisabled()), "replayed imported clip must be submit-ready");
  await render.click();

  const request = await waitForRequest(recorder, "POST", "/api/v1/video/jobs");
  equal(request.body.model, "imported_hybrid", "replay must select the recorded imported model");
  equal(request.body.sourceAssetId, null, "replay must keep last-only conditioning");
  equal(request.body.lastFrameAssetId, "last", "replay must carry the recorded last frame");
  equal(request.body.expectedImportedModelRef, IMPORTED_MODEL_REF, "replay must carry the original imported checkpoint identity");

  const stored = await page.evaluate(() => {
    try {
      return JSON.parse(window.localStorage.getItem("sceneworks-studio-video-project-1") ?? "null");
    } catch {
      return null;
    }
  });

  return {
    evidence: {
      request: {
        model: request.body.model,
        sourceAssetId: request.body.sourceAssetId,
        lastFrameAssetId: request.body.lastFrameAssetId,
        expectedImportedModelRef: request.body.expectedImportedModelRef,
      },
      storedExpectedImportedModelRef: stored?.expectedImportedModelRef ?? null,
    },
  };
}

export const CASES = [
  { id: "model-manager-repair", run: caseModelManagerRepair },
  { id: "model-manager-reimport", run: caseModelManagerReimport },
  { id: "video-last-only", run: caseVideoLastOnly },
  { id: "image-empty-refs", run: caseImageEmptyRefs },
  { id: "video-replay-expected-ref", run: caseVideoReplayExpectedRef },
];
