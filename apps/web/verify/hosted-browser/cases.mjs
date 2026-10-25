// sc-26687 S8 browser cases. Every case drives the REAL production frontend through
// its rendered controls and asserts the request the app actually serialized. Nothing
// here calls a component helper, an oracle, or injects UI state.
import { IMPORTED_MODEL_REF } from "../../verify-fixtures.mjs";

const SHORT = 5000;
const LONG = 15000;

export class CaseAssertionError extends Error {}

function assert(condition, message) {
  if (!condition) throw new CaseAssertionError(message);
}

function equal(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
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

async function pickAsset(page, fieldLabel, assetName) {
  const field = page
    .locator(".asset-picker-field")
    .filter({ has: page.locator(".asset-picker-label", { hasText: fieldLabel }) })
    .first();
  await field.waitFor({ state: "visible", timeout: LONG });
  await field.getByRole("button", { name: /Select (image|images)|Edit references/ }).first().click();
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
  assert((await toggle.getAttribute("aria-expanded")) === "true", "reimport must open the existing import panel");

  const form = page.locator('form[aria-label="Add to SceneWorks"]').first();
  await form.waitFor({ state: "visible", timeout: LONG });

  const ownership = page.locator('[role="radio"][aria-checked="true"]').first();
  assert(((await ownership.textContent()) ?? "").includes("Add to SceneWorks"), "managed ownership must stay selected");

  const typeValue = await form.getByLabel(/^Type$/).inputValue();
  const familyValue = await form.getByLabel(/^Family$/).inputValue();
  const baseValue = await form.getByLabel(/^Compatible base model$/).inputValue();
  const partitionValue = await form.getByLabel(/^MiniMax-H3 partition$/).inputValue();
  const seededName = await form.getByLabel(/^Name$/).inputValue();
  equal({ typeValue, familyValue, baseValue, partitionValue }, {
    typeValue: "video",
    familyValue: "minimax-h3",
    baseValue: "minimax_h3_ref",
    partitionValue: "ref2va",
  }, "reimport must preserve the imported model's declaration metadata");
  equal(seededName, "My H3", "reimport must seed the imported model's name");

  await form.getByLabel(/^Name$/).fill("Edited name");
  await form
    .locator('[aria-label="Where the checkpoint comes from"]')
    .getByRole("radio", { name: "URL", exact: true })
    .click();
  await form.getByLabel(/^Source URL$/).fill("https://example.test/new.safetensors");

  await toggle.click();
  assert((await toggle.getAttribute("aria-expanded")) === "false", "Hide options must collapse the panel");

  await new Promise((done) => setTimeout(done, SHORT));
  equal(matchingRequests(recorder, "POST", "/api/v1/models/import"), [], "opening/editing a repair must not auto-submit an import");
  equal(matchingRequests(recorder, "POST", "/api/v1/models/my_h3/download"), [], "reimport must not trigger a download");

  await card.getByRole("button", { name: "Reimport transformer", exact: true }).click();
  await toggle.waitFor({ state: "visible", timeout: LONG });
  assert((await toggle.getAttribute("aria-expanded")) === "true", "reopening the repair must show the panel again");
  const reopenedName = await form.getByLabel(/^Name$/).inputValue();
  const reopenedUrl = await form.getByLabel(/^Source URL$/).inputValue();
  equal(reopenedName, "Edited name", "the edited import draft name must survive dismiss/reopen");
  equal(reopenedUrl, "https://example.test/new.safetensors", "the edited import draft URL must survive dismiss/reopen");

  return {
    evidence: {
      metadata: { typeValue, familyValue, baseValue, partitionValue, seededName },
      draft: { name: reopenedName, url: reopenedUrl },
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
