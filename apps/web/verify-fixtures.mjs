// VERIFICATION-ONLY, UNTRACKED FIXTURE HARNESS (sc-26687 S8 browser proof).
//
// Serves controlled API responses for the real apps/web frontend. Loaded only by
// vite.verify.config.mjs. It never proxies or forwards to a real backend: every
// /api request is answered here and every unknown path is a 404.
//
// Models are built from the SHIPPED manifest (config/manifests/builtin.models.jsonc)
// exactly as the authoritative frontend fixtures do, so the studio sees the real
// limits/declarations rather than a retyped copy.
import { readFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import JSON5 from "json5";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../config/manifests/builtin.models.jsonc");

function loadManifest() {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
}

export function shippedModels() {
  const models = loadManifest();
  const byId = new Map(models.map((model) => [model.id, model]));
  const shipped = (id) => {
    const entry = byId.get(id);
    if (!entry) throw new Error(`verification fixture: manifest has no model ${id}`);
    return { ...entry, installState: "installed", usable: true };
  };
  return { shipped };
}

const { shipped } = shippedModels();
export const MINIMAX = shipped("minimax_h3");
// Closed-fixture authorization only: real upstream terms are NOT accepted by this test.
// The actual rendered consent control must be clicked before the mock repair submits.
export const MINIMAX_REF = { ...shipped("minimax_h3_ref"), installState:"missing", usable:false, downloadable:true, gated:false, requiresLicenseAcknowledgment:true, licenseName:"SYNTHETIC TEST LICENSE — not real approval", licenseUrl:"https://example.test/synthetic-license", licenseNotice:"Synthetic fixture terms only. This checkbox is not acceptance of any real model license." };
export const MANAGED_REPAIR_REQUEST = {ownershipMode:"managed",type:"video",name:"Edited name",family:"minimax-h3",baseModel:"minimax_h3_ref",minimaxH3Partition:"ref2va",source:{kind:"url",url:"https://example.test/new.safetensors"},modelId:"my_h3",licenseAcknowledged:true};

// The repair card fixture is the authoritative unit fixture (ModelManagerScreen.importedRepair):
// a synthetic user import with no builtin licence fields, so the repair button is the ONLY
// download path on offer and the test proves the componentId binding.
export const REPAIR_MODEL = {
  id: "my_h3",
  name: "My H3",
  type: "video",
  family: "minimax-h3",
  catalogScope: "user",
  installState: "missing",
  downloadable: false,
  importPlan: { checkpointId: "managed/h3" },
  importedSource: {
    family: "minimax-h3",
    ownershipMode: "managed",
    sourceVariant: "ref2va",
    declaredPartition: "ref2va",
    baseModel: "minimax_h3_ref",
    componentTier: "q4",
  },
  source: { provider: "imported-checkpoint" },
  componentReadiness: {
    ready: false,
    components: [
      { componentId: "transformer_ref", state: "missing" },
      { componentId: "audio_vae", state: "missing" },
    ],
  },
  repairActions: [
    { id: "audio", kind: "download", componentId: "audio_vae", label: "Repair audio" },
    { id: "source", kind: "reimport", label: "Reimport transformer" },
  ],
};

// Imported generation models: the SHIPPED H3 entry plus exactly the fields the import route adds.
// `supportsLastOnly` in VideoStudio accepts family minimax-h3 + baseModel minimax_h3 +
// transformer_directory + a plan checkpointId + partition fl2va|hybrid, so both partitions are
// exercised with the real catalog limits.
export function importedModel(partition) {
  return {
    ...MINIMAX,
    id: `imported_${partition}`,
    name: `Imported H3 ${partition}`,
    family: "minimax-h3",
    baseModel: "minimax_h3",
    importSourceShape: "transformer_directory",
    importPlan: {
      checkpointId: `checkpoint_${partition}`,
      planId: `plan_${partition}`,
      assembly: { schemaVersion: 1, partition },
    },
    minimaxH3Partition: partition,
    installState: "installed",
    usable: true,
  };
}

// The multi-reference image model used by the explicit-empty-reference replay.
export const IMAGE_MODEL = {
  id: "flux2_dev",
  name: "FLUX.2 Dev",
  type: "image",
  family: "flux2-dev",
  capabilities: ["text_to_image", "edit_image", "character_image"],
  installState: "installed",
  usable: true,
  defaults: { resolution: "1024x1024" },
  limits: { resolutions: ["1024x1024"] },
  ui: { multiReference: true },
};

export const MODELS = [
  REPAIR_MODEL,
  importedModel("fl2va"),
  importedModel("hybrid"),
  MINIMAX,
  MINIMAX_REF,
  IMAGE_MODEL,
];

const imageAsset = (id, displayName) => ({
  id,
  projectId: "project-1",
  type: "image",
  displayName,
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
  file: { path: `assets/images/${id}.png`, mimeType: "image/png" },
});

export const STALE_REFERENCE_ASSET = imageAsset("stale", "stale reference");
export const LAST_FRAME_ASSET = imageAsset("last", "last frame");
export const OTHER_IMAGE_ASSET = imageAsset("other", "other image");

export const IMPORTED_MODEL_REF = {
  modelId: "imported_hybrid",
  checkpointId: "original",
  planId: "plan_original",
  declaredPartition: "hybrid",
  baseModel: "minimax_h3",
  family: "minimax-h3",
};

export const VIDEO_REPLAY_ASSET = {
  id: "asset-video-replay",
  projectId: "project-1",
  generationSetId: "genset-video-replay",
  type: "video",
  displayName: "cyclist",
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
  file: { path: "assets/videos/cyclist.mp4", mimeType: "video/mp4" },
  generationSet: {
    recipe: {
      mode: "image_to_video",
      model: "imported_hybrid",
      prompt: "A cyclist rides",
      importedModelRef: IMPORTED_MODEL_REF,
      normalizedSettings: {
        sourceAssetId: null,
        lastFrameAssetId: "last",
        width: 768,
        height: 768,
        duration: 5.1667,
      },
      rawAdapterSettings: {},
    },
  },
};

export const IMAGE_EMPTY_REFS_ASSET = {
  id: "asset-image-empty-refs",
  projectId: "project-1",
  generationSetId: "genset-image-empty-refs",
  type: "image",
  displayName: "empty refs replay",
  status: { favorite: false, rating: 0, rejected: false, trashed: false },
  file: { path: "assets/images/asset-image-empty-refs.png", mimeType: "image/png" },
  generationSet: {
    recipe: {
      mode: "edit_image",
      model: "flux2_dev",
      prompt: "a lighthouse in heavy fog",
      negativePrompt: "",
      loras: [],
      normalizedSettings: { referenceAssetIds: [], width: 1024, height: 1024, count: 1 },
      rawAdapterSettings: { referenceAssetIds: ["stale"] },
    },
  },
};

export const ASSETS = [
  STALE_REFERENCE_ASSET,
  LAST_FRAME_ASSET,
  OTHER_IMAGE_ASSET,
  VIDEO_REPLAY_ASSET,
  IMAGE_EMPTY_REFS_ASSET,
];

let state;
function reset() {
  state = { requests: [], jobs: [], eventSourceOpens: 0, repairedModel:null };
}
reset();

function readBody(req) {
  return new Promise((resolveBody) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => resolveBody(data));
  });
}

export function verifyMiddleware() {
  return async function (req, res, next) {
    if (!req.url || !req.url.startsWith("/api/")) return next();
    const url = new URL(req.url, "http://127.0.0.1");
    const path = url.pathname;
    const method = req.method ?? "GET";
    const raw = await readBody(req);
    let body = null;
    if (raw) {
      try {
        body = JSON.parse(raw);
      } catch {
        body = raw;
      }
    }
    if (path === "/api/__verify/requests") {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(state));
      return;
    }
    if (path === "/api/__verify/reset" && method === "POST") {
      reset();
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    state.requests.push({ method, path, url: req.url, body });
    const json = (status, payload) => {
      res.statusCode = status;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify(payload));
    };

    if (path === "/api/v1/jobs/events") {
      state.eventSourceOpens += 1;
      res.statusCode = 200;
      res.setHeader("content-type", "text/event-stream");
      res.setHeader("cache-control", "no-cache");
      res.setHeader("connection", "keep-alive");
      res.write(": connected\n\n");
      const timer = setInterval(() => res.write(": ping\n\n"), 15000);
      req.on("close", () => clearInterval(timer));
      return;
    }
    if (path === "/api/v1/health") return json(200, { status: "ok", authRequired: false });
    if (path === "/api/v1/access") return json(200, { authRequired: false });
    if (path === "/api/v1/jobs/events/ticket") return json(200, { ticket: "stream-ticket" });
    if (path === "/api/v1/files/ticket") return json(200, { ticket: "media-ticket" });
    if (path === "/api/v1/ui-preferences") {
      if (method === "PUT") return json(200, body ?? {});
      return json(200, {
        activeView: "Models",
        embedWorkflowInImages: true,
        workflowEmbedNoticeSeen: true,
        defaultGenerationQuality: "q8",
        theme: "light",
      });
    }
    if (path === "/api/v1/projects") return json(200, [{ id: "project-1", name: "Noir" }]);
    if (/^\/api\/v1\/projects\/[^/]+\/assets$/.test(path)) return json(200, ASSETS);
    if (path === "/api/v1/models") return json(200, MODELS.map(model=>model.id==='my_h3'&&state.repairedModel?state.repairedModel:model));
    if (path === "/api/v1/loras") return json(200, []);
    if (path === "/api/v1/jobs") return json(200, state.jobs);
    if (path === "/api/v1/training/targets" || path === "/api/v1/training/presets") return json(200, []);
    if (path === "/api/v1/library-roots") return json(200, { roots: [{ rootId: "r", path: "/models" }] });
    if (path === "/api/v1/model-library") {
      return json(200, {
        schemaVersion: 1,
        configuredLibraryPath: "/models",
        expectedLibrary: null,
        probeStatus: "available",
        available: true,
      });
    }
    // Known empty collections the shell reads; anything else under /api is a 404.
    const EMPTY_COLLECTIONS = [
      "/api/v1/characters",
      "/api/v1/characters/looks",
      "/api/v1/presets",
      "/api/v1/poses",
      "/api/v1/keypoints",
      "/api/v1/voices",
      "/api/v1/person-tracks",
      "/api/v1/timelines",
      "/api/v1/credentials",
      "/api/v1/cache",
      "/api/v1/model-cache",
    ];
    if (method === "GET" && EMPTY_COLLECTIONS.includes(path)) return json(200, []);

    const download = path.match(/^\/api\/v1\/models\/([^/]+)\/download$/);
    if (download && method === "POST") {
      const job = {
        id: "repair-job",
        createdAt: "2026-10-25T00:00:00Z",
        type: "model_download",
        status: "queued",
        payload: { modelId: download[1], ...(body ?? {}) },
      };
      state.jobs.push(job);
      return json(200, job);
    }
    if(path==='/api/v1/models/my_h3/repair'&&method==='POST'){
      if(!isDeepStrictEqual(body,MANAGED_REPAIR_REQUEST))return json(422,{detail:'synthetic repair requires the exact actual DTO and original modelId'});
      state.repairedModel={...REPAIR_MODEL,name:body.name,installState:'installed',usable:true,repairActions:[]};
      const job={id:'managed-repair-job',type:'model_import',status:'queued',payload:{...body},result:{modelId:'my_h3'}};state.jobs.push(job);return json(201,job);
    }
    if (path === "/api/v1/models/import" && method === "POST") {
      const job = { id: "import-job", type: "model_import", status: "queued", payload: { ...(body ?? {}) } };
      state.jobs.push(job);
      return json(200, job);
    }
    if (path === "/api/v1/image/jobs" && method === "POST") {
      const job = {
        id: `image-job-${state.jobs.length + 1}`,
        type: "image_generate",
        status: "queued",
        createdAt: "2026-10-25T00:00:00Z",
        payload: body ?? {},
      };
      state.jobs.push(job);
      return json(200, job);
    }
    if (path === "/api/v1/video/jobs" && method === "POST") {
      const job = {
        id: `video-job-${state.jobs.length + 1}`,
        type: "video_generate",
        status: "queued",
        createdAt: "2026-10-25T00:00:00Z",
        payload: body ?? {},
      };
      state.jobs.push(job);
      return json(200, job);
    }
    return json(404, { detail: `verification mock: unhandled ${method} ${path}` });
  };
}
