import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  assertBackendCoverage,
  BESPOKE_ROUTE_ALIASES,
  catalogToWebPreviewSupport,
  derivePreviewSupport,
  indexSgaExport,
  parseEngineModelTable,
  PREVIEW_SUPPORT_GENERATOR,
  REQUIRED_AUDIO_BACKENDS,
  REQUIRED_MEDIA_BACKENDS,
  REQUIRED_SGA_EXPORTS,
  SGA_CAPABILITIES_SCHEMA_VERSION,
} from "./previewSupportDerivation.js";
import previewSupport from "./previewSupport.json";
import { fallbackModels } from "../testSupport/builtinModels.js";

function readRepositoryFile(relativePath) {
  return readFileSync(resolve(process.cwd(), "../..", relativePath), "utf8");
}

const enginesSource = readRepositoryFile("crates/sceneworks-worker/src/engines.rs");
const previewSupportManifestRaw = readRepositoryFile(
  "config/manifests/builtin.preview-support.jsonc",
);

const sgaEntries = REQUIRED_SGA_EXPORTS.map((spec) => ({
  ...spec,
  doc: JSON.parse(readRepositoryFile(`config/engine-capabilities/sga/${spec.file}`)),
}));
const sgaDocs = sgaEntries.map((entry) => entry.doc);

const rows = parseEngineModelTable(enginesSource);
const allMediaRows = [...rows, ...BESPOKE_ROUTE_ALIASES];
const derived = derivePreviewSupport(rows, sgaDocs);

const audioRouteIds = new Set(
  sgaDocs.flatMap((doc) => Object.keys(doc.audio_routes ?? {})),
);

describe("preview-support catalog: derived purely from sga.capabilities/2 exports", () => {
  it("re-deriving from engines.rs + sga.capabilities/2 reproduces builtin.preview-support.jsonc", () => {
    expect(JSON.parse(previewSupportManifestRaw)).toEqual(derived);
  });

  it("re-deriving reproduces the web fallback table previewSupport.json", () => {
    expect(previewSupport).toEqual(catalogToWebPreviewSupport(derived));
  });

  it("the two artifacts agree on every model (one derivation, two writes)", () => {
    const manifest = JSON.parse(previewSupportManifestRaw);
    expect(previewSupport.models).toEqual(manifest.models);
    expect(previewSupport.backends).toEqual(manifest.backends);
    expect(previewSupport.version).toBe(manifest.version);
  });

  it("names the generator and schema version without pin/SHA churn (M8)", () => {
    const manifest = JSON.parse(previewSupportManifestRaw);
    expect(manifest.generatedBy).toBe(PREVIEW_SUPPORT_GENERATOR);
    expect(manifest.generatedFrom).toEqual({
      schemaVersion: SGA_CAPABILITIES_SCHEMA_VERSION,
      featureSets: { candle: "candle-cuda", mlx: "macos-mlx" },
    });
  });

  it("covers every declared desktop/server media and audio backend", () => {
    expect(() =>
      assertBackendCoverage(
        REQUIRED_MEDIA_BACKENDS,
        sgaDocs.map((doc) => doc.backend),
        "media",
      ),
    ).not.toThrow();
    expect(() =>
      assertBackendCoverage(
        REQUIRED_AUDIO_BACKENDS,
        [...new Set(sgaDocs.map((doc) => doc.audio_backend).filter(Boolean))],
        "audio",
      ),
    ).not.toThrow();
    expect(JSON.parse(previewSupportManifestRaw).backends).toEqual(["candle", "mlx"]);
    expect(previewSupport.backends).toEqual(["candle", "mlx"]);
  });
});

describe("preview-support catalog: promoted routes read directly from sga.capabilities/2", () => {
  it("derives Candle Qwen-Image-Edit preview from registered Candle routes without source scanners", () => {
    for (const modelId of [
      "qwen_image_edit",
      "qwen_image_edit_2509",
      "qwen_image_edit_2511",
      "qwen_image_edit_2511_lightning",
    ]) {
      expect(previewSupport.models[modelId]).toEqual({ candle: true, mlx: true });
    }
  });

  it("derives PuLID and InstantID preview directly from registered pulid_flux and instantid routes", () => {
    expect(previewSupport.models.pulid_flux_dev).toEqual({ candle: true, mlx: true });
    expect(previewSupport.models.instantid_realvisxl).toEqual({ candle: false, mlx: false });
  });
});

describe("preview-support catalog: sga.capabilities/2 exports are non-vacuous and schema-checked", () => {
  it.each(sgaEntries.map((entry) => [entry.file, entry]))(
    "%s is a valid non-vacuous sga.capabilities/2 export",
    (_file, entry) => {
      expect(entry.doc.schema_version).toBe(SGA_CAPABILITIES_SCHEMA_VERSION);
      expect(entry.doc.backend).toBe(entry.backend);
      expect(entry.doc.feature_set).toBe(entry.featureSet);
      expect(Object.keys(entry.doc.routes).length).toBeGreaterThan(40);
      expect(Object.keys(entry.doc.audio_routes).length).toBeGreaterThan(10);
      expect(entry.doc.generatedFrom).toBeUndefined();
    },
  );

  it("refuses empty routes or wrong schema version", () => {
    expect(() =>
      indexSgaExport(
        { schema_version: "sga.capabilities/1", backend: "candle", feature_set: "candle-cuda", routes: {} },
        "bad.json",
      ),
    ).toThrow(/unsupported schema_version/);
    expect(() =>
      derivePreviewSupport(rows, [
        { schema_version: SGA_CAPABILITIES_SCHEMA_VERSION, backend: "candle", feature_set: "candle-cuda", routes: {} },
      ]),
    ).toThrow(/`routes` is empty/);
    expect(() => derivePreviewSupport(rows, [])).toThrow(/no sga\.capabilities\/2 exports/);
  });

  it("refuses missing or undeclared backends loudly", () => {
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle"])).toThrow(/mlx/);
    expect(() => assertBackendCoverage(["candle", "mlx"], ["mlx"])).toThrow(/candle/);
    expect(() => assertBackendCoverage(["candle"], ["candle", "mlx"])).toThrow(/mlx/);
    expect(() => assertBackendCoverage([], ["candle", "mlx"])).toThrow(/empty/);
    expect(() => assertBackendCoverage(["candle", "mlx"], ["candle", ""])).toThrow(
      /no readable `backend`/,
    );
  });
});

describe("preview-support catalog: audio routes from sga.capabilities/2", () => {
  it("answers false — not unknown — for every shipped audio route", () => {
    for (const shipped of [
      "kokoro_82m",
      "chatterbox_tts",
      "acestep_v15_turbo",
      "moss_sfx_v2",
      "moss_tts_realtime",
      "moss_ttsd_v05",
    ]) {
      expect(audioRouteIds, `${shipped} is not in audio_routes`).toContain(shipped);
      expect(previewSupport.models[shipped], shipped).toEqual({ candle: false });
    }
  });

  it("no audio route advertises live preview at this pin", () => {
    const advertising = [...audioRouteIds].filter((id) =>
      Object.values(previewSupport.models[id] ?? {}).some((supported) => supported === true),
    );
    expect(advertising).toEqual([]);
  });

  it("leaves non-generator audio providers unknown rather than inventing false", () => {
    for (const nonGenerator of ["openvoice_v2", "chatterbox_ve"]) {
      expect(audioRouteIds, nonGenerator).not.toContain(nonGenerator);
      expect(previewSupport.models[nonGenerator], nonGenerator).toBeUndefined();
    }
  });

  it("refuses a media/audio namespace collision", () => {
    const collidingId = rows[0].sceneworksId;
    const mutated = structuredClone(sgaDocs);
    mutated[0].audio_routes[collidingId] = {
      capabilities: { supports_preview: false },
    };
    expect(() => derivePreviewSupport(rows, mutated)).toThrow(
      /both a media model id and an audio route id/,
    );
  });
});

describe("preview-support catalog: the MODEL_TABLE join", () => {
  it("parses the full engines.rs model table", () => {
    expect(rows.length).toBeGreaterThan(40);
    expect(rows.every((row) => row.sceneworksId && row.engineId)).toBe(true);
  });

  it("is many-to-one: distinct model ids may share one engine id", () => {
    const shared = rows.filter((row) => row.engineId === "z_image_turbo").map((r) => r.sceneworksId);
    expect(shared).toEqual(expect.arrayContaining(["z_image_turbo", "z_image_edit"]));
    expect(previewSupport.models.z_image_turbo).toEqual(previewSupport.models.z_image_edit);
  });

  it("every emitted entry is backed by a real sga.capabilities/2 route", () => {
    for (const { backend, doc } of sgaEntries) {
      const routeIds = new Set(Object.keys(doc.routes));
      const audioIds = new Set(
        doc.audio_backend === backend ? Object.keys(doc.audio_routes ?? {}) : [],
      );
      for (const [modelId, byBackend] of Object.entries(previewSupport.models)) {
        if (!(backend in byBackend)) continue;
        if (audioIds.has(modelId)) continue;
        const row = allMediaRows.find((candidate) => candidate.sceneworksId === modelId);
        expect(
          routeIds.has(modelId) || (row && routeIds.has(row.engineId)),
          `${modelId} @ ${backend}`,
        ).toBe(true);
      }
    }
  });

  it("throws rather than silently emptying when MODEL_TABLE cannot be found", () => {
    expect(() => parseEngineModelTable("// no table here\n")).toThrow(/MODEL_TABLE/);
  });

  it("refuses a row it cannot read rather than dropping it", () => {
    const withNestedField = enginesSource.replace(
      'sceneworks_id: "sdxl",',
      'limits: Limits { max: 4 },\n        sceneworks_id: "sdxl",',
    );
    expect(withNestedField).not.toBe(enginesSource);
    expect(() => parseEngineModelTable(withNestedField)).toThrow(
      /declares \d+ ModelRow entries but only \d+ parsed/,
    );
  });

  it("does not parse a commented-out row as a live one", () => {
    const withGhost = enginesSource.replace(
      "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n",
      "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n" +
        "    /*\n" +
        "    ModelRow {\n" +
        '        sceneworks_id: "ghost_model",\n' +
        '        engine_id: "ghost_engine",\n' +
        '        default_repo: "x/y",\n' +
        "        default_steps: 1,\n" +
        "        default_guidance: 1.0,\n" +
        '        adapter_label: "ghost",\n' +
        "    },\n" +
        "    */\n",
    );
    expect(withGhost).not.toBe(enginesSource);
    const ghosted = parseEngineModelTable(withGhost);
    expect(ghosted.map((row) => row.sceneworksId)).not.toContain("ghost_model");
    expect(ghosted.length).toBe(rows.length);
  });

  it("is not truncated by a `];` that only appears inside a comment", () => {
    for (const injected of [
      "    // the table ends with\n    // ];\n",
      "    /* like this:\n];\n    */\n",
    ]) {
      const withStray = enginesSource.replace(
        "pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n",
        `pub(crate) const MODEL_TABLE: &[ModelRow] = &[\n${injected}`,
      );
      expect(withStray).not.toBe(enginesSource);
      expect(parseEngineModelTable(withStray).length, injected).toBe(rows.length);
    }
  });
});

describe("preview-support catalog: the shipped answers match sga.capabilities/2", () => {
  it.each(sgaEntries.map((entry) => [entry.backend, entry.doc]))(
    "advertises live preview on %s for exactly the sga.capabilities/2 routes that say so",
    (backend, doc) => {
      const expected = allMediaRows
        .filter((row) => {
          const route = doc.routes[row.sceneworksId] ?? doc.routes[row.engineId];
          return route?.capabilities?.supports_preview === true;
        })
        .map((row) => row.sceneworksId)
        .concat(
          doc.audio_backend === backend
            ? Object.entries(doc.audio_routes ?? {})
                .filter(([, route]) => route?.capabilities?.supports_preview === true)
                .map(([id]) => id)
            : [],
        )
        .sort();
      const actual = Object.entries(previewSupport.models)
        .filter(([, byBackend]) => byBackend[backend] === true)
        .map(([id]) => id)
        .sort();
      expect(actual).toEqual(expected);
      expect(actual.length).toBeGreaterThan(0);
    },
  );

  it.each(sgaEntries.map((entry) => [entry.backend, entry.doc]))(
    "says false — not unknown — for every %s route that is wired and does not preview",
    (backend, doc) => {
      const wiredFalse = allMediaRows.filter((row) => {
        const route = doc.routes[row.sceneworksId] ?? doc.routes[row.engineId];
        return route?.capabilities?.supports_preview === false;
      });
      expect(wiredFalse.length).toBeGreaterThan(0);
      for (const row of wiredFalse) {
        expect(
          previewSupport.models[row.sceneworksId]?.[backend],
          `${row.sceneworksId} -> ${row.engineId} @ ${backend}`,
        ).toBe(false);
      }
    },
  );

  it("is engine-KEYED: every entry is a per-backend map, never a bare boolean", () => {
    for (const [modelId, byBackend] of Object.entries(previewSupport.models)) {
      expect(typeof byBackend, modelId).toBe("object");
      for (const [backend, supported] of Object.entries(byBackend)) {
        expect(previewSupport.backends, `${modelId}.${backend}`).toContain(backend);
        expect(typeof supported, `${modelId}.${backend}`).toBe("boolean");
      }
    }
  });
});

describe("preview-support catalog: the offline fallback mirrors the served flag", () => {
  it("stamps preview.byBackend onto every fallback model the table knows", () => {
    for (const model of fallbackModels) {
      const expected = previewSupport.models[model.id];
      if (!expected) {
        expect(model.preview, `${model.id} is not in the generated table`).toBeUndefined();
        continue;
      }
      expect(model.preview?.byBackend, model.id).toEqual(expected);
    }
  });

  it("seeds BOTH answers, so the offline path can exercise every state", () => {
    const seeded = fallbackModels.filter((model) => previewSupport.models[model.id]);
    expect(seeded.length).toBeGreaterThan(0);
    const answers = seeded.flatMap((model) => Object.values(model.preview?.byBackend ?? {}));
    expect(answers, "no fallback model advertises live preview").toContain(true);
    expect(answers, "no fallback model reports a wired non-previewing route").toContain(false);
  });
});
