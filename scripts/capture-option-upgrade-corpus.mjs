// Capture real released builder outputs. Requires git and Node; never loads an inference engine.
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const revision = "d3f7b05bf22826e44fd43da19cff0985fb17d68b";
let server;
const temp = mkdtempSync(join(tmpdir(), "sceneworks-historical-builders-"));
try {
  const archive = execFileSync("git", ["archive", revision, "apps/web/src", "apps/web/package.json", "config/manifests/builtin.appearance.json", "crates/sceneworks-core/src/yue2_score/fixtures/web-job-requests.json", "tests/fixtures/rust_migration_contracts", "config/film-harness/courier-workshop/plan.jsonc"], { maxBuffer:128*1024*1024 });
  execFileSync("tar", ["-x", "-C", temp], { input: archive });
  symlinkSync(resolve("apps/web/node_modules"), join(temp,"apps/web/node_modules"));
  const { createServer } = await import(pathToFileURL(resolve("apps/web/node_modules/vite/dist/node/index.js")).href);
  server = await createServer({root:join(temp,"apps/web"),configFile:false,server:{middlewareMode:true}, appType:"custom"});
  const load = (path) => server.ssrLoadModule(`/src/${path}`);
  const { buildSimpleImageRequest, buildSimpleVideoRequest, buildSimpleAudioRequest } = await load("simple/simpleJobs.js");
  const { trainingConfigSnapshot, configDraftFromTarget } = await load("training/trainingConfig.js");
  const { buildImageJobRequest } = await load("imageJobRequest.js");
  const { imageRecipeRequest } = await load("regenerateImage.js");
  const inputs = { prompt:"A fox in snow", mode:"text_to_image", model:"z_image_turbo", resolution:"1024x1024", count:1, styleId:null, seed:"0", steps:"9", guidance:"0" };
  const advancedSettings = { ...inputs, sampler:"euler", scheduler:"simple", schedulerShift:3, schedulerShiftExplicit:true };
  const image = buildSimpleImageRequest(inputs);
  const advanced = buildImageJobRequest({ ...advancedSettings, promptToSend:inputs.prompt, width:1024,height:1024,loras:[],referenceAssetIds:[],posePayload:[], submitIntent:"", sendStructured:false });
  const recipe = { mode:"text_to_image",model:inputs.model,prompt:inputs.prompt,seed:0,normalizedSettings:{resolution:"1024x1024",count:1},rawAdapterSettings:image.advanced };
  const target = JSON.parse(readFileSync(join(temp,"tests/fixtures/rust_migration_contracts/training/target-registry.json"),"utf8")).targets[0];
  const dataset = JSON.parse(readFileSync(join(temp,"tests/fixtures/rust_migration_contracts/training/dataset.json"),"utf8"));
  const draft = configDraftFromTarget(target, dataset, ["auto"], "fox");
  const rows = [
    { name:"simple-image", carrier:"request", value:image },
    { name:"advanced-image", carrier:"request", value:advanced },
    { name:"simple-video", carrier:"request", value:buildSimpleVideoRequest({prompt:"A fox walks",mode:"text_to_video",model:"ltx_2_3",resolution:"768x512",duration:5,loras:[]}) },
    { name:"simple-audio", carrier:"request", value:buildSimpleAudioRequest({model:"kokoro",prompt:"hello",mode:"speech",voice:"af_heart"}) },
    { name:"recipe-regeneration", carrier:"request", value:imageRecipeRequest({type:"image",recipe}) },
    { name:"training-builder", carrier:"request", value:trainingConfigSnapshot({activeDataset:dataset,configDraft:draft,selectedTarget:target,dryRun:true}) },
  ];
  // These settings snapshots are produced by the historical persistence writers, not an invented envelope.
  const storage = new Map();
  globalThis.window = { localStorage:{ getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v) } };
  const { seedStudioSettingsFromServer, loadStudioSettings } = await load("hooks/useStudioSettings.js");
  seedStudioSettingsFromServer({historical:{image:advancedSettings}});
  rows.push({name:"advanced-settings-writer",carrier:"studioSettings",value:loadStudioSettings("image","historical")});
  const {seedSimpleStudiosFromServer,readSimpleStudioSnapshot}=await load("simpleStudioStore.js");
  seedSimpleStudiosFromServer({historical:{image:inputs}});
  rows.push({name:"simple-settings-writer",carrier:"simpleSettings",value:readSimpleStudioSnapshot("historical").image});
  const yue = JSON.parse(readFileSync(join(temp,"crates/sceneworks-core/src/yue2_score/fixtures/web-job-requests.json"),"utf8"));
  for (const [name, body] of Object.entries(yue.bodies)) rows.push({name:`historical-yue2-${name}`,carrier:"request",value:body});
  const protocol=JSON.parse(readFileSync(join(temp,"tests/fixtures/rust_migration_contracts/job_protocol.json"),"utf8"));
  rows.push({name:"historical-job-snapshot",carrier:"job",value:protocol.jobSnapshot});
  rows.push({name:"historical-http-create",carrier:"request",value:protocol.requests.createJob});
  rows.push({name:"historical-recipe-sidecar",carrier:"recipe",value:JSON.parse(readFileSync(join(temp,"tests/fixtures/rust_migration_contracts/sidecars/recipe.json"),"utf8"))});
  const { default: JSON5 } = await import(pathToFileURL(resolve("apps/web/node_modules/json5/lib/index.js")).href);
  rows.push({name:"historical-film-plan",carrier:"filmPlan",value:JSON5.parse(readFileSync(join(temp,"config/film-harness/courier-workshop/plan.jsonc"),"utf8"))});
  rows.push({name:"historical-training-plan",carrier:"trainingPlan",value:JSON.parse(readFileSync(join(temp,"tests/fixtures/rust_migration_contracts/training/training-plan.json"),"utf8"))});
  const destination = resolve("crates/sceneworks-options/tests/fixtures/historical-builders.json");
  mkdirSync(resolve("crates/sceneworks-options/tests/fixtures"),{recursive:true});
  writeFileSync(destination,JSON.stringify({sourceRevision:revision,capture:"node scripts/capture-option-upgrade-corpus.mjs",rows},null,2)+"\n");
  console.log(`Captured ${rows.length} historical outputs at ${revision}`);
} finally { await server?.close(); rmSync(temp,{recursive:true,force:true}); }
