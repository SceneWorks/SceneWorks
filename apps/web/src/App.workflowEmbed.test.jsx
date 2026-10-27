import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click } from "./testUtils/dom.js";
import { resetStartupTimingForTests } from "./startupTiming.js";
import { resetAccessTokenForTests } from "./accessToken.js";

// Capture the pre-submit hook from the one image-job creator shared by every image studio. That
// proves the request is held before its POST without coupling this suite to a particular form.
const hoisted = vi.hoisted(() => ({ beforeCreate: null, imageCreator:null, imageProject:null }));

vi.mock("./createJob.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    makeCreateJob(options) {
      if (options.definition === actual.CREATE_JOB_DEFINITIONS.image) {
        hoisted.beforeCreate = options.beforeCreate;
      }
      const creator=actual.makeCreateJob(options);
      if(options.definition===actual.CREATE_JOB_DEFINITIONS.image){hoisted.imageCreator=creator;hoisted.imageProject=options.project;}
      return creator;
    },
  };
});

vi.mock("react-konva",async()=>{const React=await import('react');const pass=name=>({children})=>React.createElement('div',{'data-konva':name},children);return {Stage:pass('stage'),Layer:pass('layer'),Image:()=>null,Rect:()=>null};});
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import JSON5 from 'json5';
import { AppContext } from './context/AppContext.js';
import { ImageStudio } from './screens/ImageStudio.jsx';
import { SimpleShell } from './simple/SimpleShell.jsx';
import { ImageEditor } from './screens/ImageEditor.jsx';
import { terminalBrowserDom } from './testUtils/terminalBrowserDom.js';
import { acknowledgeRecipeChoice, configure } from '../../../scripts/lib/moss-edit-terminal-controls.mjs';
const { App } = await import("./main.jsx");

const MODAL_TITLE = "Include the recipe in generated images?";

function fakeApi(storedPreferences, failurePlan = { workflowPuts: 0 }) {
  return vi.fn((url, init) => {
    const path = new URL(url,window.location.href).pathname;
    if(path.endsWith('/host-capabilities'))return Promise.resolve(response({memoryGb:128,memoryKind:'unified',platform:'macos'}));
    if(path.endsWith('/image/jobs')&&init?.method==='POST'&&failurePlan.imagePostError)return Promise.reject(new Error('owned CPU rejection'));
    if(path.endsWith('/image/jobs')&&init?.method==='POST')return Promise.resolve(response({id:'owned-test-job',type:'image_generate',status:'queued'}));
    if (path.endsWith("/health")) {
      return Promise.resolve(response({ status: "ok", authRequired: false }));
    }
    if (path.endsWith("/access")) {
      return Promise.resolve(response({ authRequired: false }));
    }
    if (path.endsWith("/jobs/events/ticket")) {
      return Promise.resolve(response({ ticket: "stream-ticket" }));
    }
    if (path.endsWith("/projects")) {
      return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
    }
    if (path.endsWith("/ui-preferences")) {
      if (init?.method === "PUT") {
        const patch = JSON.parse(init.body);
        if (
          Object.hasOwn(patch, "workflowEmbedNoticeSeen") &&
          failurePlan.workflowPuts > 0
        ) {
          failurePlan.workflowPuts -= 1;
          return Promise.reject(new Error("offline"));
        }
        return Promise.resolve(response({ ...storedPreferences, ...patch }));
      }
      if (failurePlan.preferencesGet) {
        return failurePlan.preferencesGet.then((preferences) => response(preferences));
      }
      return Promise.resolve(response(storedPreferences));
    }
    return Promise.resolve(response([]));
  });
}

function preferenceWrites() {
  return global.fetch.mock.calls
    .filter(
      ([url, init]) =>
        new URL(url,window.location.href).pathname.endsWith("/ui-preferences") && init?.method === "PUT",
    )
    .map(([, init]) => JSON.parse(init.body));
}

const workflowDecisionWrites = () =>
  preferenceWrites().filter(
    (patch) =>
      Object.hasOwn(patch, "embedWorkflowInImages") ||
      Object.hasOwn(patch, "workflowEmbedNoticeSeen"),
  );

const modalIsShowing = () => document.body.textContent.includes(MODAL_TITLE);
const buttonNamed = (name) =>
  [...document.body.querySelectorAll("button")].find(
    (button) => button.textContent.trim() === name,
  );

describe("App - the one-time embedded-workflow choice (sc-16390)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    resetAccessTokenForTests();
    window.innerWidth = 1440;
    hoisted.beforeCreate = null;
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    window.localStorage.clear();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function renderApp(storedPreferences = {}, failurePlan) {
    global.fetch = fakeApi(storedPreferences, failurePlan);
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  async function startGeneration() {
    expect(hoisted.beforeCreate, "App must provide the image pre-submit hook").toBeTypeOf(
      "function",
    );
    let gate;
    await act(async () => {
      gate = hoisted.beforeCreate({ prompt: "roses" });
    });
    await settle();
    return { gate };
  }

  it("does not interrupt the app before a generation is attempted", async () => {
    await renderApp();
    expect(modalIsShowing()).toBe(false);
  });

  it("holds the first generation until Ok atomically enables embedding and records the choice", async () => {
    await renderApp();
    const { gate } = await startGeneration();
    expect(modalIsShowing()).toBe(true);
    expect(workflowDecisionWrites()).toEqual([]);
    expect(
      [...document.body.querySelector("[role=dialog]").querySelectorAll("button")].map(
        (button) => button.textContent.trim(),
      ),
    ).toEqual(["Go to settings", "No thanks!", "Ok, got it"]);
    const dialog = document.body.querySelector("[role=dialog]");
    const description = document.getElementById(dialog.getAttribute("aria-describedby"));
    expect(description?.textContent).toContain("model, and LoRA details");

    await click(buttonNamed("Ok, got it"));
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
    expect(preferenceWrites()).toContainEqual({
      embedWorkflowInImages: true,
      workflowEmbedNoticeSeen: true,
    });
    const second = await startGeneration();
    await expect(second.gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
  });

  it("persists No thanks atomically and resumes the held generation with embedding off", async () => {
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("No thanks!"));
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(preferenceWrites()).toContainEqual({
      embedWorkflowInImages: false,
      workflowEmbedNoticeSeen: true,
    });
  });

  it("opens Sharing and cancels the held request without recording a decision", async () => {
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("Go to settings"));
    await settle();

    await expect(gate).resolves.toBe(false);
    expect(workflowDecisionWrites()).toEqual([]);
    expect(document.body.textContent).toContain("Include the recipe in generated images");
    expect(document.activeElement?.textContent).toBe("Sharing");

    const { gate: nextGate } = await startGeneration();
    expect(modalIsShowing()).toBe(true);
    await click(buttonNamed("No thanks!"));
    await expect(nextGate).resolves.toBe(true);
  });

  it("never asks again when the durable decision marker is already present", async () => {
    await renderApp({ workflowEmbedNoticeSeen: true });
    const { gate } = await startGeneration();
    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
  });

  it("waits for the durable preference GET before deciding whether to show the modal", async () => {
    let releasePreferences;
    const preferencesGet = new Promise((resolve) => {
      releasePreferences = resolve;
    });
    await renderApp({}, { preferencesGet, workflowPuts: 0 });

    const { gate } = await startGeneration();
    expect(modalIsShowing()).toBe(false);

    await act(async () => {
      releasePreferences({ embedWorkflowInImages: false, workflowEmbedNoticeSeen: true });
    });
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
    expect(workflowDecisionWrites()).toEqual([]);
  });

  it("asks an upgraded install that has no decision marker", async () => {
    await renderApp({ theme: "dark", accent: "coral", defaultGenerationQuality: "q8" });
    await startGeneration();
    expect(modalIsShowing()).toBe(true);
  });

  it("asks for an explicit decision even when the current setting is off", async () => {
    await renderApp({ embedWorkflowInImages: false });
    await startGeneration();
    expect(modalIsShowing()).toBe(true);
  });

  it("keeps the choice open after a persistence failure and retries the same action", async () => {
    const failurePlan = { workflowPuts: 3 };
    await renderApp({}, failurePlan);
    const { gate } = await startGeneration();

    await click(buttonNamed("No thanks!"));
    await new Promise((resolve) => setTimeout(resolve, 1300));
    await settle();

    expect(modalIsShowing()).toBe(true);
    expect(document.body.textContent).toContain("could not save your choice");
    let resolved = false;
    gate.then(() => {
      resolved = true;
    });
    await Promise.resolve();
    expect(resolved).toBe(false);

    await click(buttonNamed("No thanks!"));
    await settle();
    await expect(gate).resolves.toBe(true);
  });

  it("opens and focuses Sharing inside the Simple shell too", async () => {
    window.innerWidth = 390;
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("Go to settings"));
    await settle();
    // SimpleSettings arms its Sharing focus in a `setTimeout(0)` MACROTASK, and `settle()` only
    // drains microtasks and lazy-screen imports — whether the timer had fired by now depended on
    // how many event-loop turns the lazy imports happened to take, which the sc-17137 main sync
    // tightened into a visible race. Grant the timer its turn explicitly instead of relying on it.
    await act(async () => {
      await new Promise((resolve) => {
        setTimeout(resolve, 1);
      });
    });

    await expect(gate).resolves.toBe(false);
    expect(document.body.querySelector(".su-group-title")?.textContent).toBe("Appearance");
    expect(document.activeElement?.textContent).toBe("Sharing");
  });
  it.each([['full',false],['simple',false],['editor',false],['editor',true],['editor','cancel'],['editor','unmount'],['editor','gate-error']])('fresh App recipe choice gates actual %s form before one POST (error=%s)',async(kind,requestError)=>{
    await renderApp({}, {workflowPuts:0,imagePostError:requestError===true});expect(hoisted.imageProject?.id).toBe('project-default');
    const manifest=JSON5.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)),'../../../config/manifests/builtin.models.jsonc'),'utf8'));
    const model={...manifest.models.find(item=>item.id==='moss_image_base'),installState:'installed',usable:true,capabilities:['text_to_image','edit_image'],conditionalPurposes:{moss_image_edit:{installState:'installed'}}};
    const image={id:'ref-owned',type:'image',projectId:'project-default',displayName:'B.png',file:{path:'assets/B.png'},status:{}};
    const noop=()=>{},ctx={activeProject:hoisted.imageProject,assets:[image],characters:[],models:[model],imageModels:[model],jobs:[],gpuOptions:[],imageLocalJobs:[],videoLocalJobs:[],audioLocalJobs:[],latestImageAssets:[],recentImageAssets:[],visibleWorkers:[],loras:[],presets:[],token:'',requestedGpu:'auto',theme:'light',activeView:'Image',selectedAsset:null,
      createImageJob:hoisted.imageCreator,confirmWorkflowEmbeddingBeforeGeneration:requestError==='gate-error'?async()=>{throw Error('owned gate rejection');}:hoisted.beforeCreate,createPreset:noop,refinePrompt:noop,qwenRewritePrompt:noop,deleteAsset:noop,purgeAsset:vi.fn(async()=>{}),setPreviewAsset:noop,setActiveView:noop,setSelectedAssetId:noop,setRequestedGpu:noop,updateAssetStatus:noop,rememberLocalGenerationJob:noop,jobAction:noop,changeTheme:noop,createVideoJob:noop,createAudioJob:noop,createModelDownloadJob:noop,createLoraDownloadJob:noop,
      registerLeaveGuard:()=>noop,registerEditorScratchClaim:()=>noop,trackEditorScratchOp:vi.fn(),releaseEditorScratchOp:vi.fn(),clearEditorLaunch:noop,
      importAsset:vi.fn(async()=>({...image,id:'scratch-owned'})),editorLaunch:{id:'terminal-editor',assetId:image.id},studioLaunch:{id:'terminal-full',view:'Image',recipe:{model:model.id,mode:'edit_image',prompt:'initial',rawAdapterSettings:{referenceAssetIds:[image.id]}}}};
    if(kind==='editor'){
      vi.stubGlobal('Image',class {constructor(){this.naturalWidth=512;this.naturalHeight=512;}set src(value){queueMicrotask(()=>this.onload?.());}});
      vi.spyOn(URL,'createObjectURL').mockReturnValue('blob:cpu-editor');vi.spyOn(URL,'revokeObjectURL').mockImplementation(noop);
      vi.spyOn(window.HTMLCanvasElement.prototype,'getContext').mockReturnValue({drawImage:noop,save:noop,restore:noop,translate:noop,scale:noop,rotate:noop});
      vi.spyOn(window.HTMLCanvasElement.prototype,'toBlob').mockImplementation(callback=>callback(new Blob(['owned CPU fixture'],{type:'image/png'})));
      const previous=global.fetch;global.fetch=vi.fn((url,init)=>String(url).includes('/assets/')?Promise.resolve({ok:true,blob:async()=>new Blob(['owned CPU fixture'],{type:'image/png'})}):previous(url,init));
    }
    const surface=document.createElement('div');document.body.appendChild(surface);const surfaceRoot=createRoot(surface);let unmounted=false;
    try{
      await act(async()=>surfaceRoot.render(<AppContext.Provider value={ctx}>{kind==='full'?<ImageStudio/>:kind==='simple'?<SimpleShell accent="teal" lockedToSimple={false} onAccentChange={noop} onModeChange={noop} onSimpleDefaultChange={noop} simpleDefault/>:<ImageEditor/>}</AppContext.Provider>));await settle();
      const page=terminalBrowserDom(document.body);
      if(kind==='simple')await page.getByRole('tab',{name:'Edit',exact:true}).click();
      if(kind==='editor')await page.locator('.ie-acc-head').filter({hasText:/^AI Edit$/}).click();
      const expected={model:model.id,prompt:'Change only the blue bottle body and blue cap to deep red. Keep its silver collar, handle shape, water droplets, exact silhouette and position unchanged. Preserve both orange pieces, pale stone surface, lighting, shadows, background, framing and empty space on the left. Do not add text or objects.',negativePrompt:'',referenceAssetIds:[image.id],width:768,height:512,count:1,seed:173,advanced:{steps:30,guidanceScale:4,schedulerShift:5,mossImage:{training_shift:1,max_sequence_length:4096},mossImageEdit:{reference_max_edge:512,use_kv_cache:true}}};
      await configure(page,expected,kind,[]);
      const posts=()=>global.fetch.mock.calls.filter(([url,init])=>new URL(url,window.location.href).pathname.endsWith('/image/jobs')&&init?.method==='POST');
      await page.getByRole('button',{name:kind==='full'?'Generate':kind==='simple'?'Generate 1 image':'Generate edit',exact:true}).click();await settle();
      if(requestError==='gate-error'){expect(modalIsShowing()).toBe(false);expect(posts()).toHaveLength(0);expect(ctx.importAsset).not.toHaveBeenCalled();expect(surface.textContent).toContain('owned gate rejection');return;}
      expect(modalIsShowing()).toBe(true);expect(posts()).toHaveLength(0);
      if(kind==='editor')expect(ctx.importAsset).not.toHaveBeenCalled();
      if(requestError==='cancel'){await click(buttonNamed('Go to settings'));await settle();expect(posts()).toHaveLength(0);expect(ctx.importAsset).not.toHaveBeenCalled();expect(ctx.trackEditorScratchOp).not.toHaveBeenCalled();return;}
      if(requestError==='unmount'){await act(async()=>surfaceRoot.unmount());unmounted=true;await acknowledgeRecipeChoice(terminalBrowserDom(document.body),[]);await settle();expect(posts()).toHaveLength(0);expect(ctx.importAsset).not.toHaveBeenCalled();expect(ctx.trackEditorScratchOp).not.toHaveBeenCalled();return;}
      const actions=[];expect(await acknowledgeRecipeChoice(terminalBrowserDom(document.body),actions)).toBe(true);await settle();
      expect(modalIsShowing()).toBe(false);expect(posts()).toHaveLength(1);const posted=JSON.parse(posts()[0][1].body);expect(posted).toMatchObject({model:model.id,prompt:expected.prompt,width:768,height:512,seed:173,count:1,projectId:'project-default',mode:'edit_image',referenceAssetIds:[kind==='editor'?'scratch-owned':image.id],advanced:{steps:30,guidanceScale:4,schedulerShift:5}});expect(posted.advanced.mossImageAdvanced?.positive).toBeUndefined();
      expect(workflowDecisionWrites()).toContainEqual({embedWorkflowInImages:true,workflowEmbedNoticeSeen:true});
      expect(actions).toEqual([expect.objectContaining({action:'click',control:'Include recipe in generated images: Ok, got it'})]);
      expect(await acknowledgeRecipeChoice(terminalBrowserDom(document.body),actions)).toBe(false);expect(posts()).toHaveLength(1);
      if(kind==='editor'){
        if(requestError){expect(surface.querySelector('.ie-busy')).toBeNull();expect(surface.textContent).toContain('owned CPU rejection');expect(ctx.purgeAsset).toHaveBeenCalledWith(expect.objectContaining({id:'scratch-owned'}));expect(ctx.trackEditorScratchOp).not.toHaveBeenCalled();}
        else{expect(surface.querySelector('.ie-busy')).not.toBeNull();expect(ctx.trackEditorScratchOp).toHaveBeenCalledWith('owned-test-job',[expect.objectContaining({id:'scratch-owned'})],{retainCompleted:true});
          const completed={id:'owned-test-job',status:'completed',result:{assets:[{...image,id:'owned-result',displayName:'CPU result',file:{path:'assets/result.png'}}]}};
          await act(async()=>surfaceRoot.render(<AppContext.Provider value={{...ctx,jobs:[completed]}}><ImageEditor/></AppContext.Provider>));await settle();
          expect(surface.querySelector('.ie-busy')).toBeNull();expect(surface.querySelector('.ie-unsaved-badge')).not.toBeNull();expect(surface.querySelector('.ie-moss-results')).not.toBeNull();expect(ctx.releaseEditorScratchOp).toHaveBeenCalledWith('owned-test-job',completed);
        }
      }
    }finally{if(!unmounted)await act(async()=>surfaceRoot.unmount());surface.remove();}
  });

  it('terminal recipe acknowledgment fails closed on a durable-choice persistence error',async()=>{
    await renderApp({}, {workflowPuts:3});let pending;
    await act(async()=>{pending=hoisted.imageCreator({model:'moss_image_base',prompt:'held request'});});await settle();
    await acknowledgeRecipeChoice(terminalBrowserDom(document.body),[]);
    await new Promise(resolve=>setTimeout(resolve,1300));await settle();
    await expect(acknowledgeRecipeChoice(terminalBrowserDom(document.body),[])).rejects.toThrow('recipe choice persistence failed');
    expect(global.fetch.mock.calls.some(([url,init])=>new URL(url,window.location.href).pathname.endsWith('/image/jobs')&&init?.method==='POST')).toBe(false);
    let resolved=false;pending.then(()=>{resolved=true;});await Promise.resolve();expect(resolved).toBe(false);
  });

});
