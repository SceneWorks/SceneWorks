import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile as fsWrite } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { buildReceipt, cancellationSafe, compareReceipts, detectFullCacheTemporary, readDarwinMemory, readSealedJson, validateCampaign, validateReceipt, writeSealedJson } from "./kv-baseline-harness.mjs";

const life = Object.fromEntries(["append","chunkedPrefill","singleShotPrefill","promptCacheReuse","trim","rollback","clear","cancel","clone","batchSplit","batchMerge","prefixCopyOnWrite","pageImport","pageExport","serialization","restore","denseFallback","postRunRelease"].map(k=>[k,true]));
const fixture = (mode="dense", extra={}) => buildReceipt({
  runId: mode, capturedAt:"2026-08-29T12:00:00Z", mode, status:"complete",
  provenance:{sceneWorksRevision:"a".repeat(40),inferenceRevision:"b".repeat(40),mlxRevision:"c".repeat(40),os:"macOS",xcode:"Xcode",hardware:"Apple",modelId:"fixture",modelFileSha256:"d".repeat(64),powerMode:"AC",thermalState:"nominal",command:"fixture"},
  matrix:{family:"llama",contextBand:"short",requestMode:"single",prefillMode:"single-shot",processTemperature:"cold"},
  geometry:{batch:1,queryHeads:8,kvHeads:8,headDimension:128,queryLength:1,kvLength:4096,layers:2,elementBytes:2,capacity:4096},
  memory:{modelWeightsBytes:1000,persistentKvBytes:mode==="dense"?1000:500,transientWorkspaceBytes:100,denseTheoreticalKvBytes:16777216,
    phaseSamples:[{phase:"before",pid:9,source:"footprint",timestamp:"t",physFootprintBytes:2000,mlx:{activeBytes:100,cacheBytes:10,peakBytes:110}},{phase:"after",pid:9,source:"footprint",timestamp:"t2",physFootprintBytes:mode==="dense"?4000:3000,mlx:{activeBytes:1000,cacheBytes:100,peakBytes:1100}}],
    allocationEvents:[{kind:"compressed_cache",phase:"after",timestamp:"t2",bytes:500}],reconciliation:{expectedDenseKvBytes:16777216,observedPersistentKvBytes:500,toleranceBytes:0},release:{verified:true,afterBytes:100}},
  timings:{loadMs:1,prefillMs:2,firstTokenMs:3,decodeTokensPerSecond:mode==="dense"?100:98,coldCompileMs:4,warmCompileMs:1},
  quality:{parityMaxError:0,perplexityDelta:0,greedyTokenAgreement:0,structuredToolAgreement:0,needleRetrieval:0,multiTurnPromptCache:0,statistics:{repeats:5,warmups:2,confidenceInterval:"95%",outlierPolicy:"report all",variancePolicy:"frozen"}},
  lifecycle:life,cancellation:{cleanupVerified:true},...extra
});
test("sealed comparison and canonical hash",()=>{const d=fixture(),c=fixture("compressed");assert.doesNotThrow(()=>validateReceipt(d));assert.equal(compareReceipts(d,c).persistentKvReduction,.5);d.memory.persistentKvBytes=2;assert.throws(()=>validateReceipt(d),/receiptSha256 mismatch/);});
test("real footprint units and identity",()=>{assert.deepEqual(readDarwinMemory(9,()=>"phys_footprint: 1664 KB","prefill","t"),{phase:"prefill",pid:9,source:"footprint -p",timestamp:"t",physFootprintBytes:1703936});assert.throws(()=>readDarwinMemory(9,()=>"phys_footprint: 2 bananas"),/units/);});
test("attribution detection and lifecycle cleanup",async()=>{assert.equal(detectFullCacheTemporary([{kind:"dense_cache_temporary",phase:"decode",timestamp:"t",bytes:1}],100).detected,true);let n=0;await assert.rejects(cancellationSafe(async()=>{throw Error("cancel")},async()=>{n++}));const ac=new AbortController();ac.abort();await assert.rejects(cancellationSafe(async()=>{},async()=>{n++},ac.signal));assert.equal(n,2);});
test("sidecar verification and incomplete campaign",async()=>{const dir=await mkdtemp(path.join(tmpdir(),"kv-seal-")),file=path.join(dir,"receipt.json");await writeSealedJson(file,fixture());assert.equal((await readSealedJson(file)).runId,"dense");await fsWrite(file+".sha256","0".repeat(64));await assert.rejects(readSealedJson(file),/sidecar hash mismatch/);await rm(dir,{recursive:true,force:true});assert.throws(()=>validateCampaign([fixture()]),/campaign incomplete/);});
