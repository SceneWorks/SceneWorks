import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { loadUpstreamReference, PARITY_SOURCE_INDICES, UPSTREAM_REVISION } from "./starvector-terminal-upstream-reference.mjs";
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
test("oracle consumer binds twenty distinct source rows and immutable output bytes", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "oracle-reference-")); t.after(() => rm(root, { recursive:true, force:true }));
  const rows = Array.from({length:120}, (_,i) => ({png_sha256:hash(`input ${i}`)}));
  const artifact = async (name, bytes) => { await writeFile(path.join(root,name),bytes); return hash(bytes); };
  const config = await artifact("config.json", '{"max_position_embeddings":8192}\n'), processor = await artifact("processor.json", '{"size":224}\n'), transcript = await artifact("transcript.json", '{"source":"upstream"}\n');
  const reference = {schema_version:1, upstream_reference:{implementation_repository:"https://github.com/joanrod/star-vector",implementation_revision:UPSTREAM_REVISION,checkpoint_repository:"starvector/starvector-1b-im2svg",checkpoint_revision:"380ab95d25a8e9ab1dc825debe238b4953ae13b9",checkpoint_inventory_sha256:hash("weights"),config_sha256:config,processor_sha256:processor,transcript_sha256:transcript},config_path:"config.json",processor_path:"processor.json",transcript_path:"transcript.json",cases:[]};
  for (const [i,source] of PARITY_SOURCE_INDICES.entries()) reference.cases.push({case_index:i,source_case_index:source,seed:i,input_png_sha256:rows[source].png_sha256,upstream_svg:`${i}.svg`,upstream_svg_sha256:await artifact(`${i}.svg`,`<svg xmlns="http://www.w3.org/2000/svg"><circle r="${i+1}"/></svg>`),upstream_preview_png:`${i}.png`,upstream_preview_png_sha256:await artifact(`${i}.png`,Buffer.from([137,80,78,71,i]))});
  const manifest = path.join(root,"upstream-reference-1b.json"); await writeFile(manifest,JSON.stringify(reference));
  assert.equal((await loadUpstreamReference(root,"1b",rows)).cases.length,20);
  await writeFile(path.join(root,"upstream-reference-1b.pending.json"),JSON.stringify(reference));
  assert.equal((await loadUpstreamReference(root,"1b",rows,{pending:true})).cases.length,20);
  const copy=structuredClone(reference); copy.cases[6].source_case_index=6; await writeFile(manifest,JSON.stringify(copy)); await assert.rejects(loadUpstreamReference(root,"1b",rows), /selected source/);
  await writeFile(manifest,JSON.stringify(reference)); await writeFile(path.join(root,"0.svg"),"changed"); await assert.rejects(loadUpstreamReference(root,"1b",rows), /content changed/);
});

test("oracle consumer binds accepted and rejected outcomes with immutable rejection evidence", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "oracle-outcomes-")); t.after(() => rm(root, { recursive:true, force:true }));
  const rows = Array.from({length:120}, (_,i) => ({png_sha256:hash(`input ${i}`)}));
  const artifact = async (name, bytes) => { await writeFile(path.join(root,name),bytes); return hash(bytes); };
  const config = await artifact("config.json", "{}\n"), processor = await artifact("processor.json", "{}\n"), transcript = await artifact("transcript.json", "{}\n");
  const reference = {schema_version:2, upstream_reference:{implementation_repository:"https://github.com/joanrod/star-vector",implementation_revision:UPSTREAM_REVISION,checkpoint_repository:"starvector/starvector-1b-im2svg",checkpoint_revision:"380ab95d25a8e9ab1dc825debe238b4953ae13b9",checkpoint_inventory_sha256:hash("weights"),config_sha256:config,processor_sha256:processor,transcript_sha256:transcript},config_path:"config.json",processor_path:"processor.json",transcript_path:"transcript.json",cases:[]};
  for (const [i,source] of PARITY_SOURCE_INDICES.entries()) {
    if (i === 1) {
      const reason = "provider SVG is malformed: tag not closed", raw = await artifact("rejected.svg", "<svg>"), stdoutBytes = JSON.stringify({outcome:"rejected",error_code:reason,canonical_svg_sha256:null,preview_png_sha256:null,published_paths:[],staging_residue:[],result_contains_inline_svg:false}), stdout = await artifact("sanitizer.json",stdoutBytes), stderr = await artifact("sanitizer.err","");
      reference.cases.push({case_index:i,source_case_index:source,seed:i,input_png_sha256:rows[source].png_sha256,outcome:"rejected",rejection_stage:"sanitizer",rejection_reason:reason,upstream_raw_svg:"rejected.svg",upstream_raw_svg_sha256:raw,sanitizer_stdout:"sanitizer.json",sanitizer_stdout_sha256:stdout,sanitizer_stderr:"sanitizer.err",sanitizer_stderr_sha256:stderr});
    } else reference.cases.push({case_index:i,source_case_index:source,seed:i,input_png_sha256:rows[source].png_sha256,outcome:"accepted",upstream_svg:`${i}.svg`,upstream_svg_sha256:await artifact(`${i}.svg`,"<svg/>") ,upstream_preview_png:`${i}.png`,upstream_preview_png_sha256:await artifact(`${i}.png`,Buffer.from([137,80,78,71,i]))});
  }
  await writeFile(path.join(root,"upstream-reference-1b.json"),JSON.stringify(reference));
  const loaded = await loadUpstreamReference(root,"1b",rows);
  assert.equal(loaded.cases[1].outcome,"rejected");
  assert.equal(loaded.cases[1].rejection_code,"malformed_svg");
  const forged=structuredClone(reference);forged.cases[1].rejection_reason="generic infrastructure error";await writeFile(path.join(root,"upstream-reference-1b.json"),JSON.stringify(forged));
  await assert.rejects(loadUpstreamReference(root,"1b",rows),/typed SVG policy/);
});
