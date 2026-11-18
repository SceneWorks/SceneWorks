// Isolated terminal acceptance. Take an edited, reviewed row through Studio formatting,
// the real prompt-batch store and the same portable export/import functions as Studio.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {formatPromptBatch,snapRatio} from '../apps/web/src/promptBatchDescribe.js';
import {extractKeys} from '../apps/web/src/promptBatch.js';
import {serializePromptBatchExport,fromPromptBatchImport} from '../apps/web/src/promptBatchIO.js';

const [origin,directory] = process.argv.slice(2);
assert.equal(new URL(origin).hostname,'127.0.0.1');
const {items,dataset,model,variant} = JSON.parse(await readFile(path.join(directory,'review-input.json'),'utf8'));
const edit = 'A reviewed geometric print.';
const rows = items.map(item => ({...item,fields:{...item}}));
rows[0].fields.style = edit;
const {text,skipped} = formatPromptBatch(rows);
assert.equal(skipped.length,0);
const prompts = text.split('\n---\n');
assert.equal(prompts.length,2);
assert(prompts[0].includes('Style: '+edit));
const source = {kind:'training_dataset',datasetId:dataset.id,datasetName:dataset.name,datasetVersion:dataset.version,model,variant};
const sourceImages = items.map(({itemId,width,height})=>({itemId,width,height,ratio:snapRatio(width,height),model,variant}));
const variables = extractKeys(prompts).map(key=>({key,values:[]}));
const route = new URL('/api/v1/prompt-batches',origin);
const create = await fetch(route,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:'Reviewed SenseNova shapes',scope:'global',prompts,variables,source,sourceImages})});
assert.equal(create.status,201,await create.clone().text());
const {id} = await create.json();
const load = await fetch(route.href+'/'+encodeURIComponent(id));
assert.equal(load.status,200);
const saved = await load.json();
assert.deepEqual(saved.prompts,prompts);
assert.deepEqual(saved.source,source);
assert.deepEqual(saved.sourceImages,sourceImages);
const exported = serializePromptBatchExport(saved);
assert.deepEqual(fromPromptBatchImport(exported).prompts,saved.prompts);
await writeFile(path.join(directory,'reviewed-prompt-batch.json'),JSON.stringify(saved,null,2));
await writeFile(path.join(directory,'reviewed-export.json'),exported);
