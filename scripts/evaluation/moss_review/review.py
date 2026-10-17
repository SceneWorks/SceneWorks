#!/usr/bin/env python3
"""Local CPU review packets; never render, acquire weights, or call a judge service."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import statistics
import struct
import sys
import zlib

HARNESS = Path(__file__).resolve().parents[1] / 'moss' / 'evaluate.py'
DEPENDENCIES = ('calibration_source.py', 'calibration_repair.py', 'native.py', 'ownership.py')
# The offline reader also resolves the separately bound calibration archive.
sys.path.insert(0, str(HARNESS.parent))
_spec = importlib.util.spec_from_file_location('frozen_moss_evaluate', HARNESS)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
AXES = ('visual_quality', 'instruction_following', 'composition')
ROLES = ('annotator', 'judge', 'custodian')
BASE_INSTRUCTIONS = {
    'annotator': 'Inspect only the opaque image. Transcribe ALL visible text exactly, preserving case, punctuation and Unicode; never reconstruct text from context. Record text regions and salient object boxes in normalized xyxy coordinates, with independent observation IDs and literal labels. Mark illegible text explicitly and describe uncertainty. Do not inspect prompts, expected answers, generator identities, or the private mapping.',
    'judge': 'Inspect both arms at useful resolution against the supplied scene requirements and frozen rubric. Score EACH arm independently on all three axes (1-5); then select A, B, tie, or unavailable. If either arm is unavailable, use unavailable, never a win. Record visible evidence and uncertainty. Do not inspect generator identities or the private mapping. Machine-assisted judgments must never be labeled human.',
    'custodian': 'After independent observations are sealed, align observed text indices and object observation IDs to private expected element IDs. Null means missing. Do not rewrite observations or fill in requested text. Preserve unexpected text and unmatched objects in original observation records. Keep private mappings and all generator metadata away from annotator and judge.',
}


def check(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    return ev.unseal(load(path))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def evaluator(spec):
    check(set(spec) == {'reviewer', 'reviewer_kind', 'model', 'version', 'tool_revision'}, 'exact evaluator identity fields required; no credentials')
    check(spec['reviewer_kind'] in ('human', 'machine_assisted'), 'invalid reviewer kind')
    check(all(isinstance(spec[k], str) and spec[k].strip() for k in spec), 'empty evaluator identity')
    check(spec['reviewer_kind'] != 'human' or spec['model'] == 'not-applicable', 'human model must be not-applicable')
    return spec


def freeze(lock_path, run, specs_path, out):
    lock = load(lock_path)
    manifest = ev.unseal(lock)
    check(not list(Path(run).rglob('scored.*')), 'review protocol must precede all scored artifacts, including orphan outputs')
    specs = load(specs_path)
    check(set(specs) == set(ROLES), 'declare annotator, judge and custodian before scoring')
    roles = {}
    for role in ROLES:
        check(set(specs[role]) == {'identity', 'task_contexts', 'additional_instructions'}, 'invalid evaluator specification')
        identity = evaluator(specs[role]['identity'])
        instructions = BASE_INSTRUCTIONS[role] + '\n' + specs[role]['additional_instructions']
        roles[role] = dict(identity=identity, task_contexts=specs[role]['task_contexts'], instructions=instructions,
                           instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest())
    # Independence is a declared separate reviewer/context, not a cryptographic identity claim.
    check(len({roles[r]['identity']['reviewer'] for r in ROLES}) == 3, 'independent role/context identities required')
    disjoint_contexts({role: roles[role]['task_contexts'] for role in ROLES}, require_nonempty=True)
    protocol = dict(schema_version=1, lock_sha256=lock['sha256'], run_root=str(Path(run).resolve()), created_at=datetime.now(timezone.utc).isoformat(),
                    before_scored_receipts=True, rubric=manifest['visual_rubric'], thresholds=manifest['thresholds'], scoring_protocol=manifest['scoring_protocol'], roles=roles,
                    review_tool_sha256=ev.file_hash(__file__), harness_sha256=ev.file_hash(HARNESS),
                    harness_dependencies_sha256={name:ev.file_hash(HARNESS.with_name(name)) for name in DEPENDENCIES},
                    legacy_rubric_meaning='componentwise minimum of A/B absolute scores; unavailable pairs omit rubric',
                    limitations='Hashes bind recorded bytes; custody and reviewer independence must be established operationally. Freeze is not a trusted timestamp or signature.')
    save_new(out, ev.seal(protocol))


def verify_review_code(proto):
    check(proto['harness_sha256'] == ev.file_hash(HARNESS) and proto['review_tool_sha256'] == ev.file_hash(__file__), 'frozen review code changed')
    check(proto['harness_dependencies_sha256']=={name:ev.file_hash(HARNESS.with_name(name)) for name in DEPENDENCIES}, 'frozen review dependency changed')


def png_pixels(data):
    """Decode the harness-supported 8-bit RGB/RGBA PNG rows, without color transforms."""
    check(data.startswith(b'\x89PNG\r\n\x1a\n'), 'review image must be PNG')
    offset = 8; chunks = []; compressed = []; header = None
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], 'big'); end = offset + size + 12
        check(end <= len(data), 'truncated PNG')
        kind = data[offset + 4:offset + 8]; body = data[offset + 8:end - 4]
        check(zlib.crc32(kind + body) & 0xffffffff == int.from_bytes(data[end - 4:end], 'big'), 'PNG CRC mismatch')
        chunks.append((kind, data[offset:end]))
        if kind == b'IHDR': header = body
        if kind == b'IDAT': compressed.append(body)
        offset = end
        if kind == b'IEND': break
    check(offset == len(data) and header is not None, 'invalid PNG end/header')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header)
    check(0 < width <= 8192 and 0 < height <= 8192 and depth == 8 and color in (2, 6) and compression == filtering == interlace == 0, 'unsupported review pixel encoding')
    channels = 3 if color == 2 else 4; stride = width * channels
    decoder = zlib.decompressobj(); raw = decoder.decompress(b''.join(compressed), height * (stride + 1) + 1)
    check(decoder.eof and not decoder.unused_data and len(raw) == height * (stride + 1), 'invalid PNG pixels')
    previous = bytearray(stride); digest = hashlib.sha256(header)
    for y in range(height):
        start = y * (stride + 1); kind = raw[start]; row = bytearray(raw[start + 1:start + 1 + stride])
        check(kind in range(5), 'invalid PNG filter')
        if kind:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                up = previous[x]; upper_left = previous[x - channels] if x >= channels else 0
                if kind == 1: predictor = left
                elif kind == 2: predictor = up
                elif kind == 3: predictor = (left + up) // 2
                else:
                    p = left + up - upper_left
                    dl, du, dc = abs(p - left), abs(p - up), abs(p - upper_left)
                    predictor = left if dl <= du and dl <= dc else up if du <= dc else upper_left
                row[x] = (row[x] + predictor) & 255
        digest.update(row); previous = row
    return chunks, digest.hexdigest()


def review_png(source, target):
    original = source.read_bytes(); chunks, pixel_hash = png_pixels(original)
    # Keep original run output immutable. Remove only ancillary identity-bearing metadata.
    removed = [kind.decode('ascii') for kind, _ in chunks if kind in (b'tEXt', b'zTXt', b'iTXt', b'eXIf')]
    reviewed = b'\x89PNG\r\n\x1a\n' + b''.join(chunk for kind, chunk in chunks if kind not in (b'tEXt', b'zTXt', b'iTXt', b'eXIf'))
    _, after = png_pixels(reviewed)
    check(after == pixel_hash, 'review derivative changed decoded pixels')
    target.write_bytes(reviewed)
    return dict(original_sha256=hashlib.sha256(original).hexdigest(), review_sha256=hashlib.sha256(reviewed).hexdigest(), decoded_pixel_sha256=pixel_hash,
                pixel_method='SHA256(IHDR bytes + decoded 8bit RGB/RGBA row bytes), no color/orientation transform', removed_chunks=removed)


def export_verified(protocol_doc, manifest, results, run, public, private):
    protocol = ev.unseal(protocol_doc)
    check(protocol['run_root'] == str(Path(run).resolve()), 'checked run root differs from review freeze')
    public, private = Path(public).resolve(), Path(private).resolve()
    check(not private.is_relative_to(public) and not public.is_relative_to(private), 'private and public directories must be disjoint')
    check(not public.exists() and not private.exists(), 'fresh packet directories required')
    check(len(results) == 288, 'require all 288 validated scored receipts')
    public.mkdir(parents=True); private.mkdir(parents=True)
    image_map = {}; image_tasks = []; image_for_cell = {}
    for cell_id, row in results.items():
        if row['status'] != 'ok':
            image_for_cell[cell_id] = None
            continue
        opaque = secrets.token_hex(16)
        src = ev.artifact(Path(run), row['output'])
        target = public / 'images' / (opaque + '.png')
        target.parent.mkdir(exist_ok=True)
        pixel_proof = review_png(src, target)
        image_map[opaque] = dict(cell_id=cell_id, output_sha256=row['output']['sha256'], **pixel_proof)
        image_for_cell[cell_id] = opaque
        image_tasks.append(dict(image_id=opaque, path='images/' + target.name,
                                sha256=pixel_proof['review_sha256'], width=row['cell']['width'], height=row['cell']['height']))
    pairs = []; pair_map = {}
    for case in manifest['corpus']['cases']:
        for seed in manifest['corpus']['seeds']:
            for moss in ev.CONFIG_IDS[:3]:
                for baseline in ev.CONFIG_IDS[3:]:
                    ids = [f'scored.{moss}.{case["id"]}.{seed}', f'scored.{baseline}.{case["id"]}.{seed}']
                    if secrets.randbelow(2): ids.reverse()
                    opaque = secrets.token_hex(16)
                    pair_map[opaque] = dict(A=ids[0], B=ids[1])
                    arms = {arm: dict(image_id=image_for_cell[cell_id], status='available' if results[cell_id]['status'] == 'ok' else 'unavailable') for arm, cell_id in zip(('A', 'B'), ids)}
                    pairs.append(dict(pair_id=opaque, prompt=case['prompt_raw'], requirements=case['requirements'], arms=arms))
    check(len(pairs) == 432, 'require exactly 432 pairs')
    secrets.SystemRandom().shuffle(image_tasks); secrets.SystemRandom().shuffle(pairs)
    # Annotation packet never includes expected strings, expected boxes, scene prompts, or generator IDs.
    annotator_packet = ev.seal(dict(protocol_sha256=protocol_doc['sha256'], role=protocol['roles']['annotator'], images=image_tasks))
    judge_packet = ev.seal(dict(protocol_sha256=protocol_doc['sha256'], role=protocol['roles']['judge'], rubric=protocol['rubric'], images=image_tasks, pairs=pairs))
    save_new(public / 'annotations.packet.json', annotator_packet)
    save_new(public / 'judgments.packet.json', judge_packet)
    mapping = dict(protocol_sha256=protocol_doc['sha256'], run_root=protocol['run_root'], images=image_map, pairs=pair_map,
                   annotations_packet_sha256=annotator_packet['sha256'], judgments_packet_sha256=judge_packet['sha256'])
    save_new(private / 'mapping.json', ev.seal(mapping))
    # A fresh vision context receives only one role's bounded batch and its opaque image files.
    for name, packet, records_key in (('annotation', annotator_packet, 'images'), ('judge', judge_packet, 'pairs')):
        data = ev.unseal(packet)
        for offset in range(0, len(data[records_key]), 24):
            batch = dict(data)
            batch[records_key] = data[records_key][offset:offset + 24]
            if name == 'judge':
                needed = {a['image_id'] for pair in batch['pairs'] for a in pair['arms'].values() if a['image_id'] is not None}
                batch['images'] = [i for i in image_tasks if i['image_id'] in needed]
            batch.update(parent_packet_sha256=packet['sha256'], batch_index=offset // 24)
            save_new(public / (name + '-batches') / f'{offset // 24:03d}.json', ev.seal(batch))
    return mapping


def disjoint_contexts(by_role, require_nonempty=False):
    contexts = {}
    for role, values in by_role.items():
        check(isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values), 'exact actual task/context identities required')
        contexts[role] = set(values)
        check(not require_nonempty or (values and len(values) == len(contexts[role])), 'freeze distinct nonempty actual context lists')
    for index, role in enumerate(ROLES):
        for other in ROLES[index + 1:]:
            check(not contexts[role].intersection(contexts[other]), 'actual context reused across independent review roles')


def verify_protocol(protocol, manifest):
    disjoint_contexts({role: protocol['roles'][role]['task_contexts'] for role in ROLES}, require_nonempty=True)
    check(protocol['rubric'] == manifest['visual_rubric'] and protocol['thresholds'] == manifest['thresholds'], 'frozen rubric/thresholds mismatch')
    for role in ROLES:
        item = protocol['roles'][role]
        evaluator(item['identity'])
        check(hashlib.sha256(item['instructions'].encode()).hexdigest() == item['instructions_sha256'], 'instruction bytes/hash mismatch')


def export(lock, run, protocol, public, private):
    doc = load(protocol); proto = ev.unseal(doc)
    check(load(lock)['sha256'] == proto['lock_sha256'], 'protocol/lock mismatch')
    check(proto['run_root'] == str(Path(run).resolve()), 'checked run root differs from review freeze')
    verify_review_code(proto)
    manifest, results = ev.receipts(lock, run)
    verify_protocol(proto, manifest)
    export_verified(doc, manifest, results, run, public, private)


def provenance(record, role, protocol, evidence):
    check(record['evaluator'] == protocol['roles'][role]['identity'], 'unfrozen evaluator identity')
    check(record['instructions_sha256'] == protocol['roles'][role]['instructions_sha256'], 'instruction provenance mismatch')
    check(record['uncertainty'] in ('low', 'medium', 'high') and isinstance(record['notes'], str) and record['notes'].strip(), 'observation evidence/uncertainty required')
    # Transcript paths stay private and must be regular confined files, not arbitrary external paths.
    check(ev.artifact(Path(evidence), record['transcript']).stat().st_size > 0, 'empty evaluator transcript')
    check(record.get('task_context') in protocol['roles'][role]['task_contexts'], 'actual context not bound to frozen evaluator identity')


def observations(rows, packet, protocol, evidence):
    tasks = {x['image_id']: x for x in packet['images']}
    check(len(rows) == len(tasks) and {r['image_id'] for r in rows} == set(tasks), 'missing or duplicate independent observations')
    output = {}
    for row in rows:
        provenance(row, 'annotator', protocol, evidence)
        check(row['output_sha256'] == tasks[row['image_id']]['sha256'], 'observed image hash mismatch')
        check(isinstance(row['text'], list) and all(isinstance(t, str) for t in row['text']), 'raw text must be observed strings')
        check(isinstance(row['boxes'], list), 'boxes must be independent observations')
        ids = [x['observation_id'] for x in row['boxes']]
        check(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), 'duplicate or empty observed box ID')
        for box in row['boxes']:
            check(isinstance(box['label'], str) and box['label'].strip(), 'literal observed object/text label required')
            ev.iou([0, 0, 1, 1], box['box'])
        output[row['image_id']] = row
    return output


def score_arm(value):
    check(set(value) == set(AXES) and all(type(value[k]) is int and 1 <= value[k] <= 5 for k in AXES), 'each available arm requires three absolute 1-5 scores')


def translate_verified(manifest, results, protocol_doc, mapping_doc, annotation_packet, judge_packet, raw, ratings, alignments, public, evidence):
    protocol, mapping = ev.unseal(protocol_doc), ev.unseal(mapping_doc)
    verify_protocol(protocol, manifest)
    check(mapping['run_root'] == protocol['run_root'], 'mapping run root differs from review freeze')
    disjoint_contexts({'annotator': [r.get('task_context') for r in raw], 'judge': [r.get('task_context') for r in ratings], 'custodian': [r.get('task_context') for r in alignments]})
    annotation_tasks, judge_tasks = ev.unseal(annotation_packet), ev.unseal(judge_packet)
    check(mapping['protocol_sha256'] == annotation_tasks['protocol_sha256'] == judge_tasks['protocol_sha256'] == protocol_doc['sha256'], 'review freeze mismatch')
    check(mapping['annotations_packet_sha256'] == annotation_packet['sha256'] and mapping['judgments_packet_sha256'] == judge_packet['sha256'], 'public packet changed')
    for task in annotation_tasks['images']:
        ev.artifact(Path(public), dict(path=task['path'], sha256=task['sha256']))
    obs = observations(raw, annotation_tasks, protocol, evidence)
    check(set(obs) == set(mapping['images']), 'private image coverage mismatch')
    case_by_id = {c['id']: c for c in manifest['corpus']['cases']}
    check(len(alignments) == len(obs) and {a['image_id'] for a in alignments} == set(obs), 'exact private alignment coverage required')
    annotations = {}
    for a in alignments:
        provenance(a, 'custodian', protocol, evidence)
        image_id = a['image_id']; row = obs[image_id]; cell_id = mapping['images'][image_id]['cell_id']
        receipt = results[cell_id]; case = case_by_id[receipt['cell']['case_id']]
        image_binding = mapping['images'][image_id]
        check(row['output_sha256'] == image_binding['review_sha256'] and image_binding['original_sha256'] == receipt['output']['sha256'], 'observations no longer bind result')
        original = ev.artifact(Path(protocol['run_root']), receipt['output'])
        reviewed = ev.artifact(Path(public), next(dict(path=i['path'], sha256=i['sha256']) for i in annotation_tasks['images'] if i['image_id'] == image_id))
        check(png_pixels(original.read_bytes())[1] == png_pixels(reviewed.read_bytes())[1] == image_binding['decoded_pixel_sha256'], 'review/original decoded pixel proof mismatch')
        indices = a['text_indices']; box_ids = a['box_ids']
        check(len(indices) == len(case['expected_text']), 'expected-line alignment count')
        used = [i for i in indices if i is not None]
        check(all(type(i) is int and 0 <= i < len(row['text']) for i in used) and len(used) == len(set(used)), 'invalid or reused text observation')
        check(set(box_ids) == {b['id'] for b in case['boxes']}, 'exact expected box alignment required')
        boxes = {b['observation_id']: b['box'] for b in row['boxes']}
        used_boxes = [i for i in box_ids.values() if i is not None]
        check(all(i in boxes for i in used_boxes) and len(used_boxes) == len(set(used_boxes)), 'invalid or reused box observation')
        annotations[cell_id] = dict(output_sha256=receipt['output']['sha256'], review_image_sha256=row['output_sha256'], annotator=row['evaluator']['reviewer'], method='independent pixel-first observations, private custodian alignment', tool_revision=row['evaluator']['tool_revision'],
            text=['' if i is None else row['text'][i] for i in indices], boxes={k: None if i is None else boxes[i] for k, i in box_ids.items()},
            observation_sha256=ev.digest(row), alignment_sha256=ev.digest(a))
    tasks = {p['pair_id']: p for p in judge_tasks['pairs']}
    expected = {frozenset((f'scored.{m}.{c["id"]}.{seed}', f'scored.{b}.{c["id"]}.{seed}')) for c in manifest['corpus']['cases'] for seed in manifest['corpus']['seeds'] for m in ev.CONFIG_IDS[:3] for b in ev.CONFIG_IDS[3:]}
    actual = [frozenset(p.values()) for p in mapping['pairs'].values()]
    check(len(actual) == len(expected) == 432 and set(actual) == expected, 'missing or duplicated matchup coverage')
    check(len(ratings) == len(tasks) == len(mapping['pairs']) == 432 and {r['pair_id'] for r in ratings} == set(tasks) == set(mapping['pairs']), 'require exactly 432 unique judgments')
    judgments = []
    for rating in ratings:
        provenance(rating, 'judge', protocol, evidence)
        pair = tasks[rating['pair_id']]; private_pair = mapping['pairs'][rating['pair_id']]
        check(rating['winner'] in ('A', 'B', 'tie', 'unavailable'), 'invalid winner')
        unavailable = False
        for arm in ('A', 'B'):
            cell_id = private_pair[arm]; actual = results[cell_id]
            image_id = pair['arms'][arm]['image_id']
            expected_hash = mapping['images'][image_id]['review_sha256'] if image_id else None
            check(rating['viewed_image_hashes'][arm] == expected_hash, 'judgment must bind each viewed image hash')
            check((image_id is None) == (actual['status'] != 'ok'), 'arm availability changed')
            if image_id is not None: check(mapping['images'][image_id]['cell_id'] == cell_id, 'arm mapping changed')
            if actual['status'] != 'ok':
                unavailable = True; check(rating['scores'][arm] is None, 'unavailable arm cannot be scored')
            else: score_arm(rating['scores'][arm])
        check((rating['winner'] == 'unavailable') == unavailable, 'unavailable pair cannot become win/tie')
        identity = rating['evaluator']
        j = dict(pair_id=rating['pair_id'], winner=rating['winner'], reviewer=identity['reviewer'], reviewer_kind=identity['reviewer_kind'], judge_version=identity['model'] + '@' + identity['version'], notes=rating['notes'], uncertainty=rating['uncertainty'], record_sha256=ev.digest(rating))
        if not unavailable: j['rubric'] = {k: min(rating['scores'][a][k] for a in ('A', 'B')) for k in AXES}
        judgments.append(j)
    successful = {k for k, r in results.items() if r['status'] == 'ok'}
    check(set(annotations) == successful, 'successful output annotation coverage mismatch')
    return annotations, judgments


def legacy_pairs(protocol_doc, mapping_doc, judge_packet, results, public):
    mapping, tasks = ev.unseal(mapping_doc), ev.unseal(judge_packet)
    image_tasks = {i['image_id']: i for i in tasks['images']}
    rows = []
    for pair in tasks['pairs']:
        entry = dict(id=pair['pair_id'], prompt=pair['prompt'], requirements=pair['requirements'], arms=[])
        for arm in ('A', 'B'):
            i = pair['arms'][arm]['image_id']; task = image_tasks[i] if i else None
            cell_id = mapping['pairs'][pair['pair_id']][arm]
            original = results[cell_id].get('output') if task else None
            entry['arms'].append(dict(arm=arm, image=str(ev.artifact(Path(mapping['run_root']),original)) if original else None, sha256=original['sha256'] if original else None, status='available' if task else 'unavailable'))
        rows.append(entry)
    return ev.seal(dict(lock_sha256=ev.unseal(protocol_doc)['lock_sha256'], pairs=rows)), ev.seal(dict(lock_sha256=ev.unseal(protocol_doc)['lock_sha256'], mapping=[dict(id=k, **v) for k, v in mapping['pairs'].items()]))


def summaries(manifest, results, mapping_doc, ratings, annotations):
    mapping = ev.unseal(mapping_doc); groups = {}
    for m in ev.CONFIG_IDS[:3]:
        for b in ev.CONFIG_IDS[3:]:
            for scenario in sorted({c['scenario'] for c in manifest['corpus']['cases']}):
                groups[(m, b, scenario)] = dict(moss=m, baseline=b, scenario=scenario, comparisons=0, wins=Counter(), scores={m: {k: [] for k in AXES}, b: {k: [] for k in AXES}})
    for r in ratings:
        pair = mapping['pairs'][r['pair_id']]; rows = {a: results[pair[a]] for a in ('A', 'B')}
        configs = {a: rows[a]['cell']['config_id'] for a in rows}
        m = next(v for v in configs.values() if v in ev.CONFIG_IDS[:3]); b = next(v for v in configs.values() if v in ev.CONFIG_IDS[3:])
        g = groups[(m, b, rows['A']['cell']['scenario'])]; g['comparisons'] += 1
        g['wins'][configs[r['winner']] if r['winner'] in ('A', 'B') else r['winner']] += 1
        for a, config in configs.items():
            if r['scores'][a] is not None:
                for axis in AXES: g['scores'][config][axis].append(r['scores'][a][axis])
    tables = []
    for g in groups.values():
        costs = {}
        for config in (g['moss'], g['baseline']):
            rows = [r for r in results.values() if r['cell']['config_id'] == config and r['cell']['scenario'] == g['scenario']]
            # Keep source metrics intact: cold/warm, sampled device peaks and allocator peaks are different measures.
            def measured(key, condition=None):
                values = [r['metrics'].get(key) for r in rows if r['status'] == 'ok' and (condition is None or r['metrics'].get('latency_condition') == condition)]
                values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
                return dict(n=len(values), median=statistics.median(values) if values else None, minimum=min(values) if values else None, maximum=max(values) if values else None)
            objective = [ev.objective_scores(next(c for c in manifest['corpus']['cases'] if c['id'] == r['cell']['case_id']), annotations[r['cell']['id']]) for r in rows if r['status'] == 'ok']
            costs[config] = dict(statuses=dict(Counter(r['status'] for r in rows)),failure_kinds=dict(Counter(r.get('failure_kind','none') for r in rows)),censored=sum(r.get('failure_kind')=='runtime_censored' for r in rows),
                measured_summary={key: measured(key) for key in ('cold_seconds', 'warm_seconds', 'peak_host_rss_bytes', 'download_bytes')},
                gpu_peaks_by_method={method: [r['metrics']['peak_gpu_bytes'] for r in rows if r['status'] == 'ok' and r['metrics'].get('gpu_memory_method') == method] for method in {r['metrics'].get('gpu_memory_method') for r in rows if r['status'] == 'ok'}},
                independent_objective_scores=objective, observations=[dict(cell_id=r['cell']['id'], metrics=r.get('metrics'), elapsed_seconds=r['elapsed_seconds'],attempted=r.get('attempted'),failure_kind=r.get('failure_kind'),runtime_failure_cell=r.get('runtime_failure_cell'),censored_reason=r.get('censored_reason'),error=r.get('error')) for r in rows],
                quality={k: dict(n=len(v), mean=statistics.mean(v) if v else None) for k, v in g['scores'][config].items()})
        tables.append(dict(moss=g['moss'], baseline=g['baseline'], scenario=g['scenario'], comparisons=g['comparisons'], wins=dict(g['wins']), costs=costs))
    check(len(tables) == 108 and all(t['comparisons'] == 4 for t in tables), 'nine matchups x twelve scenarios x four comparisons required')
    return dict(tables=tables, limitations='Descriptive screening only; repeated pair exposures are not independent samples. Scores remain machine-assisted or human as recorded. Do not combine cold/warm latency, allocator/device peaks, or differing runtime/precision into architecture-only efficiency. Preparation acquisition bytes are separate evidence; absent metrics stay null. Analyst conclusions and Michael inclusion decision remain required.')


def validate_translate(lock, run, protocol_path, public, private, raw_path, ratings_path, alignment_path, evidence, out):
    protocol = load(protocol_path)
    check(ev.unseal(protocol)['run_root'] == str(Path(run).resolve()), 'checked run root differs from review freeze')
    manifest, results = ev.receipts(lock, run)
    protocol = load(protocol_path); check(ev.unseal(protocol)['lock_sha256'] == load(lock)['sha256'], 'lock identity mismatch')
    verify_review_code(ev.unseal(protocol))
    mapping = load(Path(private) / 'mapping.json'); ap = load(Path(public) / 'annotations.packet.json'); jp = load(Path(public) / 'judgments.packet.json')
    raw, ratings, alignment = load(raw_path), load(ratings_path), load(alignment_path)
    annotations, judgments = translate_verified(manifest, results, protocol, mapping, ap, jp, raw, ratings, alignment, public, evidence)
    out = Path(out).resolve(); check(not out.exists() and not out.is_relative_to(Path(public).resolve()), 'fresh private translated output outside public required')
    out.mkdir(parents=True)
    for name, value in [('annotations.json', annotations), ('judgments.json', judgments)]: save_new(out / name, value)
    pairs, secret = legacy_pairs(protocol, mapping, jp, results, public)
    save_new(out / 'pairs.json', pairs); save_new(out / 'pairs.json.private.json', secret)
    save_new(out / 'analyst-tables.json', summaries(manifest, results, mapping, ratings, annotations))
    bindings = {str(Path(p).resolve()): ev.file_hash(p) for p in (lock, protocol_path, raw_path, ratings_path, alignment_path)}
    save_new(out / 'review-receipt.json', ev.seal(dict(inputs=bindings, protocol_sha256=protocol['sha256'], outputs={p.name: ev.file_hash(p) for p in out.iterdir()}, actual_judgments=len(judgments), independent_image_observations=len(raw))))


def merge_records(key, inputs, output):
    check(key in ('image_id', 'pair_id'), 'merge requires opaque record key')
    records = []
    for path in inputs:
        rows = load(path)
        check(isinstance(rows, list), 'batch responses must be JSON lists')
        records.extend(rows)
    check(len({r[key] for r in records}) == len(records), 'duplicate checkpoint records; do not silently replace')
    save_new(output, records)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('freeze')
    for name in ('lock', 'run', 'evaluators', 'output'): a.add_argument(name)
    a = sub.add_parser('export')
    for name in ('lock', 'run', 'protocol', 'public', 'private'): a.add_argument(name)
    a = sub.add_parser('translate')
    for name in ('lock', 'run', 'protocol', 'public', 'private', 'observations', 'ratings', 'alignments', 'evidence', 'output'): a.add_argument(name)
    a = sub.add_parser('merge'); a.add_argument('key', choices=('image_id', 'pair_id')); a.add_argument('output'); a.add_argument('inputs', nargs='+')
    a = p.parse_args()
    if a.command == 'freeze': freeze(a.lock, a.run, a.evaluators, a.output)
    elif a.command == 'export': export(a.lock, a.run, a.protocol, a.public, a.private)
    elif a.command == 'merge': merge_records(a.key, a.inputs, a.output)
    else: validate_translate(a.lock, a.run, a.protocol, a.public, a.private, a.observations, a.ratings, a.alignments, a.evidence, a.output)


if __name__ == '__main__': main()
