#!/usr/bin/env python3
"""Frozen 30-task upstream versus independent-object-probability experiment."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from experiment_utils import ROOT, key, write, metrics
import experiment_utils as records


def prepare(batch,reference=None):
    source = ROOT / 'experiments/manifests/mp3d_sr_dev100_seed20260911_runtime.json'
    manifest = json.loads(source.read_text())
    manifest['episodes'] = manifest['episodes'][:30]
    assert len(manifest['episodes']) == len(set(map(key, manifest['episodes']))) == 30
    baseline_hashes={}
    if reference is not None:
        reference=reference.resolve()
        old=json.loads((reference/'manifest.json').read_text())
        if old['episodes']!=manifest['episodes']:
            raise ValueError('Reference baseline task list/order mismatch')
        if set(records.read_records(reference/'baseline/results'))!=set(map(key,manifest['episodes'])):
            raise ValueError('Reference baseline is incomplete')
        provenance=list((reference/'baseline/results').rglob('baseline_source.json'))
        if len(provenance)!=1 or json.loads(provenance[0].read_text())['commit']!='049971b613fcd2554b034bd4a853d79cc1e61613':
            raise ValueError('Expected frozen official baseline provenance')
        old_hashes=json.loads((reference/'source_sha256.json').read_text())
        for name in ('scripts/run_apexnav.sh','scripts/run_upstream_apexnav.sh','config/habitat_eval_mp3d.yaml'):
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=old_hashes[name]:
                raise ValueError('Shared baseline runner/config changed: '+name)
        for path in list((reference/'baseline/results').rglob('record.txt'))+provenance:
            baseline_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    batch.mkdir(parents=True, exist_ok=False)
    if reference is not None:
        (batch/'baseline').symlink_to(reference/'baseline',target_is_directory=True)
        write(batch/'baseline_reference.json',dict(source=str(reference),sha256=baseline_hashes))
    write(batch / 'manifest.json', manifest)
    write(batch / 'conditions.json', dict(
        tasks_per_arm=30, selection='Unchanged first 30 entries of fixed100; development sample',
        source=str(source), control='Frozen upstream Apexnav_1 via run_upstream_apexnav.sh',
        treatment='Current workspace, original-cluster real crops + context -> independent object marginals',
        contract='frontier_object_marginals_v1', provider='local', generated_images=0,
        probability_normalization=False, score_mode='text_frontier',baseline_reference=str(reference) if reference else None,
        max_calls_per_episode=500, cooldown_steps=0, warmup_steps=26,
        max_candidates=3, route_commitment_max_steps=6,
        note='500-query ceiling covers the 500-step episode; no extra cooldown. Existing exploration eligibility retained. '
             'Upstream comparison measures the complete system, not an isolated causal ablation of the dictionary.'))
    paths = list((ROOT / 'avic_bridge').glob('*.py')) + [ROOT / p for p in (
        'habitat_evaluation.py','scripts/experiment_utils.py', 'config/frontier_object_dictionary.json',
        'config/habitat_eval_mp3d.yaml', 'scripts/frontier_dictionary_batch.py',
        'scripts/run_frontier_dictionary30.sh', 'scripts/run_apexnav.sh',
        'scripts/run_upstream_apexnav.sh',
        'src/planner/exploration_manager/src/exploration_manager.cpp',
        'src/planner/exploration_manager/src/exploration_fsm.cpp',
        'devel/lib/exploration_manager/exploration_node')]
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        if not relative.startswith('devel/'):
            dest = batch / 'source_snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    hashes['@manifest'] = hashlib.sha256((batch / 'manifest.json').read_bytes()).hexdigest()
    write(batch / 'source_sha256.json', hashes)
    print('Prepared 30 fixed tasks per arm: upstream -> dictionary.', flush=True)


def verify(batch):
    for name, expected in json.loads((batch / 'source_sha256.json').read_text()).items():
        path = batch / 'manifest.json' if name == '@manifest' else ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source/manifest changed during experiment: ' + name)
    reference=batch/'baseline_reference.json'
    if reference.exists():
        for name,expected in json.loads(reference.read_text())['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Reference baseline data changed: '+name)


def compare(batch):
    verify(batch)
    wanted = set(map(key, json.loads((batch / 'manifest.json').read_text())['episodes']))
    outcomes = {arm: records.read_records(batch / arm / 'results') for arm in ('baseline', 'dictionary')}
    if any(set(rows) != wanted for rows in outcomes.values()):
        raise ValueError('Incomplete/mismatched paired30 results; no final SR comparison')
    rows = [dict(scene=k[0], episode=k[1], baseline=outcomes['baseline'][k],
                 dictionary=outcomes['dictionary'][k]) for k in sorted(wanted)]
    arms = {arm: dict(successes=sum(v == 'success' for v in values.values()), tasks=30,
                     metrics=metrics(batch / arm / 'results')) for arm, values in outcomes.items()}
    rescued = [r for r in rows if r['baseline'] != 'success' and r['dictionary'] == 'success']
    lost = [r for r in rows if r['baseline'] == 'success' and r['dictionary'] != 'success']
    decisions = acknowledgements = changed_targets = predictions = 0
    for path in (batch / 'dictionary/logs').rglob('evaluation.log'):
        with path.open() as stream:
            for line in stream:
                decisions += line.startswith('[FRONTIER_DICTIONARY] ')
                if line.startswith('[AVIC_FRONTIER_ACK] '):
                    acknowledgements += 1
    for path in (batch / 'dictionary/logs').rglob('ros.log'):
        with path.open() as stream:
            for line in stream:
                if '[AVIC_EFFECT]' in line and re.search(r'\btarget_changed=1\b', line) and re.search(r'\bapplied=1\b', line):
                    changed_targets += 1
    for path in (batch / 'predictions').glob('*/result.json'):
        result = json.loads(path.read_text())
        if result.get('contract') != 'frontier_object_marginals_v1' or result.get('generated_images') != 0:
            raise ValueError('Unexpected model pipeline: ' + str(path))
        predictions += 1
    report = dict(arms=arms, paired_tasks=rows, rescued=rescued, lost=lost,
                  model_predictions=predictions, decisions=decisions,
                  acknowledgements=acknowledgements, changed_targets=changed_targets)
    write(batch / 'analysis.json', report)
    print('原版 ApexNav: {}/30 ({:.2%}); 字典概率版: {}/30 ({:.2%}); 救回 {}, 损失 {}, 净增 {:+d}'.format(
        arms['baseline']['successes'], arms['baseline']['successes']/30,
        arms['dictionary']['successes'], arms['dictionary']['successes']/30,
        len(rescued), len(lost), len(rescued)-len(lost)), flush=True)
    print(f'模型预测 {predictions}; 决策 {decisions}; 规划确认 {acknowledgements}; 目标改变 {changed_targets}', flush=True)
    if not decisions or not acknowledgements:
        raise ValueError('Dictionary was not exercised/acknowledged; investigate before interpreting SR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'verify', 'compare'])
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--reference',type=Path)
    args = parser.parse_args()
    if args.operation=='prepare':prepare(args.batch,args.reference)
    else:globals()[args.operation](args.batch)
