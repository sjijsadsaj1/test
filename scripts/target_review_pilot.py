#!/usr/bin/env python3
"""Prespecified 3-case four-arm feasibility study; no large-batch promotion."""
import argparse
import hashlib
import json
from pathlib import Path
from experiment_utils import ROOT, key, write, metrics
import experiment_utils as records

REFERENCE=ROOT/'experiments/runs/frontier_dictionary30/20261026_123344_323566'
KEYS=[('zsNo4HB9uLZ','12'),('X7HyMhZNoso','22'),('2azQ1b91cZZ','85')]


def prepare(b):
    m=json.loads((REFERENCE/'manifest.json').read_text());lookup={key(e):e for e in m['episodes']}
    m['episodes']=[lookup[k] for k in KEYS]
    assert len(m['episodes'])==3
    r=records.read_records(REFERENCE/'baseline/results');assert all(k in r for k in KEYS)
    b.mkdir(parents=True,exist_ok=False)
    write(b/'manifest.json',m)
    write(b/'conditions.json',dict(tasks=3,selection='two historic false STOPs + one successful chair control, fixed before new inference',
        arms=['baseline','dictionary','review','combined'],baseline=str(REFERENCE/'baseline'),
        baseline_outcomes={str(k):r[k] for k in KEYS},dictionary_calls=4,dictionary_cooldown=0,
        review_calls_per_episode=4,observation_actions_per_instance=4,one_scan_per_instance=True,
        instance_exclusion_steps=60,episode_step_budget=500,
        distinction='Real object review can veto original arrival STOP; dictionary conjectures never authorize STOP',
        warning='Development cases; dictionary budget differs from preceding 500-query batch. '
                'Original baseline is reused, current workspace arms compare complete systems, not pure module causality.'))
    paths=[p for p in (ROOT/'avic_bridge').glob('*.py')]+[ROOT/p for p in (
        'habitat_evaluation.py','scripts/experiment_utils.py','scripts/run_target_review_pilot.sh','scripts/target_review_pilot.py',
        'config/frontier_object_dictionary.json','devel/lib/exploration_manager/exploration_node',
        'devel/lib/libplan_env.so','src/planner/plan_env/src/object_map2d.cpp',
        'src/planner/plan_env/include/plan_env/object_map2d.h','src/planner/plan_env/srv/ReviewTarget.srv',
        'src/planner/exploration_manager/src/exploration_fsm.cpp',
        'src/planner/exploration_manager/src/exploration_manager.cpp')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        relative=p.relative_to(ROOT)
        if str(relative).startswith('devel/'):continue
        dest=b/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(b/'source_sha256.json',hashes)
    print('Prepared 3 paired cases; original baseline reused. New order: review -> dictionary -> combined.')


def verify(b):
    for p,h in json.loads((b/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,'source changed: '+p


def compare(b):
    verify(b);expected=set(KEYS);baseline=records.read_records(REFERENCE/'baseline/results')
    report={'baseline':dict(successes=sum(baseline[k]=='success' for k in expected),tasks=3)}
    for arm in ['review','dictionary','combined']:
        result=records.read_records(b/arm/'results')
        if set(result)!=expected:raise ValueError('Incomplete arm '+arm)
        events=[];blocked=set()
        for path in (b/arm/'logs').rglob('evaluation.log'):
            for line in path.read_text().splitlines():
                if line.startswith('[TARGET_REVIEW] '):events.append(json.loads(line.split(' ',1)[1]))
                if line.startswith('[TARGET_REVIEW_COUNTERFACTUAL] '):
                    d=json.loads(line.split(' ',1)[1])
                    if d['would_fail_stop']:
                        parts=d['episode'].rsplit(':',2);blocked.add((Path(parts[-2]).stem,parts[-1]))
        rescued=[k for k in expected if baseline[k]!='success' and result[k]=='success']
        lost=[k for k in expected if baseline[k]=='success' and result[k]!='success']
        good=sum(result.get(k)=='success' for k in blocked)
        report[arm]=dict(tasks=3,successes=sum(v=='success' for v in result.values()),rescued=rescued,lost=lost,
            net=len(rescued)-len(lost),metrics=metrics(b/arm/'results'),
            extra_actions=sum(e.get('event') in ('observe','reject') for e in events),
            review_seconds=sum(e.get('elapsed',0) for e in events),
            verdicts=sum(e.get('event')=='verdict' for e in events),
            blocked_wrong_stop_tasks=len(blocked),successful_after_block=good,
            success_after_block_rate=good/len(blocked) if blocked else None,
            outcomes={str(k):v for k,v in result.items()})
    write(b/'analysis.json',report)
    for arm,r in report.items():print(arm, str(r['successes'])+'/3', 'net',r.get('net','reference'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','verify','compare']);p.add_argument('--batch',type=Path,required=True)
    a=p.parse_args();globals()[a.op](a.batch)
