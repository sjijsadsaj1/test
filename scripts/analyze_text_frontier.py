#!/usr/bin/env python3
"""Report execution evidence separately from predictions and completed SR."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import experiment_utils as records


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch
    counts=Counter();decisions=[];episode=None;latest=None
    for path in (b/'text/logs').rglob('evaluation.log'):
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith('[SYNC_PLAN] '):
                episode=json.loads(line.split(' ',1)[1])['episode']
            if line.startswith('[TEXT_FRONTIER] '):
                latest=dict(json.loads(line.split(' ',1)[1]),episode=episode)
                decisions.append(latest);counts['valid_decisions']+=1
                ps=list(latest['beliefs'].values())
                latest['probability_spread']=max(ps)-min(ps)
                if latest['probability_spread']<=1e-8:counts['equal_beliefs']+=1
            if line.startswith('[AVIC_FRONTIER_ACK] ') and latest:
                ack=json.loads(line.split(' ',1)[1])
                if ack['query_id']!=latest['query_id']:raise ValueError('ACK query mismatch')
                latest['ack']=ack
                if ack['applied']:
                    counts['actual_target_changes']+=1
                    counts['changes_with_belief_difference' if latest['probability_spread']>1e-8 else 'changes_on_tie']+=1
    results=[];times=[];scene_retries=goal_retries=0
    for path in (b/'text_predictions').glob('*/result.json'):
        d=json.loads(path.read_text());i=json.loads((path.parent/'input.json').read_text())
        seconds=d['scene_seconds']+d['goal_seconds'];times.append(seconds)
        scene_retries+=d.get('scene_attempts',1)-1;goal_retries+=d.get('goal_attempts',1)-1
        results.append(dict(episode=i['episode'],seconds=seconds,candidates=d['candidates'],directory=str(path.parent)))
    arms={arm:records.read_records(b/arm/'results') for arm in ('off','text')}
    expected=json.loads((b/'manifest.json').read_text())['episodes']
    complete=all(len(v)==len(expected) for v in arms.values())
    summary=dict(complete=complete,counts=dict(counts),scene_retries=scene_retries,goal_retries=goal_retries,
        valid_inference_seconds=dict(n=len(times),median=statistics.median(times) if times else None,
            total=sum(times)),generated_images=0,
        success={arm:sum(v=='success' for v in rs.values()) for arm,rs in arms.items()},
        completed={arm:len(rs) for arm,rs in arms.items()},
        caveat='A target change on equal probabilities is a planner tie-break effect, not language semantic evidence.')
    (b/'text_mechanism_analysis.json').write_text(json.dumps(dict(summary=summary,decisions=decisions,inferences=results),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
