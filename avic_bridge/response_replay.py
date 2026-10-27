"""Opt-in exact-input response freezing; separate from navigation caches."""
import copy
import hashlib
import json
from pathlib import Path
import uuid

from .protocol import validate_response


def canonical_request(payload):
    value=copy.deepcopy(payload)
    value.pop('request_id',None)
    if isinstance(value.get('episode'),dict):value['episode'].pop('generation_epoch',None)
    def remove_audit_ids(obj):
        if isinstance(obj,dict):
            for key in ['request_id','generation_epoch','epoch']:obj.pop(key,None)
            for child in obj.values():remove_audit_ids(child)
        elif isinstance(obj,list):
            for child in obj:remove_audit_ids(child)
    if 'alignment_audit' in value:remove_audit_ids(value['alignment_audit'])
    return value


def fingerprint(payload):
    episode=payload.get('episode') or {}
    if not episode.get('scene_id') or 'episode_id' not in episode:
        raise ValueError('Response replay requires explicit scene and episode identity')
    return hashlib.sha256(json.dumps(canonical_request(payload),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ResponseArchive:
    def __init__(self,archive=None,replay=None):
        self.archive=Path(archive) if archive else None
        self.replay=Path(replay) if replay else None
        self.index={};self.matches=0;self.misses=0
        if self.replay:
            for path in sorted(self.replay.glob('*.response.json')):
                try:
                    record=json.loads(path.read_text());key=fingerprint(record['request'])
                    validate_response(record['response'],record['request']['request_id'])
                    self.index.setdefault(key,[]).append((path,record))
                except Exception as exc:self.log('INELIGIBLE',reason=type(exc).__name__,path=str(path))

    def log(self,event,**values):
        print('[AVIC_RESPONSE_REPLAY] '+json.dumps(dict(event=event,matches=self.matches,misses=self.misses,**values)),flush=True)

    def lookup(self,payload):
        if not self.replay:return None
        try:key=fingerprint(payload)
        except (ValueError,TypeError) as exc:
            self.misses+=1;self.log('MISS',reason='input cannot be fingerprinted: '+type(exc).__name__);return None
        entries=self.index.get(key,[])
        if not entries:
            self.misses+=1;self.log('MISS',reason='no exact complete-input fingerprint',fingerprint=key,request_id=payload['request_id']);return None
        # Conflicting outputs for the same physical input must not be selected arbitrarily.
        normalized=[]
        for _,record in entries:
            response=copy.deepcopy(record['response']);response.pop('request_id',None)
            normalized.append(json.dumps(response,sort_keys=True,allow_nan=False))
        if len(set(normalized))!=1:
            self.misses+=1;self.log('MISS',reason='ambiguous archived responses',fingerprint=key);return None
        path,record=entries[0];response=copy.deepcopy(record['response']);response['request_id']=payload['request_id']
        validate_response(response,payload['request_id']);self.matches+=1
        self.log('MATCH',fingerprint=key,source=str(path),original_request_id=record['request']['request_id'],
                 current_request_id=payload['request_id'],original_epoch=record['request'].get('episode',{}).get('generation_epoch'),
                 current_epoch=payload.get('episode',{}).get('generation_epoch'),
                 candidate_plan_ids=[r.get('plan_id') for r in response.get('candidate_scores',[])],
                 scope='Frozen response experiment, NOT navigation cache hit')
        return response

    def save(self,payload,response):
        if not self.archive:return
        key=fingerprint(payload);validate_response(response,payload['request_id'])
        self.archive.mkdir(parents=True,exist_ok=True)
        manifest=self.archive/'manifest.json'
        if not manifest.exists():
            root=Path(__file__).resolve().parents[1]
            names=['avic_bridge/response_replay.py','avic_bridge/client.py','avic_bridge/controller.py',
                   'avic_bridge/service.py','avic_bridge/svc_adapter.py',
                   'src/planner/exploration_manager/src/exploration_fsm.cpp']
            names += [str(p.relative_to(root)) for p in sorted((root/'config').glob('*.yaml'))]
            manifest.write_text(json.dumps(dict(schema='exact_response_archive_v1',source_sha256={
                name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}),indent=2)+'\n')
        path=self.archive/(key+'_'+uuid.uuid4().hex+'.response.json')
        path.write_text(json.dumps(dict(request=payload,response=response),sort_keys=True,allow_nan=False)+'\n')
        self.log('ARCHIVE',fingerprint=key,path=str(path),request_id=payload['request_id'])
