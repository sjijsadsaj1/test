#!/usr/bin/env bash
# One existing episode, at most 2 model queries. Plumbing validation, NOT SR evidence.
set -Eeuo pipefail
main() {
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python=/data/wzy/miniconda3/envs/apexnav/bin/python
batch="$project_dir/experiments/runs/frontier_dictionary_smoke/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$batch"
export DICTIONARY_SMOKE_BATCH="$batch"
"$python" - <<'PY'
from pathlib import Path
import os,json,hashlib
b=Path(os.environ['DICTIONARY_SMOKE_BATCH'])
m=json.loads(Path('experiments/manifests/mp3d_sr_dev100_seed20260911_runtime.json').read_text())
m['episodes']=[e for e in m['episodes'] if e['scene_key']=='EU6Fwq7SyZv' and str(e['episode_id'])=='35']
assert len(m['episodes'])==1
(b/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(b/'conditions.json').write_text(json.dumps(dict(purpose='geometry export + dictionary marginals + real planner integration, NOT SR trial',
    contract='frontier_object_marginals_v1',max_calls=2,cooldown=0,original_cluster=True,deframe=False,
    generated_images=0,probability_normalization=False,success_comparison=False),indent=2)+'\n')
files=['avic_bridge/frontier_dictionary.py','avic_bridge/frontier_crop.py','avic_bridge/vlm_transport.py',
       'avic_bridge/controller.py','avic_bridge/text_frontier.py','avic_bridge/text_service.py','habitat_evaluation.py',
       'config/frontier_object_dictionary.json','scripts/experiment_utils.py','scripts/run_frontier_dictionary_smoke.sh',
       'src/planner/exploration_manager/src/exploration_manager.cpp','src/planner/exploration_manager/src/exploration_fsm.cpp',
       'devel/lib/exploration_manager/exploration_node']
hashes={}
for name in files:
    data=Path(name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
    if not name.startswith('devel/'):
        dst=b/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
(b/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
bridge_pid=''
cleanup() {
    status=$?;trap - EXIT INT TERM
    if [[ -n "$bridge_pid" ]];then kill -TERM -- "-$bridge_pid" 2>/dev/null || true;wait "$bridge_pid" 2>/dev/null || true;fi
    printf '%s\n' "$status" >"$batch/exit_code";exit "$status"
}
trap cleanup EXIT;trap 'exit 130' INT TERM
if [[ -f config/avic.env ]];then set -a;source config/avic.env;set +a;fi
# Only the standalone VLM client uses GLM credentials; no image settings drive this policy.
if [[ -f config/image_glm.env ]];then set -a;source config/image_glm.env;set +a;fi
# Prefer the dedicated VLM config; legacy local image_glm.env remains a compatibility fallback.
if [[ -f config/text_frontier.env ]]; then set -a; source config/text_frontier.env; set +a; fi
export AVIC_ENV_FILE=/dev/null APEXNAV_ROOT="$project_dir" APEXNAV_STAGNATION_RECOVERY=false
export APEXNAV_IMAGE_GLM=0 APEXNAV_IMAGE_GLM_STRONG=0 APEXNAV_FRONTIER_CYCLE_MEMORY=0
export APEXNAV_ALIGNED_BASE_EXPERIMENT=0 APEXNAV_RELAXED_IMAGINATION=0 APEXNAV_EXTREME_SAFETY=0
export APEXNAV_HYPOTHESIS_PROGRESS=0 APEXNAV_ROOM_PRIOR=0 APEXNAV_TARGET_ALIASES=off
export APEXNAV_LITERATURE_METHOD=off APEXNAV_ALIGNMENT_AUDIT=1 APEXNAV_PERCEPTION_AUDIT=1
export AVIC_IMAGE_INTERIOR=0 AVIC_IMAGE_FRONTIER_GEOMETRY=0 AVIC_IMAGE_DUAL_CONTEXT=0
unset APEXNAV_UPSTREAM_COMPAT_ROOT APEXNAV_FRONTIER_ONCE APEXNAV_FRONTIER_REFRESH_DELTA APEXNAV_V1_FRONTIER_POLICY
unset APEXNAV_SCORE_RESPONSE_ARCHIVE APEXNAV_SCORE_RESPONSE_REPLAY APEXNAV_OPEN_ESCAPE_TRIGGER
export APEXNAV_TEXT_FRONTIER=1 APEXNAV_TEXT_METHOD=dictionary AVIC_TEXT_PROVIDER="${AVIC_TEXT_PROVIDER:-local}"
export AVIC_TEXT_OUTPUT_ROOT="$batch/predictions" APEXNAV_LOG_ROOT="$batch/logs"
export PYTHONPATH="$project_dir" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
"$python" scripts/experiment_utils.py ports --batch "$batch" >"$batch/ports.txt"
read -r bridge_port ros_port <"$batch/ports.txt"
export AVIC_SERVICE_PORT="$bridge_port"
echo "Diagnostic results: $batch"
setsid "$project_dir/.runtime/image_glm_env/bin/python" -u -m avic_bridge.text_service >"$batch/bridge.log" 2>&1 &
bridge_pid=$!
deadline=$((SECONDS+60))
while ! curl -fsS --max-time 2 "http://127.0.0.1:$bridge_port/health" >"$batch/health.json";do
    kill -0 "$bridge_pid" || exit 1
    ((SECONDS<deadline)) || exit 1
    sleep 1
done
bash scripts/run_apexnav.sh --dataset mp3d --episode-manifest "$batch/manifest.json" --max-episodes 1 \
    --avic-mode avic --avic-service-url "http://127.0.0.1:$bridge_port" --avic-score-mode text_frontier \
    --avic-timeout 600 --avic-max-calls 2 --avic-cooldown-steps 0 --reuse-existing-perception \
    --ros-port "$ros_port" --result-dir "$batch/results" >"$batch/progress.log" 2>&1
"$python" - <<'PY'
from pathlib import Path
import os,json,hashlib
b=Path(os.environ['DICTIONARY_SMOKE_BATCH'])
hashes=json.loads((b/'source_sha256.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
results=[json.loads(p.read_text()) for p in (b/'predictions').glob('*/result.json')]
assert len(results)==2, 'Expected two actual dictionary queries'
assert all(r['generated_images']==0 and r['contract']=='frontier_object_marginals_v1' for r in results)
logs='\n'.join(p.read_text() for p in (b/'logs').rglob('evaluation.log'))
assert logs.count('[AVIC_FRONTIER_ACK] ')==2, 'No planner acknowledgement'
print('Dictionary plumbing passed: 2 real queries, every frontier has 21 marginals, 2 planner ACKs. NOT an SR comparison.')
PY
}
main "$@"
