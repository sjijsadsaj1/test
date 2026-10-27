#!/usr/bin/env python3
"""Run unmodified upstream main() with only task selection/output adaptation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import signal
import sys


def select(episodes, manifest, maximum=0):
    entries = json.loads(Path(manifest).read_text())["episodes"]
    available = {(Path(e.scene_id).name.split(".")[0], str(e.episode_id), e.object_category): e for e in episodes}
    keys = [(e["scene_key"], str(e["episode_id"]), e["object_category"]) for e in entries]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate manifest")
    missing = set(keys) - available.keys()
    if missing:
        raise ValueError("Manifest episodes missing from runtime dataset: " + repr(sorted(missing)))
    selected = [available[k] for k in keys]
    return selected[:maximum] if maximum else selected


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--dataset", required=True, choices=["mp3d", "hm3dv1", "hm3dv2"])
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--video", action="store_true", help="Save upstream evaluation video; no policy changes")
    args = parser.parse_args()
    if args.max_episodes < 0:
        parser.error("--max-episodes must be nonnegative")
    workspace = Path(args.workspace).resolve()
    sys.path.insert(0, str(workspace))
    upstream = importlib.import_module("habitat_evaluation")
    if Path(upstream.__file__).resolve() != workspace / "habitat_evaluation.py":
        raise RuntimeError("Refusing a non-upstream evaluator")
    origins = {}
    for module_name in ('habitat_evaluation', 'habitat2ros.habitat_publisher', 'params',
                        'basic_utils.failure_check.failure_check', 'vlm.utils.get_object_utils',
                        'vlm.utils.get_itm_message', 'llm.answer_reader.answer_reader'):
        module = sys.modules[module_name]
        path = Path(module.__file__).resolve()
        if workspace not in path.parents:
            raise RuntimeError('Baseline imported non-isolated code: ' + str(path))
        origins[module_name] = str(path)
    print('[UPSTREAM_IMPORTS] ' + json.dumps(origins, sort_keys=True), flush=True)
    import habitat
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import rospy

    output = Path(args.result_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "record.txt").exists() or (output / "continue.txt").exists():
        raise RuntimeError("Never resume/overwrite a baseline implicitly")
    (output / 'baseline_imports.json').write_text(json.dumps(origins, indent=2) + '\n')
    with initialize_config_dir(version_base=None, config_dir=str(workspace / "config")):
        cfg = compose(config_name="habitat_eval_" + args.dataset, overrides=[
            "test_epi_num=-1", "need_video=" + str(args.video).lower(), "habitat.environment.iterator_options.shuffle=false",
        ])
    cfg.video_output_path = str(output)
    # Both modes use the same real perception weights/commonsense answer files
    # in the main project working directory, not archived model copies.
    (output / "effective_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    original_env = habitat.Env
    env_holder = []

    def selected_env(*a, **kw):
        env = original_env(*a, **kw)
        env.episodes = select(env.episodes, args.episode_manifest, args.max_episodes)
        # Habitat's episodes setter rebuilds the iterator but does not update
        # this cached count; upstream main() uses it as its loop bound.
        env.number_of_episodes = len(env.episodes)
        if os.environ.get('APEXNAV_STOP_AUDIT_DIR'):
            if len(env.episodes) != 1:
                raise ValueError('STOP diagnostic is restricted to one episode')
            from stop_evidence import install
            install(env, upstream, cfg, os.environ['APEXNAV_STOP_AUDIT_DIR'])
        env_holder.append(env)
        print(f"[UPSTREAM_BASELINE] Selected {len(env.episodes)} fixed runtime episodes", flush=True)
        return env

    habitat.Env = selected_env
    rospy.init_node("habitat_eval_node", anonymous=True)
    signal.signal(signal.SIGINT, upstream.signal_handler)
    try:
        upstream.main(cfg)
    except StopIteration:
        # Upstream advances the iterator once after the last recorded episode.
        # Accept ONLY this terminal case after exact completion, never mid-run.
        record = output / "record.txt"
        numbers = re.findall(r"No\.(\d+) task is finished", record.read_text() if record.exists() else "")
        expected = len(env_holder[0].episodes) if env_holder else 0
        if not expected or sorted(map(int, numbers)) != list(range(1, expected + 1)):
            raise
        print("[UPSTREAM_BASELINE] All tasks recorded; handled upstream final iterator exhaustion", flush=True)
    finally:
        habitat.Env = original_env
        for env in env_holder:
            env.close()


if __name__ == "__main__":
    run()
