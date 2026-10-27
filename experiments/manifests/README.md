# Evaluation manifests

- `mp3d_sr_dev100_seed20260911_runtime.json` is the frozen SR development set:
  100 uniform random tasks (seed 20260911), all 11 MP3D validation scenes,
  16 raw target categories. Excludes the previous three random-50 samples and
  initial single-scene 50 (195 unique tasks total). No outcome filtering.
  Do not redraw it after observing results. Repeated tuning makes this a
  development set, not an untouched test set. See `docs/FIXED100_SR.zh-CN.md`.
- `hm3dv2_clear_targets_50_habitat.json` is the Habitat 0.3.1 runtime-ID
  manifest used for paired 50-episode diagnostics.
- `hm3dv2_clear_targets_top10_apexnav_spl.json` is a biased diagnostic subset
  selected from successful, high-SPL baseline episodes. It is useful for
  debugging but must not be reported as an unbiased evaluation set.
- `mp3d_random50_seed20260906_habitat.json` is a reproducible uniform random
  sample without replacement from all 2,195 MP3D v1 validation episodes. It
  uses seed `20260906` and covers all 11 validation scenes and 16 target
  categories. ApexNav and AVIC must both use this manifest for paired results.
- `mp3d_random50_seed20260907_disjoint_runtime.json` is the corrected,
  reproducible uniform random sample from the remaining validation episodes
  after excluding the seed-20260906 sample. It has no episode overlap with
  that prior 50-task set and uses Habitat runtime per-scene episode IDs.
- `mp3d_random50_seed20260909_disjoint_runtime.json` is a third reproducible
  random sample. It excludes both earlier paired 50-episode samples and uses
  the same Habitat runtime per-scene episode-ID convention.
- `mp3d_random50_seed20260907_disjoint_habitat.json` is superseded: it used
  raw source-file IDs, which Habitat remaps during loading, and must not be run.

Generated results belong under `experiments/runs/` and are intentionally
excluded from Git.
