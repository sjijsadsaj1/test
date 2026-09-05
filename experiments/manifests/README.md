# Evaluation manifests

- `hm3dv2_clear_targets_50_habitat.json` is the Habitat 0.3.1 runtime-ID
  manifest used for paired 50-episode diagnostics.
- `hm3dv2_clear_targets_top10_apexnav_spl.json` is a biased diagnostic subset
  selected from successful, high-SPL baseline episodes. It is useful for
  debugging but must not be reported as an unbiased evaluation set.

Generated results belong under `experiments/runs/` and are intentionally
excluded from Git.
