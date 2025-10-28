# Evaluation Pipeline

This directory hosts the Hydra-driven evaluation suite for computing downstream metrics on
pretrained Hescape checkpoints. The default configuration focuses on the BRCA subtyping task
with 10-fold cross-validation and reports AUROC/AUPRC for both the shared CLIP embedding and a
Gigapath-only baseline.

## Quickstart

```bash
python experiments/evaluation/run.py
```

By default this performs both stages (embedding + evaluation). Outputs land in
`experiments/evaluation/outputs/<timestamp>/`; embeddings are stored under the `embeddings/`
subfolder while metrics are written to `results.json`.

### Required assets

- **Checkpoint:** Update `clip_model.checkpoint_path` in `config.yaml` if you want to evaluate a
  different Lightning checkpoint.
- **CpGPT resources:** The defaults assume the pre-packaged CpGPT dependencies living under
  `data/cpgpt_files/`.

### Two-step workflow

For faster iteration on downstream tasks, split the run into two stages:

1. **Embed once** (choose a persistent location so you can reuse the files):

   ```bash
   python experiments/evaluation/run.py mode=embed output_dir=/tmp/hescape_eval
   ```

2. **Evaluate** using the cached embeddings (reuse the same `output_dir` or point
   `embedding.dir` directly to the folder created in step 1):

   ```bash
   python experiments/evaluation/run.py mode=eval embedding.dir=/tmp/hescape_eval/embeddings
   ```

Set `embedding.force=true` if you need to regenerate embeddings after producing a new checkpoint.

## Customising Tasks

Add new entries to the `tasks` section in `config.yaml`. For instance, to create a different
binary classification task you can append:

```yaml
tasks:
  my_task:
    type: binary_classification
    label_field: diagnosis.some_other_field
    positive_label: POS_CODE
    negative_label: NEG_CODE
    folds: 5
```

Then register a job in `evaluation.jobs`, mapping the task to an embedder configuration:

```yaml
evaluation:
  jobs:
    - name: clip_shared_my_task
      embedder: fusion
      task: my_task
```

Embedders live under `embedders`. You can duplicate `fusion` or `image` and tweak projection /
normalisation settings, or create a new entry with `type: clip_fusion` (shared embeddings) or
`type: clip_image` (image branch only). If you need to reproduce experiments that only fine-tuned
one branch, set `clip_model.image_encoder.finetune_tile` or `finetune_slide` to `false`; the model
instantiation honours those switches while still restoring LoRA adapters on the slide encoder.

Embeddings are cached by sample ID only, so remember to re-run the embedding stage whenever you
change the model weights.

## Output Layout

Each evaluation run produces:

- `results.json` – summary metrics and fold-level scores for every job.
- `hydra/` – Hydra bookkeeping (config and overrides).

The JSON structure is intentionally simple to ease downstream aggregation or reporting.
