# CirScore

CirScore is a paper-aligned implementation of executable program induction for action quality assessment. It separates three responsibilities:

1. an LLM may propose a structured relationship between task concepts and registered evidence sources;
2. training-only out-of-fold evidence selects the executable program;
3. test labels are opened only after predictions have been frozen.

The repository contains implementation code only. It does not include pretrained weights, datasets, API credentials, or reported experimental results.

## Method overview

CirScore operates on a base prediction and three registered evidence sources: `i3d`, `swin`, and `pose`.

```text
training data
    -> out-of-fold source predictions
    -> language-induced and non-language candidate programs
    -> training-side program selection
    -> bounded correction fitting
    -> frozen test predictions
    -> final evaluation with separately stored test labels
```

Each executable candidate specifies:

- a subset of registered sources;
- normalized source weights;
- a rank-composition temperature;
- a base/source mixing coefficient `beta`;
- whether bounded correction is enabled;
- correction coordinate, bound, and strength.

LLM output does not directly determine the final program. It constrains the language-induced candidate family, while the final candidate is selected using training-only out-of-fold predictions.

## Repository structure

```text
build_train_bundle.py       Build training-only OOF predictions
build_diagnostics.py        Derive LLM-facing training-only diagnostics
compile_program.py          Compile language and non-language candidates
select_program.py           Select both program families on training folds
program_execution.py        Rank composition and bounded correction
freeze_predictions.py       Refit and freeze label-free test predictions
score_predictions.py        Open test labels and compute final metrics
core_model.py               Base/source prediction model
data_io.py                  Strict data loading and ID alignment
evidence.py                 Training-side evidence construction
contract.py                 Typed executable-program contract
privacy_check.py            Detect private paths and unsafe embedded values
config/
  task.txt                  Anonymous induction-task template
  evidence_specification.json  Registered evidence-source specification
induction/
  build_request.py          Build the structured induction request
  induce.py                 Run an induction backend
  backends.py               External-command and local backends
  schema.py                 Request and response validation
  repair.py                 JSON-only response extraction
  archive.py                Archive request, response, and metadata
  model_catalog.py          Declared model labels
  models.example.json       External connector configuration template
  run_sensitivity.py        Repeated induction runner
  summarize_sensitivity.py  Aggregate induction stability records
```

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- scikit-learn
- PyTorch
- `transformers` only when using the optional local-model backend

The external-command backend does not prescribe a provider SDK. The user supplies a command that reads the induction prompt from standard input and writes either raw response text or a JSON response envelope to standard output.

## Data contract

All IDs are treated as strings and must be unique.

### Split manifest

The split file is JSON:

```json
{
  "dataset_size": 100,
  "train_ids": ["sample-001"],
  "test_ids": ["sample-081"],
  "source_sha256": "sha256-of-the-authoritative-split-source"
}
```

The implementation enforces a non-overlapping 80/20 train/test split.

### Training labels

The training-label NPZ file must contain:

```text
sample_id    [N]
target       [N]
difficulty   [N]
```

### Evaluation metadata

The evaluation-metadata NPZ file must contain:

```text
sample_id    [M]
difficulty   [M]
```

It must not contain `target` or `final_score`.

### Evidence tokens

Each source NPZ file must contain:

```text
sample_id    [N + M]
tokens       [N + M, ...]
```

The same format is used for the I3D, Swin, and pose sources. Feature dimensions may differ across sources; the base model infers its I3D input dimension at runtime.

### Test labels

The physically separate test-label NPZ file must contain:

```text
sample_id    [M]
target       [M]
```

This file is passed only to the final scoring command.

## End-to-end workflow

The following commands use repository-relative example paths. Replace them with paths appropriate to the local data layout.

### 1. Build the training-only bundle

```bash
python build_train_bundle.py \
  --split data/split.json \
  --train-labels data/train_labels.npz \
  --i3d data/i3d_tokens.npz \
  --swin data/swin_tokens.npz \
  --pose data/pose_tokens.npz \
  --folds 5 \
  --device cuda:0 \
  --output artifacts/train_bundle.npz
```

### 2. Build training-only diagnostics

```bash
python build_diagnostics.py \
  --bundle artifacts/train_bundle.npz \
  --output artifacts/training_diagnostics.json
```

This stage derives fold-level base and source statistics from training OOF evidence only. It does not accept evaluation metadata or test labels.

### 3. Build the induction request

```bash
python induction/build_request.py \
  --task config/task.txt \
  --specification config/evidence_specification.json \
  --diagnostics artifacts/training_diagnostics.json \
  --output artifacts/induction_request.json
```

The repository includes anonymous templates for the task description and evidence specification. The request contains only these supplied files, training diagnostics, and the required output schema.

### 4. Run structured induction

External command:

```bash
python induction/induce.py \
  --request artifacts/induction_request.json \
  --backend command \
  --command "python connector.py" \
  --model-name "DeepSeek-V4-Pro" \
  --output artifacts/induction.json
```

Local model:

```bash
python induction/induce.py \
  --request artifacts/induction_request.json \
  --backend local \
  --model-path models/local-model \
  --output artifacts/induction.json
```

No API key or provider endpoint is included in this repository. External connectors should read credentials from their execution environment and must not write credentials into induction archives.

### 5. Compile candidate programs

```bash
python compile_program.py \
  --induction artifacts/induction.json \
  --output artifacts/program_family.json
```

This creates two distinct candidate families:

- language-induced candidates constrained by the validated concept/source bindings;
- non-language candidates constructed without the language-induced bindings.

### 6. Select programs using training evidence

```bash
python select_program.py \
  --bundle artifacts/train_bundle.npz \
  --family artifacts/program_family.json \
  --output artifacts/selection.json
```

Language and non-language candidates are selected independently using:

```text
argmax (mean fold SRCC, worst-fold gain, deterministic tie-break)
```

When correction is enabled, it is cross-fitted: a held-out fold is never corrected by a model fitted on that fold's labels.

### 7. Freeze test predictions

```bash
python freeze_predictions.py \
  --split data/split.json \
  --train-labels data/train_labels.npz \
  --eval-metadata data/eval_metadata.npz \
  --i3d data/i3d_tokens.npz \
  --swin data/swin_tokens.npz \
  --pose data/pose_tokens.npz \
  --family artifacts/program_family.json \
  --selection artifacts/selection.json \
  --device cuda:0 \
  --output artifacts/frozen_predictions.npz \
  --audit artifacts/freeze_audit.json
```

This stage does not accept a test-label argument. It records hashes for the split, selected program, and frozen predictions.

### 8. Score the frozen predictions

```bash
python score_predictions.py \
  --split data/split.json \
  --predictions artifacts/frozen_predictions.npz \
  --audit artifacts/freeze_audit.json \
  --test-labels data/test_labels.npz \
  --output artifacts/metrics.json
```

The scorer verifies the freeze audit and file hashes before reading the test labels. It reports SRCC, NMSE, MAE, and RMSE.

## Ablation definitions

- `CirScore`: selected language-induced program with its selected correction setting.
- `w_o_LLM`: best non-language candidate selected with the same training-only rule.
- `w_o_NS`: base prediction without the neural-symbolic program composition.
- `w_o_Correction`: the selected CirScore program with correction disabled.
- `worst_legal`: the lowest-ranked legal candidate under the fixed selection evidence.
- random legal programs: distinct legal candidates used to compute the random-program reference distribution.

If the selected CirScore program does not enable correction, `w_o_Correction` is marked not applicable rather than reported as an artificial duplicate.

## Reproducibility and leakage controls

- Fixed random seeds are used for folds, model fitting, candidate construction, and random-program sampling.
- Candidate selection uses training-only OOF predictions.
- Evaluation metadata is rejected if it contains labels.
- Test labels are accepted only by the final scoring stage.
- Sample IDs are aligned explicitly; missing, duplicate, or overlapping IDs raise errors.
- Frozen artifacts are protected by SHA-256 checks.
- Invalid induction output raises an error; JSON extraction never invents missing semantic fields.
- Missing inputs and incomplete predictions fail closed instead of producing substitute values.

Run the source privacy check with:

```bash
python privacy_check.py
```

## Model labels and external connectors

The declared model labels are:

- DeepSeek-V4-Pro (default)
- DeepSeek-V4-Flash
- GPT-6 Astra
- Claude Sonnet 5
- Gemini 3.8 Flash
- Qwen3.8-Max

These entries are configuration labels, not bundled clients or claims of provider availability. The example model configuration intentionally leaves all external commands blank. Users are responsible for supplying an authorized connector and valid credentials.

## Evidence boundary

This codebase defines an executable and auditable evaluation protocol. Metric values are produced only after the user supplies the required data, features, frozen predictions, and test labels. The repository itself does not provide or imply empirical performance values.
