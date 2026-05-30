# Three-Level Mental Model Network

A neural architecture for concept learning and controlled adaptation, structured
according to the three-level mental-model framework (use → adaptation → control
of adaptation) from Bhalwankar & Treur (2021), implemented for deep learning and
validated on a continual-learning benchmark.

## Thesis claim

> A neural architecture with explicit separation between concept *use*, concept
> *adaptation*, and *control of adaptation* outperforms flat architectures and
> standard continual-learning baselines on tasks requiring concept reuse,
> controlled updating, and generalisation under distribution shift.

## The three levels

| Level | Role | Implementation | File |
|-------|------|----------------|------|
| **1 — Use** | The mental model that is applied to the current input | Conv encoder + Slot Attention concept layer + linear head | `src/levels/level1_concepts.py` |
| **2 — Adaptation** | First-order change of the model in response to experience | Hypernetwork generating a *gated weight delta* for the head, conditioned on the model's own concept activations + prediction error | `src/levels/level2_adaptation.py` |
| **3 — Control** | Second-order, self-modeling regulation of *when/how much* to adapt | Handcrafted uncertainty gate (Paper 1) or a learned self-model gate (Paper 2) | `src/levels/level3_control.py` |

The distinctive contribution is **Level 3**: instead of a hand-designed
consolidation rule (as in EWC), the adaptation gate is driven by the system's
own epistemic state — the self-modeling network idea made architectural.

## Install

```bash
uv sync
```

For the optional swap-ins (hypnettorch, avalanche-lib):

```bash
uv sync --extra extras
```

> `uv sync` creates `.venv` automatically. Prefix any command with `uv run` to use it, or activate with `source .venv/bin/activate`.

## Quick sanity check (CPU, no download)

```bash
uv run python tests/test_forward.py
```

## Full experiment (Split-CIFAR-100)

```bash
# fast debug run
uv run python scripts/run_experiment.py --tasks 3 --epochs 1 --device cpu

# full run (GPU recommended)
uv run python scripts/run_experiment.py --tasks 10 --epochs 5 --device cuda
```

Outputs `results/results.csv` with ACC / BWT / FWT for each configuration:

```
finetune            (lower bound, catastrophic forgetting)
ewc                 (Kirkpatrick et al. 2017)
L1+L2 (no control)  (adaptation without Level-3 gating)
L1+L2+L3 handcrafted
L1+L2+L3 learned    (ours)
```

The ablation ladder (`finetune < ewc < L1+L2 < L1+L2+L3`) is the core results
table for the paper, isolating the contribution of each level.

## Metrics

- **ACC** — average accuracy across all tasks after training on all tasks
- **BWT** — backward transfer (negative ⇒ forgetting)
- **FWT** — forward transfer above chance
- *(qualitative)* concept interpretability via the Level-1 slots — where the
  cognitive grounding differentiates this from a pure engineering baseline.

## Repository layout

```
mental-model-dl/
├── src/
│   ├── levels/
│   │   ├── level1_concepts.py    # encoder + slot attention concept layer
│   │   ├── level2_adaptation.py  # hypernetwork adaptation
│   │   └── level3_control.py     # handcrafted + learned control gates
│   ├── model.py                  # ThreeLevelNet (combines all three levels)
│   ├── baselines.py              # finetune + EWC
│   ├── data.py                   # split-CIFAR-100 loader
│   ├── metrics.py                # ACC / BWT / FWT
│   └── train.py                  # continual-learning loop
├── scripts/run_experiment.py     # baselines vs. ablations
├── tests/test_forward.py         # forward/backward smoke tests
├── configs/default.yaml
└── requirements.txt
```

## Design notes & known rough edges

- **Self-contained on purpose.** Slot Attention, the hypernetwork, and the
  split benchmark are implemented directly (not pulled from
  `slot-attention` / `hypnettorch` / `avalanche`) so every mechanism is
  transparent and yours to modify. Swap-in points are marked in code comments.
- **Zero-init adaptation.** The hypernetwork's final layer is zero-initialised so
  early-training adaptation is a no-op — this is the single most important trick
  for training stability. Verified by `test_adaptation_toggle_changes_output`.
- **What needs your attention next:**
  1. Hyperparameters (`delta_scale`, `gate_threshold`, `ewc_lambda`, lr schedule)
     are sensible defaults, not tuned — expect to sweep them.
  2. The head-delta mechanism adapts only the classifier head. Extending Level 2
     to also modulate slot-attention parameters is the natural next experiment.
  3. Class-incremental (no task ID at test time) is harder than the
     task-incremental eval used here; report both.
  4. The learned Level-3 gate can collapse to always-on/always-off; consider a
     small entropy/sparsity regulariser on the gate.

## Reference

Bhalwankar, R. & Treur, J. (2021). *Modeling learner-controlled mental model
learning processes by a second-order adaptive network model.* PLoS ONE 16(8).
