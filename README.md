# A Multi-Stakeholder Decision-Support Framework

Reframes automated decision-making as a **collective-choice problem**: stakeholder
preferences become endogenous reward signals that drive both the per-row action and
the cross-actor aggregation, so the normative trade-offs that conventional pipelines
hide become inspectable.

This repository accompanies the paper *"A Multi-Stakeholder Framework for
Automated Decision-Support Systems"*.

---

## Interactive dashboard (no install required)

A read-only Streamlit instance is publicly hosted at:

**[https://accountability-dashboard.streamlit.app/](https://accountability-dashboard.streamlit.app/)**

It lets you pick a reward variant × predictive model combination, adjust per-metric
weights and watch the ranking update, explore the two-metric trade-off table, and
inspect the per-stakeholder reward formulas and compromise-rule definitions. All
values are aggregated across seeds (mean ± SEM). The dashboard is scoped to the
**lending** use case; the health scenario is fully supported by the pipeline but not
exposed in the dashboard.

---

## What the framework does

Conventional ADM pipelines collapse heterogeneous stakeholder objectives into a
single optimization target. This framework keeps the layers separate:

1. **Preference elicitation** — each stakeholder $i$ has its own reward function $r_i$
   (simulated here, no real-world elicitation).
2. **Per-stakeholder regression** — supervised learning of $\hat E_{i,a}$, the expected
   reward for each context-action pair, per stakeholder.
3. **Compromise aggregation** — a score map $\Phi_j$ aggregates the expected-reward
   tensor across stakeholders (Maximin, Nash Bargaining, Kalai-Smorodinsky, Nash Social
   Welfare, Compromise Programming, Proportional Fairness).
4. **Post-hoc weighted ranking** — a transparent weighted sum over evaluation metrics
   selects the best compromise rule under user-chosen normative weights.

The reward, decision, and aggregation layers are decoupled by design; the dashboard
makes that decoupling inspectable.

---

## Repository layout

```
.
├── main.py                        # Hydra entry-point: trains models, writes run artifacts
├── dashboard.py                   # Streamlit accountability dashboard (lending)
├── plots_paper.ipynb              # Notebook producing the radar / sensitivity figures
├── ablate_weights.py              # Weight sweeps: pairwise, ternary, Dirichlet
├── plot_ablation.py               # Per-bucket plots from an ablate_weights CSV
├── plot_compare_ablations.py      # Overlay several buckets (pairwise / ternary / Dirichlet)
├── plot_pareto.py                 # Pareto frontier (single- or multi-bucket)
├── plot_combined_pareto_ternary.py# Combined Pareto + ternary figure (shared legend)
├── plot_stacked_dirichlet.py      # Stacked per-bucket Dirichlet panels
├── plot_with_baselines.py         # Static radar + ranking for one metrics CSV
├── run_all.ps1                    # Full reproducibility recipe (Windows)
├── run_sensitivity.ps1            # Sensitivity training sweeps (Windows)
├── conf/                          # Hydra configs
│   ├── config.yaml
│   └── use_case/{lending,lending_knn,health}.yaml
├── src/                           # Pipeline + preprocessing
├── utils/                         # Models, decisions, metrics, rewards, ranking
├── ablations/                     # Weight-sweep CSVs (created by ablate_weights.py)
├── figs/                          # Rendered figures
├── results/                       # Per-run artifacts (created by main.py)
├── requirements.txt               # Dashboard deps (installed by Streamlit Cloud)
├── requirements-training.txt      # Full training-pipeline deps
└── .streamlit/config.toml         # Dashboard theme + server config
```

---

## Setup

```bash
# Python 3.11 recommended
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-training.txt
```

`requirements-training.txt` covers training and every figure. The lightweight
`requirements.txt` holds only the dashboard's dependencies and is the file Streamlit
Community Cloud installs automatically.

### Data

Two public datasets ship with the repo under `data/`:

- `data/lending_club_data.csv` — Lending Club credit-decision CSV with `Action`,
  `Outcome`, `Applicant_Type`, `Loan_Amount`, `Interest_Rate`, and demographic/financial
  features.
- `data/health/ihdp_npci_1.csv` — Infant Health and Development Program causal-inference
  benchmark (Hill, 2011), first realization, 25 features, treatment `Action ∈ {A, C}`.

Each use-case YAML in `conf/use_case/` references its dataset via a repo-relative
`data_path`. Relative `data_path` and `result_path` values resolve against the repo
root, so runs work regardless of the working directory.

---

## Reproducing the experiments

The framework uses [Hydra](https://hydra.cc/); any parameter can be overridden on the
command line.

### One run

```bash
python main.py use_case=lending sample_size=10000 cv_splits=5 seed=111
```

This writes `results/lending/run_<timestamp>_seed111_Acc_*_Fair_*/` containing:

| File | Contents |
|---|---|
| `final_ranked_decision_metrics.csv` | One row per actor × decision rule: evaluation metrics + weighted normalized sum |
| `cv_ranked_decision_metrics.csv` | Same, aggregated across CV folds |
| `run_summary.json` | Seed, sample size, CV splits, reward variant, outcome classifier, selected hyperparameters, model scores |
| `suggested_params_and_scores.txt` | Text log |

### Override cheatsheet

| Override | Effect |
|---|---|
| `use_case=lending` / `lending_knn` / `health` | Use case (and outcome model class) |
| `sample_size=10000` | Training sample size |
| `cv_splits=5` | K-fold splits for hyperparameter search |
| `seed=42` | Random seed |
| `use_case.reward_calculator.reward_variant=strictest` | Reward parameterization (`base` / `mild` / `strictest`) |
| `ranking_weights.Accuracy=0.6 ranking_weights.Demographic_Parity=0.4` | Per-metric ranking weights |

### Headline lending results (RF, base reward, n=10000 × 3 seeds)

```bash
for seed in 42 111 1111; do
    python main.py use_case=lending sample_size=10000 cv_splits=5 seed=$seed
done
```

Group the runs into a bucket folder so the discovery scripts can find them:

```bash
mkdir -p results/lending/rf_base_10000
mv results/lending/run_*Acc_0.4_Fair_0.2 results/lending/rf_base_10000/
```

The bucket name is a human convention — discovery walks `results/{use_case}/`
recursively.

### Sensitivity sweeps

```bash
# Reward-variant axis (RF, n=10000)
for variant in base mild strictest; do
    for seed in 42 111 1111; do
        python main.py use_case=lending sample_size=10000 cv_splits=5 \
            use_case.reward_calculator.reward_variant=$variant seed=$seed
    done
done

# Model-architecture axis (base reward, n=10000)
for uc in lending lending_knn; do
    for seed in 42 111 1111; do
        python main.py use_case=$uc sample_size=10000 cv_splits=5 seed=$seed
    done
done
```

On Windows, `run_sensitivity.ps1` runs these sweeps and `run_all.ps1` runs the full
recipe (training + all figures).

### Health scenario

The health (IHDP) use case is fully supported and documented, though it is not shown
in the dashboard:

```bash
for seed in 42 111 1111; do
    python main.py use_case=health cv_splits=3 seed=$seed
done
```

---

## Figures

### Weight ablations

`ablate_weights.py` sweeps ranking weights over the fixed per-actor metric tensor — a
pure post-hoc computation that retrains nothing. It writes one long-format CSV
(`config × actor`).

```bash
# Pairwise (e.g. Accuracy vs Demographic Parity)
python ablate_weights.py --use-case lending --sweep pairwise \
    --metric-a Accuracy --metric-b Demographic_Parity --n-steps 21 \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/rf_base_10000_acc_vs_dp.csv

# Ternary (3-metric simplex)
python ablate_weights.py --use-case lending --sweep ternary \
    --metric-a Accuracy --metric-b Demographic_Parity --metric-c Total_Profit --n-grid 10 \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/rf_base_10000_ternary.csv

# Dirichlet (robustness over the full simplex)
python ablate_weights.py --use-case lending --sweep dirichlet --n-samples 500 \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/rf_base_10000_dirichlet.csv

# Render a single bucket (sweep type auto-detected)
python plot_ablation.py ablations/rf_base_10000_dirichlet.csv \
    --output figs/ablation_rf_base_10000_dirichlet.png
```

Reference actors (`Oracle`, `Random`, `Outcome_Maxim`, `Nash Social Welfare`) are
excluded from the winner search by default; `Outcome_Pred_Model` is kept as a
prediction baseline. Override with `--include-all` or a custom `--exclude` list.

### Pareto frontier

```bash
python plot_pareto.py --use-case lending \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --x Accuracy --y Demographic_Parity \
    --output figs/pareto_rf_base_10000_acc_vs_dp.png
```

### Combined and stacked paper figures

```bash
# Pareto (left) + ternary (right), one shared legend
python plot_combined_pareto_ternary.py --use-case lending \
    --pareto-x Accuracy --pareto-y Demographic_Parity \
    --pareto-globs \
        "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
        "results/lending/knn_base_10000/run_*/final_ranked_decision_metrics.csv" \
        "results/lending/rf_stricter_10000/run_*/final_ranked_decision_metrics.csv" \
    --ternary-inputs \
        ablations/rf_base_10000_ternary.csv \
        ablations/knn_base_10000_ternary.csv \
        ablations/rf_stricter_10000_ternary.csv \
    --labels "RF / base" "KNN / base" "RF / strictest" \
    --output figs/combined_pareto_ternary.png

# Stacked per-bucket Dirichlet panels
python plot_stacked_dirichlet.py \
    --inputs ablations/rf_base_10000_dirichlet.csv \
             ablations/rf_stricter_10000_dirichlet.csv \
             ablations/knn_base_10000_dirichlet.csv \
    --labels "RF / base" "RF / strictest" "KNN / base" \
    --output figs/stacked_dirichlet.png
```

All figures share one canonical actor → colour map, so a decision function reads as the
same hue across every figure.

---

## Running the dashboard locally

```bash
streamlit run dashboard.py
```

The dashboard discovers every `run_summary.json` under `results/`, groups runs by
`(reward_variant, predictive_model)`, shows mean ± SEM across matching seeds, and never
retrains or modifies anything. When a selected configuration has no matching runs, it
prints the exact `python main.py …` command to fill the gap.

---

## Methodological notes

1. **Normalization equivalence.** `utils/ranking/ranker.py` implements
   `max` → $(x-\min)/(\max-\min)$, `min` → $(\max-x)/(\max-\min)$,
   `zero` → $1 - |x|/\max|x|$, with `Accuracy` passed through. The dashboard, the plot
   scripts, and the training pipeline all call the same `rerank`, so any ranking in the
   dashboard is reproducible by rerunning `main.py` with the equivalent
   `ranking_weights.*` overrides.

2. **Fairness metric definitions.** `Demographic_Parity`, `Equal_Opportunity`, and
   `Conditional_Outcome_Parity` follow Mehrabi et al. (2021), 
   including the convention that lower $|\cdot|$ is fairer. See
   `utils/metrics/fairness_metrics.py`.

3. **Hyperparameter selection is weight-independent.** CV picks hyperparameters by the
   outcome model's accuracy / MAE, not the weighted-ranking score, so the weight choice
   only affects post-hoc ranking — the precondition for the accountability claim.

4. **Use-case-aware fallbacks.** Nash Bargaining's no-agreement default action is the
   first non-positive action of the active use case (`Not_Grant` for lending, `C` for
   health), from `cfg.actions_outcomes.positive_actions_set`.

5. **Compromise Programming weights.** The framework permits non-uniform actor weights
   $w_i$; the experiments use $w_i = 1$, matching Table 1 of the paper.

---

## Citation

If you use this framework, please cite the paper:

> *(citation block — to be filled in upon acceptance)*
