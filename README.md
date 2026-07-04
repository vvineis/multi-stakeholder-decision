# Participatory Training — A Multi-Stakeholder Decision-Support Framework

> Reframes automated decision-making as a **collective-choice problem**: stakeholder preferences become endogenous reward signals that drive both the per-row action and the cross-actor aggregation, so the normative trade-offs that conventional pipelines hide become inspectable.

This repository accompanies the JAIR submission *"A Multi-Stakeholder Framework for Automated Decision-Support Systems"*.

---

## 🌐 Try the interactive dashboard (no install required)

A read-only Streamlit instance reproducing every ranking, radar, and trade-off table from the paper is publicly hosted at:

### **[https://accountability-dashboard.streamlit.app/](https://accountability-dashboard.streamlit.app/)**

In ~30 seconds you can:

- pick any **use case × reward variant × predictive model** combination we trained,
- adjust per-metric **weights** with sliders and watch the live ranking update,
- explore the **two-metric trade-off table** to see which compromise rule wins at each weight pair,
- inspect the **per-stakeholder reward formulas** (with color-coded base-reward lookup tables for lending) and the **compromise-rule definitions** as LaTeX,
- read the **aggregated mean ± SEM** across multiple seeds for the underlying predictive and reward models.

Cold start takes ~30 seconds; once warm, every interaction is bit-for-bit reproducible because the dashboard uses the same `Ranker.rerank` function the training pipeline does.


---

## What problem does the framework solve?

Conventional ADM pipelines collapse heterogeneous stakeholder objectives into a single optimization target, hiding the trade-offs and decision recommendations hard to audit. We treat the same decision space as a multi-actor collective-choice problem:

1. **Per-stakeholder reward elicitation** — each actor $i$ has its own preference vector $r_i$ (no real world elicitaion in this simulation).
2. **Per-stakeholder regression** — supervised learning of $\hat E_{i,a}$, the expected reward for each context-action pair, *per actor*.
3. **Compromise aggregation** — a score map $\Phi_j$ aggregates the expected-reward tensor across actors (e.g. Maximin, Nash Bargaining, Kalai-Smorodinsky, Nash Social Welfare, Compromise Programming, Proportional Fairness).
4. **Post-hoc weighted normalization** — a transparent weighted sum over evaluation metrics selects the "best" compromise rule under user-chosen normative priorities.

The reward, decision, and aggregation layers are **decoupled by design** and the dashboard makes that decoupling explicit and inspectable.

---

## Repository contents

```
.
├── dashboard.py                 # Streamlit accountability dashboard
├── plots_paper.ipynb            # Notebook producing Fig. 3 directly from results/
├── main.py                      # Hydra entry-point — trains models and saves run artifacts
├── ablate_weights.py            # Weight ablations: pairwise trade-off and Dirichlet robustness sweep
├── plot_ablation.py             # Paper-quality plots from ablate_weights output (winner-region strip / violin)
├── plot_pareto.py               # Classical multi-objective Pareto frontier, with optional ablation overlay
├── plot_with_baselines.py       # Static radar + ranking bar for any decision-metrics CSV
├── run_all.ps1                  # One-shot reproducibility script (Windows)
├── run_sensitivity.ps1          # Fig. 3 sensitivity sweeps -- training only (Windows)
├── conf/                        # Hydra configs
│   ├── config.yaml
│   └── use_case/
│       ├── lending.yaml         # Random Forest outcome model (paper default)
│       ├── lending_knn.yaml     # k-NN variant for model-architecture sensitivity
│       └── health.yaml          # IHDP causal-inference scenario
├── src/                         # Pipeline + preprocessing
├── utils/                       # Models, decisions, metrics, rewards, ranking
├── results/                     # Per-run artifacts (created by main.py)
│   ├── lending/run_<…>/
│   └── health/run_<…>/
├── data/                        # Datasets (see "Data" section)
├── requirements.txt             # Full training-pipeline deps
├── requirements-dashboard.txt   # Lightweight subset for the Streamlit Cloud deploy
├── DEPLOY.md                    # Guide to redeploying the dashboard
└── .streamlit/config.toml       # Dashboard theme + server config
```

---

## Setup

```bash
# Python 3.11 recommended
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

This installs everything needed to train the models and produce every figure in the paper. The lightweight `requirements-dashboard.txt` is for the cloud deployment of the dashboard only.

### Data

Two public datasets are used.

- **Lending Club** (`data/lending_club_data.csv`) — credit-decision CSV with `Action`, `Outcome`, `Applicant_Type`, `Loan_Amount`, `Interest_Rate`, and demographic / financial features. Public on the original Lending Club site; a pre-processed copy is included here for reproducibility.
- **IHDP** (`data/health/ihdp_npci_1.csv`) — Infant Health and Development Program causal-inference benchmark (Hill, 2011). The first realization, with 25 features, treatment column `Action ∈ {A, C}`, and outcome.

Both load straight from the `data_path` field in the corresponding YAML; if you move the files, update the YAML accordingly.

---

## Reproducing the experiments

The framework uses [Hydra](https://hydra.cc/) for configuration; every parameter can be overridden on the command line.

### One headline run

```bash
python main.py use_case=lending sample_size=10000 cv_splits=5 seed=111
```

Produces a `results/lending/run_<timestamp>_seed111_Acc_*_Fair_*/` folder containing:

| File | Contents |
|---|---|
| `final_ranked_decision_metrics.csv` | One row per actor × decision rule, columns are evaluation metrics + the weighted normalized sum |
| `cv_ranked_decision_metrics.csv` | Same, aggregated across CV folds |
| `run_summary.json` | Structured metadata: seed, sample size, CV splits, reward variant, outcome classifier, CV-selected hyperparameters, outcome model test score, per-actor reward model test MSE |
| `suggested_params_and_scores.txt` | Legacy text log |

### Hydra override cheatsheet

| Override | Effect |
|---|---|
| `use_case=lending` / `lending_knn` / `health` | Pick the use case (and outcome model class) |
| `sample_size=10000` | Training sample size |
| `cv_splits=5` | K-fold splits for hyperparameter search |
| `seed=42` | Random seed (affects data split, KFold, classifier, SMOTE, numpy/random globals) |
| `use_case.reward_calculator.reward_variant=strictest` | Reward parameterization (`base` / `mild` / `strictest`) |
| `ranking_weights.Accuracy=0.6 ranking_weights.Demographic_Parity=0.4` | Per-metric weights for the final ranking |

### Reproducing the headline lending results (RF, base reward, n=10000 × 3 seeds)

```bash
for seed in 42 111 1111; do
    python main.py use_case=lending sample_size=10000 cv_splits=5 seed=$seed
done
```

After the runs finish, move them into the `rf_base_10000` bucket so the discovery scripts can group them:

```bash
mkdir -p results/lending/rf_base_10000
mv results/lending/run_*Acc_0.4_Fair_0.2 results/lending/rf_base_10000/
```

(or organise into any subfolder layout you prefer — `discover_runs` walks `results/{use_case}/` recursively, so the bucket name is purely a human convention).

(Windows PowerShell users can replace `for size in 1000 ...` with `foreach ($size in 1000, 3000, 5000, 10000)` and use backtick line continuations.)

### Reproducing the sensitivity sweeps (Fig. 3)

```bash
# Reward-variant axis (RF, n = 10000)
for variant in base mild strictest; do
    for seed in 42 111 1111; do
        python main.py use_case=lending sample_size=10000 cv_splits=5 \
            use_case.reward_calculator.reward_variant=$variant seed=$seed
    done
done

# Model-architecture axis (base reward, n = 10000)
for uc in lending lending_knn; do
    for seed in 42 111 1111; do
        python main.py use_case=$uc sample_size=10000 cv_splits=5 seed=$seed
    done
done
```

Then open the notebook to produce the four Fig. 3 panels — it discovers the runs you just created via `run_summary.json` (no intermediate aggregation CSV needed):

```bash
jupyter notebook plots_paper.ipynb
```

#### Or use the orchestration script (Windows)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_sensitivity.ps1     # trains all three sweeps; then open the notebook
```

### Healthcare

```bash
for seed in 42 111 1111; do
    python main.py use_case=health sample_size=787 cv_splits=3 seed=$seed
done
```

---

## Paper figures

### Fig. 2 and Fig. 3 — paper figures via the notebook

Both the dashboard-style three-panel radar (Fig. 2) and the four sensitivity panels (Fig. 3a/b/c/d) are produced by `plots_paper.ipynb`. Open the notebook and *Run All* — it walks every `run_summary.json` under `results/`, groups runs by the relevant metadata field, aggregates across seeds, and saves PNGs into `figs/`.

```bash
jupyter notebook plots_paper.ipynb
```

The notebook produces, in order:
- `figs/fig2_lending.png`, `figs/fig2_health.png` — three-panel radar per use case.
- `figs/fig3a_lending_reward_variant.png` — reward-structure sensitivity (RF, `n=10000`).
- `figs/fig3b_lending_model.png` — RF vs KNN (base reward, `n=10000`).
- `figs/fig3c_health.png` — healthcare across seeds.

If a panel reports *"no data for ..."* you're missing a sweep axis — the cell printouts above each plot tell you which sample sizes / reward variants / models were actually found in your `results/`.

### Pareto frontier (classical multi-objective view)

```bash
python plot_pareto.py \
    --csv results/lending/run_xxx/final_ranked_decision_metrics.csv \
    --use-case lending --x Accuracy --y Demographic_Parity \
    --output figs/pareto_acc_vs_dp.png
```

Each actor is a point in 2-D objective space; the non-dominated set is highlighted with a red dashed frontier. Optionally `--overlay-ablation ablations/lending_dirichlet.csv` consumes a long-format ablation CSV and **scales each actor's marker by how often it wins** across the sweep — so the Pareto-frontier rules that dominate the (re-weighting) reach of the framework get visually emphasized.

Reference baselines (`Oracle`, `Random`, `Outcome_Pred_Model`, `Outcome_Maxim`) are excluded by default; override with `--include-all` or pass an explicit `--actors` list.

---

## Weight ablations

Two sweep types are supported, each with a distinct paper purpose. Both write a single **long-format CSV** (one row per `config × actor`) — no separate `_summary.csv`.

```bash
# (1) Pairwise trade-off (canonical fair-ML story, e.g. Accuracy vs Demographic Parity)
python ablate_weights.py --use-case lending --sweep pairwise \
    --metric-a Accuracy --metric-b Demographic_Parity --n-steps 21 \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/lending_acc_vs_dp.csv

# (2) Dirichlet over the simplex (robustness of compromise rules)
python ablate_weights.py --use-case lending --sweep dirichlet --n-samples 500 \
    --metrics-glob "results/lending/rf_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/lending_dirichlet.csv

# (3) Disaggregated comparison: same ablation per bucket, then stack winner strips
python ablate_weights.py --use-case lending --sweep pairwise \
    --metric-a Accuracy --metric-b Demographic_Parity --n-steps 21 \
    --metrics-glob "results/lending/rf_stricter_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/rf_strictest_acc_vs_dp.csv

python ablate_weights.py --use-case lending --sweep pairwise \
    --metric-a Accuracy --metric-b Demographic_Parity --n-steps 21 \
    --metrics-glob "results/lending/knn_base_10000/run_*/final_ranked_decision_metrics.csv" \
    --output ablations/knn_base_acc_vs_dp.csv

python plot_compare_ablations.py \
    --inputs ablations/lending_acc_vs_dp.csv \
             ablations/rf_strictest_acc_vs_dp.csv \
             ablations/knn_base_acc_vs_dp.csv \
    --labels "RF / base" "RF / strictest" "KNN / base" \
    --output figs/compare_acc_vs_dp.png

# Render (auto-detects the sweep type)
python plot_ablation.py ablations/lending_acc_vs_dp.csv  --output figs/ablation_acc_vs_dp.png
python plot_ablation.py ablations/lending_dirichlet.csv  --output figs/ablation_dirichlet.png
```

`plot_ablation.py` produces:

- **Pairwise sweep** → two-panel plot: a line plot of `Weighted_Normalized_Sum` vs. the varying weight (one line per compromise rule) and a **winner-region strip** beneath it that color-codes the winning rule along the weight axis.
- **Dirichlet sweep** → two-panel plot: a **winner-share bar chart** (how often each rule is rank 1) and a **violin plot** of the score distributions, ordered by median so the most-robust rule sits at the top.

Reference actors (`Oracle`, `Random`, `Outcome_Pred_Model` for lending; `Outcome_Maxim` for health) are **excluded** from the winner search by default — they're upper-bound or stochastic baselines that say nothing about the choice of compromise rule. Override with `--include-all`.

This works because the per-actor metric tensor in the CSV is *independent of the weights* — only the final aggregation step depends on them. So sweeping weights is a pure post-hoc computation that does not retrain anything.

---

## Running the dashboard locally

The hosted instance ([above](#-try-the-interactive-dashboard-no-install-required)) is sufficient- To run a local copy against your own training artifacts:

```bash
streamlit run dashboard.py
```

The dashboard:

- discovers every `run_summary.json` under `results/`,
- groups them by `(reward_variant, outcome_classifier)` (sample size fixed at 10000),
- displays mean ± SEM across all matching seeds,
- never retrains or modifies anything — it is strictly read-only.

When you select a configuration that has no matching runs, the dashboard prints the exact `python main.py …` command you would need to execute to fill the gap.

See `DEPLOY.md` for instructions on redeploying the public version (Streamlit Community Cloud, free for public apps).

---

## Methodological notes

The framework is built on five design commitments worth stating explicitly:

1. **Normalization equivalence.** `utils/ranking/ranker.py` implements `max` → $(x-\min)/(\max-\min)$, `min` → $(\max-x)/(\max-\min)$, `zero` → $1 - |x|/\max|x|$, with `Accuracy` passed through raw. The dashboard, the CLI plot scripts, and the training pipeline all call the same `Ranker.rerank` function, so any ranking shown in the dashboard is bit-for-bit reproducible by rerunning `main.py` with the equivalent `ranking_weights.*` overrides.

2. **Fairness metric definitions.** `Demographic_Parity`, `Equal_Opportunity`, and `Conditional_Outcome_Parity` follow Mehrabi et al. (2021) / Chouldechova (2017) verbatim, including the choice of `positive_attribute_for_fairness` and the convention that lower $|\cdot|$ is fairer. They live in `utils/metrics/fairness_metrics.py`.

3. **Hyperparameter selection is weight-independent.** The CV step picks hyperparameters by the *outcome model's accuracy / MAE* alone, not by the weighted-ranking score. The weight choice therefore only affects post-hoc ranking, never which model gets trained — the precondition for the accountability claim.

4. **Use-case-aware fallbacks.** Nash Bargaining's no-agreement default action is the first *non-positive* action of the active use case (`Not_Grant` for lending, `C` for health), derived from `cfg.actions_outcomes.positive_actions_set` rather than hard-coded.

5. **Compromise Programming weights.** The framework permits non-uniform actor weights $w_i$, but our experiments use $w_i = 1$ (uniform), matching the score map shown in Table 1 of the paper.

---

## Citation

If you use this framework, please cite the JAIR paper:

> *(citation block — to be filled in upon acceptance)*

The dashboard URL above can be appended to the paper's reproducibility section.

---
