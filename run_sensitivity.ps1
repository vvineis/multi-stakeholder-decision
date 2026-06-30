# =====================================================================
# Sensitivity analysis -- training only.
#
# Three families of variation, each x 3 seeds:
#   (a) reward structure  : base / mild / strictest   (lending, RF, n=10000)
#   (b) predictive model  : RF / KNN                  (lending, base reward, n=10000)
#   (c) training sample   : 1000 / 3000 / 5000 / 10000 (lending, RF, base)
#
# Each run drops its `final_ranked_decision_metrics.csv` and `run_summary.json`
# under `results/lending/run_<...>/`. To produce the figures from those runs,
# open `plots_paper.ipynb` and run all cells -- the notebook discovers the
# matching runs directly from disk via `run_summary.json`, no intermediate
# aggregation step is needed.
#
# Wall-clock budget on a single laptop CPU:
#   ~3-6 h for (a) and (b); ~6-10 h if you include (c) at the larger sizes.
#   For a quick smoke test, override $SEEDS to @(111) at the top.
# =====================================================================
Set-Location -LiteralPath $PSScriptRoot
$ErrorActionPreference = "Stop"

# --- sanity check: PartTrainEnv must be active ---------------------------
$null = python -c "import hydra, sklearn, pandas, numpy, yaml, matplotlib" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "PartTrainEnv not active in this shell. See run_all.ps1 for the fix." -ForegroundColor Red
    exit 1
}

$SEEDS = @(42, 111, 1111)

# =====================================================================
# (a) Reward structure: base / strictest, RF, n=10000
# =====================================================================
Write-Host "=== (a) reward structure sweep ===" -ForegroundColor Cyan
foreach ($variant in "base", "strictest") {
    foreach ($seed in $SEEDS) {
        Write-Host "    reward=$variant seed=$seed" -ForegroundColor Yellow
        python main.py use_case=lending sample_size=10000 cv_splits=5 `
            "use_case.reward_calculator.reward_variant=$variant" "seed=$seed"
    }
}

# =====================================================================
# (b) Predictive model: RF (use_case=lending) vs KNN (use_case=lending_knn)
# =====================================================================
Write-Host "=== (b) model sweep ===" -ForegroundColor Cyan
foreach ($uc in "lending", "lending_knn") {
    foreach ($seed in $SEEDS) {
        Write-Host "    use_case=$uc seed=$seed" -ForegroundColor Yellow
        python main.py "use_case=$uc" sample_size=10000 cv_splits=5 "seed=$seed"
    }
}

# =====================================================================
# (c) Sample size: 1000 / 3000 / 5000 / 10000, RF, base
# =====================================================================
Write-Host "=== (c) sample-size sweep ===" -ForegroundColor Cyan
foreach ($sz in 1000, 3000, 5000, 10000) {
    foreach ($seed in $SEEDS) {
        Write-Host "    sample_size=$sz seed=$seed" -ForegroundColor Yellow
        python main.py use_case=lending sample_size=$sz cv_splits=5 "seed=$seed"
    }
}

Write-Host "=== TRAINING DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: open plots_paper.ipynb in Jupyter and run all cells."
Write-Host "The notebook discovers every run under results/lending/ via run_summary.json,"
Write-Host "groups them by sensitivity axis, and renders the four Fig. 3 panels into figs/."
