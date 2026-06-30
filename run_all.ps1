# =====================================================================
# Participatory-Training framework - full reproducibility recipe
#   * 3 seeds x 2 use cases
#   * the weight ablations the paper relies on
#   * every plot (Fig. 2 radar, Fig. 3 line plots, Pareto, ablations)
#
# Usage (from C:\Users\Vittoria\Desktop\part-train\new-code):
#   conda activate PartTrainEnv
#   .\run_all.ps1
#
# Expected wall time on a single laptop core: ~30-60 minutes for lending,
# a few minutes for health.
# =====================================================================

# Move to the script's directory regardless of where it was invoked from
Set-Location -LiteralPath $PSScriptRoot

$ErrorActionPreference = "Stop"
$SEEDS = @(42, 111, 1111)

# --- sanity check: PartTrainEnv must be active ---------------------------
$null = python -c "import hydra, sklearn, pandas, numpy, yaml, matplotlib" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "Required Python packages are missing from this shell's interpreter." -ForegroundColor Red
    Write-Host "Most likely cause: PartTrainEnv is not activated." -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "Fix:  open a terminal that shows '(PartTrainEnv) PS> ...', then:" -ForegroundColor Red
    Write-Host "        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor Yellow
    Write-Host "        .\run_all.ps1" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Red
    Write-Host "Do NOT launch via 'powershell -ExecutionPolicy Bypass -File run_all.ps1'" -ForegroundColor Red
    Write-Host "-- that starts a new shell without the conda env." -ForegroundColor Red
    exit 1
}
Write-Host "Python env looks healthy." -ForegroundColor Green

# ----- "good performance" defaults --------------------------------------
# lending : RandomForest classifier, full sample, 5 CV folds
# health  : XGBoost X-learner, full IHDP-1 sample, 3 CV folds
$LENDING_OVERRIDES = @(
    "use_case=lending",
    "sample_size=10000",
    "cv_splits=5"
)
$HEALTH_OVERRIDES = @(
    "use_case=health",
    "sample_size=787",
    "cv_splits=3"
)

# =====================================================================
# 1. Train every seed for both use cases
# =====================================================================
Write-Host "=== STAGE 1: training (3 seeds x 2 use cases) ===" -ForegroundColor Cyan

foreach ($seed in $SEEDS) {
    Write-Host "lending seed=$seed" -ForegroundColor Yellow
    python main.py @LENDING_OVERRIDES "seed=$seed"
}

foreach ($seed in $SEEDS) {
    Write-Host "health seed=$seed" -ForegroundColor Yellow
    python main.py @HEALTH_OVERRIDES "seed=$seed"
}

# =====================================================================
# 2. Pick representative runs for downstream plots
# =====================================================================
function Latest-Run([string]$useCase, [int]$seed, [string]$suffixGlob = "*") {
    $globPattern = "results/$useCase/run_*seed$seed*$suffixGlob/final_ranked_decision_metrics.csv"
    $hit = Get-ChildItem -Path $globPattern -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $hit) { throw "No run found for $useCase seed=$seed" }
    return $hit
}

$LEND_CSV    = (Latest-Run "lending" 111).FullName
$LEND_DIR    = (Get-Item $LEND_CSV).Directory.FullName

$HEALTH_CSV  = (Latest-Run "health" 111).FullName
$HEALTH_DIR  = (Get-Item $HEALTH_CSV).Directory.FullName

Write-Host "Representative lending run: $LEND_DIR"
Write-Host "Representative health  run: $HEALTH_DIR"

# =====================================================================
# 3. Weight ablations (pairwise, single-metric, Dirichlet)
# =====================================================================
Write-Host "=== STAGE 3: weight ablations ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path ablations | Out-Null

# Lending -- pairwise Accuracy vs Demographic_Parity (the canonical fair-ML trade-off)
python ablate_weights.py --use-case lending --sweep pairwise `
    --metric-a Accuracy --metric-b Demographic_Parity --n-steps 21 `
    --metrics-csv $LEND_CSV `
    --output ablations/lending_acc_vs_dp.csv

# Lending -- Dirichlet over the simplex of all evaluation metrics
python ablate_weights.py --use-case lending --sweep dirichlet --n-samples 500 `
    --metrics-csv $LEND_CSV `
    --output ablations/lending_dirichlet.csv

# Health -- Dirichlet
python ablate_weights.py --use-case health --sweep dirichlet --n-samples 500 `
    --metrics-csv $HEALTH_CSV `
    --output ablations/health_dirichlet.csv

# =====================================================================
# 4. CLI plots that DON'T require the notebook: weight-ablation plots
#    and Pareto frontiers. The Fig. 2 (three-panel radar) and Fig. 3
#    (sensitivity panels) figures are produced by the Jupyter notebook
#    `plots_paper.ipynb`, which reads the run folders directly.
# =====================================================================
Write-Host "=== STAGE 4: CLI plots ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path figs | Out-Null

# 4a. Weight-ablation plots (auto-detect sweep type)
python plot_ablation.py ablations/lending_acc_vs_dp.csv  --output figs/ablation_lending_acc_vs_dp.png
python plot_ablation.py ablations/lending_dirichlet.csv  --output figs/ablation_lending_dirichlet.png
python plot_ablation.py ablations/health_dirichlet.csv   --output figs/ablation_health_dirichlet.png

# 4b. Pareto frontier (Accuracy vs Demographic_Parity) -- with Dirichlet wins overlaid
python plot_pareto.py --csv $LEND_CSV --use-case lending `
    --x Accuracy --y Demographic_Parity `
    --overlay-ablation ablations/lending_dirichlet.csv `
    --output figs/pareto_lending_acc_vs_dp.png

python plot_pareto.py --csv $HEALTH_CSV --use-case health `
    --x Accuracy --y Demographic_Parity `
    --overlay-ablation ablations/health_dirichlet.csv `
    --output figs/pareto_health_acc_vs_dp.png

Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Ablation / Pareto plots in : figs/"
Write-Host "Weight ablations in        : ablations/"
Write-Host ""
Write-Host "For Fig. 2 (three-panel radar) and Fig. 3 (sensitivity panels):"
Write-Host "  jupyter notebook plots_paper.ipynb"
Write-Host "Launch the dashboard with:  streamlit run dashboard.py"
