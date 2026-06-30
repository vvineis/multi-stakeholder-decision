"""
Accountability dashboard -- aggregated edition.

What this version adds over the per-run one:

1.  Runs are now grouped by (sample_size, reward_variant, outcome_classifier),
    and the dashboard reports **mean +/- SEM** across all the matching seeds.
    The radar, the ranking table and the model-performance numbers all use
    the aggregated values.

2.  An expandable **"Compromise functions -- definitions"** section that prints
    each rule's score map Phi as LaTeX, matched to Table 1 of the paper.

3.  A **trade-off table**: pick two metrics, pick a weight grid (e.g. 0, 0.25,
    0.5, 0.75, 1), and the dashboard reports the best compromise function for
    each weight combination. Optionally restrict the candidate set to "compromise
    rules only" (excluding Oracle / single-stakeholder baselines).

4.  When no runs match the chosen (sample_size, reward_variant, model) triple,
    the dashboard prints the exact `python main.py ...` command that would
    produce those runs -- so the user can copy-paste it instead of clicking
    around.

Run from `new-code/`:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.ranking.ranker import rerank
from utils.rewards.get_rewards import RewardCalculator
from ablate_weights import (
    DEFAULT_EXCLUDE as ABLATE_EXCLUDE,
    ablate_to_long,
    sweep_dirichlet,
)

import matplotlib.pyplot as _plt  # for the Dirichlet comparison panel
from matplotlib.figure import Figure


# ======================================================================
# Constants
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT_CANDIDATES = [
    PROJECT_ROOT / "results",
    PROJECT_ROOT.parent / "participatory_training-main" / "results",
    PROJECT_ROOT.parent / "results",
]
_EPS = 1e-9

# Score maps Phi for each compromise rule (paper Table 1). LaTeX strings.
COMPROMISE_FORMULAS = {
    "Maximin": (
        r"\Phi_{\mathrm{Maximin}}(\mathbf{E}_{:,a}) = \min_{i\in\mathcal{I}}\, E_{i,a}",
        "Safeguards the most disadvantaged actor by maximizing the minimum "
        "expected reward across the population.",
    ),
    "Nash Bargaining": (
        r"\Phi_{\mathrm{NB}}(\mathbf{E}_{:,a}) = "
        r"\prod_{i\in\mathcal{I}} \max(0,\, E_{i,a} - d_i)",
        "Maximizes expected reward gains above each actor's disagreement "
        "payoff $d_i$, balancing fairness and efficiency.",
    ),
    "Kalai-Smorodinsky": (
        r"\Phi_{\mathrm{KS}}(\mathbf{E}_{:,a}) = "
        r"\min_{i\in\mathcal{I}}\, \frac{E_{i,a} - d_i}{E^{*}_i - d_i}",
        "Maximizes proportional gains toward each actor's ideal reward "
        "$E^{*}_i$ relative to their disagreement payoff $d_i$.",
    ),
    "Nash Social Welfare": (
        r"\Phi_{\mathrm{NSW}}(\mathbf{E}_{:,a}) = "
        r"\prod_{i\in\mathcal{I}} \max(\epsilon,\, E_{i,a})",
        "Promotes balanced improvements in collective well-being "
        "(geometric-mean welfare).",
    ),
    "Compromise Programming": (
        r"\Phi_{\mathrm{CP}}(\mathbf{E}_{:,a}) = "
        r"-\sqrt{\sum_{i\in\mathcal{I}} w_i\,(E^{*}_i - E_{i,a})^2}",
        "Minimizes the (weighted) Euclidean distance from each actor's "
        "achieved reward to their ideal $E^{*}_i$.",
    ),
    "Proportional Fairness": (
        r"\Phi_{\mathrm{PF}}(\mathbf{E}_{:,a}) = "
        r"\sum_{i\in\mathcal{I}} \log E_{i,a}",
        "Log-utility welfare; equivalent to Nash bargaining in log space "
        "with disagreement at zero.",
    ),
}

# Sets we use to decide what counts as a "compromise rule" for the trade-off table.
COMPROMISE_RULES = set(COMPROMISE_FORMULAS.keys())


TRADEOFF_EXCLUDE = {"Oracle", "Random", "Outcome_Pred_Model"}


# ----------------------------------------------------------------------
# Per-stakeholder reward functions r_i (paper Section 4.1)
# ----------------------------------------------------------------------
# Each reward combines a base lookup table indexed by (action, outcome,
# applicant_type) with context-dependent multiplicative adjustments capturing
# the loan amount L and interest rate R, plus i.i.d. uniform noise clipped
# to [0, 1]. The adjustment factors:
#   l(L) = clip(L / 10000, 0.5, 1.5)        (loan factor)
#   p(R) = clip(R, 0.05, 0.25)              (rate factor)
#   xi ~ Uniform(-0.05, 0.05)               (noise)
LENDING_REWARD_FORMULAS = {
    "Bank": (
        r"r_{\mathrm{Bank}}(a, o, T, L, R) = "
        r"\mathrm{clip}\!\left("
        r"R^{\mathrm{Bank}}_{\mathrm{base}}(a, o, T) \cdot \ell(L) \cdot (1 + \rho(R)) + \xi,\; 0, 1"
        r"\right)",
        "Maximizes interest revenue; rewards full repayment, penalizes default. "
        "Scales positively with both loan amount and interest rate.",
    ),
    "Applicant": (
        r"r_{\mathrm{App}}(a, o, T, L, R) = "
        r"\mathrm{clip}\!\left("
        r"R^{\mathrm{App}}_{\mathrm{base}}(a, o, T) \cdot (2 - \rho(R)) \cdot (1 - \ell(L)) + \xi,\; 0, 1"
        r"\right)",
        "Values loan access proportionally to financial need; penalizes "
        "high interest rates and very large loan amounts (more debt risk).",
    ),
    "Regulatory": (
        r"r_{\mathrm{Reg}}(a, o, T, L, R) = "
        r"\mathrm{clip}\!\left("
        r"R^{\mathrm{Reg}}_{\mathrm{base}}(a, o, T) \cdot (1 - \rho(R)) \cdot \ell(L) + \xi,\; 0, 1"
        r"\right)",
        "Balances systemic stability with financial inclusion; rewards "
        "responsible credit access for vulnerable applicants.",
    ),
}

# For health, the formulas are closed-form so the base table is not needed.
HEALTH_REWARD_FORMULAS = {
    "Parent": (
        r"r_{\mathrm{Parent}}(o) = \mathrm{clip}\!\left("
        r"\frac{o - o_{\min}}{o_{\max} - o_{\min}},\; 0, 1\right)",
        "Directly proportional to the normalized cognitive score; parents "
        "value any improvement in their child's outcome.",
    ),
    "Healthcare_Provider": (
        r"r_{\mathrm{HCP}}(o, a) = \mathrm{clip}\!\left("
        r"\alpha \cdot \frac{\max(0,\, o - o_{\min,a})}{o_{\max,a} - o_{\min,a}} + "
        r"(1 - \alpha) \cdot \left(1 - \frac{c_a}{\max_{a'} c_{a'}}\right) + \xi,\; 0, 1\right)",
        r"Convex combination ($\alpha = 0.8$) of normalized outcome improvement "
        r"and treatment-cost efficiency ($c_a$ is the cost of action $a$).",
    ),
    "Policy_Maker": (
        r"r_{\mathrm{PM}}(o, a, x_{23}) = \mathrm{clip}\!\left("
        r"\frac{\max(0,\, o - o_{\min,a})}{o_{\max,a} - o_{\min,a}} \cdot "
        r"\bigl(1 + \beta (x_{23} - 0.5)\bigr) + \xi,\; 0, 1\right)",
        r"Outcome improvement weighted by a demographic fairness term "
        r"($\beta = 0.5$); rewards equity across the protected attribute $x_{23}$.",
    ),
}


def _lending_base_reward_table(structures: dict, stakeholder: str, group_id: int) -> pd.DataFrame:
    """Build a (action x outcome) DataFrame of base rewards for a (group, stakeholder)."""
    raw = structures[group_id][stakeholder]
    actions = ["Grant", "Grant_lower", "Not_Grant"]
    outcomes = ["Fully_Repaid", "Partially_Repaid", "Not_Repaid"]
    df = pd.DataFrame(index=actions, columns=outcomes, dtype=float)
    for (a, o), v in raw.items():
        df.loc[a, o] = float(v)
    df.index.name = "Action"
    return df


# ======================================================================
# Helpers
# ======================================================================
def _short_model(target: str | None) -> str | None:
    if not target:
        return None
    last = target.rsplit(".", 1)[-1].lower()
    if "randomforest" in last: return "rf"
    if "kneighbors" in last:   return "knn"
    if "xgb" in last:          return "xgb"
    if "lightgbm" in last:     return "lgbm"
    return last


def _find_results_root() -> Path | None:
    for p in RESULTS_ROOT_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data
def _load_use_case_yaml(use_case: str) -> dict:
    p = PROJECT_ROOT / "conf" / "use_case" / f"{use_case}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


@st.cache_data
def discover_runs(results_root_str: str, use_case: str, *, sample_size_only: int | None = 10000) -> list[dict]:
    """Walk `results/{use_case}/` recursively and return one dict per run.

    The new repository layout buckets runs by configuration -- e.g.
    `results/lending/rf_base_10000/run_*/`. We use `rglob('run_summary.json')`
    so the discovery is depth-agnostic (works whether runs sit directly in
    `lending/` or under any bucket subfolder).

    By default we filter to runs at `sample_size == sample_size_only` (10000 for
    lending; ignored for health where there's only one size). Pass
    `sample_size_only=None` to keep everything.
    """
    results_root = Path(results_root_str)
    out = []
    for child in ("lending", "health"):
        base = results_root / child
        if not base.exists():
            continue
        # Recursive: matches results/lending/<bucket>/run_*/run_summary.json
        # AND results/lending/run_*/run_summary.json (legacy flat layout).
        for sj in sorted(base.rglob("run_summary.json")):
            run_dir = sj.parent
            csv = run_dir / "final_ranked_decision_metrics.csv"
            if not csv.exists():
                continue
            try:
                summary = json.loads(sj.read_text())
            except json.JSONDecodeError:
                continue
            base_uc = (summary.get("use_case") or "").replace("_knn", "")
            if base_uc != use_case:
                continue
            if (sample_size_only is not None
                    and base_uc == "lending"
                    and summary.get("sample_size") != sample_size_only):
                continue
            out.append({
                "path": str(run_dir),
                "csv": str(csv),
                "seed": summary.get("seed"),
                "sample_size": summary.get("sample_size"),
                "cv_splits": summary.get("cv_splits"),
                "reward_variant": summary.get("reward_variant") or "base",
                "model": _short_model(summary.get("outcome_classifier")) or "unknown",
                "summary": summary,
            })
    return out


@st.cache_data
def load_metric_csvs(csv_paths: tuple[str, ...]) -> list[pd.DataFrame]:
    """Read each CSV and drop the cached ranking columns -- we recompute live."""
    drop_keywords = (" Normalized", " Rank", "Weighted Normalized-Sum")
    out = []
    for p in csv_paths:
        df = pd.read_csv(p)
        cols = [c for c in df.columns if any(k in c for k in drop_keywords)]
        out.append(df.drop(columns=cols, errors="ignore"))
    return out


def aggregate_per_seed(per_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate and compute mean/std/sem per (Actor/Criterion, metric)."""
    combined = pd.concat(per_seed, ignore_index=True, sort=False)
    metric_cols = [
        c for c in combined.columns
        if c != "Actor/Criterion" and pd.api.types.is_numeric_dtype(combined[c])
    ]
    grouped = combined.groupby("Actor/Criterion")[metric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0).add_suffix("_std")
    n = grouped.count().add_suffix("_n")
    sem_arr = std.values / np.sqrt(np.maximum(n.values, 1))
    sem = pd.DataFrame(
        sem_arr, index=std.index,
        columns=[c.replace("_std", "_sem") for c in std.columns],
    )
    out = pd.concat([mean, std, sem], axis=1).reset_index()
    out["n_seeds"] = grouped.size().values
    return out


def aggregated_to_metrics_df(agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Extract a metrics_df-style frame (one row per Actor) using the _mean columns."""
    out = agg[["Actor/Criterion"]].copy()
    for m in metrics:
        col = f"{m}_mean"
        if col in agg.columns:
            out[m] = agg[col]
    return out


def normalize_for_display(df: pd.DataFrame, metrics: list[str], ranking_criteria: dict) -> pd.DataFrame:
    """Same rule the Ranker uses, so the radar agrees with the score."""
    out = df.copy()
    for m in metrics:
        if m not in out.columns:
            continue
        col = out[m]
        direction = ranking_criteria.get(m, "max")
        if m == "Accuracy":
            out[m] = col.copy()
            continue
        if direction == "max":
            lo, hi = col.min(), col.max()
            out[m] = (col - lo) / max(hi - lo, _EPS)
        elif direction == "min":
            lo, hi = col.min(), col.max()
            out[m] = (hi - col) / max(hi - lo, _EPS)
        elif direction == "zero":
            max_abs = col.abs().max()
            out[m] = 1.0 - col.abs() / max(max_abs, _EPS)
    return out


def build_tradeoff_table(
    metrics_df: pd.DataFrame,
    metric_a: str, metric_b: str,
    weight_grid: list[float],
    ranking_criteria: dict,
    actors_filter: set[str] | None = None,
    excluded: set[str] = TRADEOFF_EXCLUDE,
) -> pd.DataFrame:
    """For each (w_a, 1-w_a) in `weight_grid`, find the best actor restricted to `actors_filter`.

    `excluded` (default `TRADEOFF_EXCLUDE`) is always subtracted from the
    candidate set, even when `actors_filter` is None. This is how we keep
    Oracle out of the table regardless of the "Only compromise rules" toggle.
    """
    rows = []
    for w_a in weight_grid:
        w_b = 1.0 - w_a
        weights = {metric_a: float(w_a), metric_b: float(w_b)}
        try:
            ranked, _, _ = rerank(
                metrics_df,
                ranking_criteria=ranking_criteria,
                ranking_weights=weights,
                metrics_for_evaluation=[metric_a, metric_b],
            )
            if actors_filter:
                ranked = ranked[ranked["Actor/Criterion"].isin(actors_filter)]
            if excluded:
                ranked = ranked[~ranked["Actor/Criterion"].isin(excluded)]
            if ranked.empty:
                best, score = "(no matching actor)", np.nan
            else:
                best = ranked.iloc[0]["Actor/Criterion"]
                score = float(ranked.iloc[0]["Weighted Normalized-Sum"])
        except Exception as e:  # pragma: no cover -- defensive
            best, score = f"(error: {e})", np.nan
        rows.append({
            f"w({metric_a})": round(float(w_a), 3),
            f"w({metric_b})": round(float(w_b), 3),
            "Best compromise": best,
            "Weighted Sum": round(score, 4) if pd.notna(score) else None,
        })
    return pd.DataFrame(rows)


def hydra_command_for(missing: dict) -> str:
    """Print the exact `python main.py` line that would produce the missing config."""
    uc = "lending" if missing["model"] in ("rf", "xgb", "lgbm") else missing["model"]
    if missing["model"] == "knn":
        uc = "lending_knn"
    parts = [
        "python main.py",
        f"use_case={uc}",
        f"sample_size={missing['sample_size']}",
        "cv_splits=5",
        f"use_case.reward_calculator.reward_variant={missing['reward_variant']}",
        "seed=<one of 42, 111, 1111>",
    ]
    return " ".join(parts)


# ======================================================================
# Streamlit page layout
# ======================================================================
st.set_page_config(
    page_title="Decision Accountability Dashboard",
    layout="wide",
)
st.title("Decision Accountability Dashboard")
st.caption(
    "Aggregated view across seeds. Pick a configuration in the sidebar; "
    "all matching seed runs are pooled into a single mean +/- SEM view."
)

results_root = _find_results_root()
if results_root is None:
    st.error("Cannot find a `results/` folder. Have you run `main.py` yet?")
    st.stop()


# ---- Sidebar: use case + configuration ------------------------------
with st.sidebar:
    st.header("1. Use case")
    use_case = st.selectbox("Use case", options=("lending", "health"), index=0)
    try:
        use_case_cfg = _load_use_case_yaml(use_case)
    except FileNotFoundError:
        st.error("Cannot find conf/use_case/*.yaml. Are you running from new-code/?")
        st.stop()

    all_runs = discover_runs(str(results_root), use_case)
    if not all_runs:
        st.error(
            f"No runs found under `results/{use_case}/`. "
            f"Run `python main.py use_case={use_case} ...` first."
        )
        st.stop()

    reward_variants = sorted({r["reward_variant"] for r in all_runs})
    models = sorted({r["model"] for r in all_runs})

    st.header("2. Configuration")
    reward_variant = st.selectbox(
        "Reward variant", options=reward_variants,
        index=reward_variants.index("base") if "base" in reward_variants else 0,
    )
    model = st.selectbox(
        "Predictive model", options=models,
        index=models.index("rf") if "rf" in models else 0,
    )
    # Sample size is no longer a user-facing knob -- we always filter to 10000
    # for lending (handled in discover_runs). All matched runs therefore share
    # the same sample size.
    sample_size = next((r["sample_size"] for r in all_runs), None)

    matched = [r for r in all_runs
               if r["reward_variant"] == reward_variant
               and r["model"] == model]

    seeds_present = sorted({r["seed"] for r in matched if r["seed"] is not None})
    if not matched:
        missing = {"sample_size": sample_size or 10000, "reward_variant": reward_variant, "model": model}
        st.error("No runs for this configuration. Run, e.g.:")
        st.code(hydra_command_for(missing), language="bash")
        st.stop()
    elif len(seeds_present) < 2:
        st.warning(
            f"Only 1 seed run for this configuration ({seeds_present}). "
            "Uncertainty bands will be zero. Run more seeds for SEM estimates."
        )
    else:
        st.success(f"{len(matched)} seed runs found: seeds = {seeds_present}")


# ---- Load + aggregate ----------------------------------------------
csv_paths = tuple(r["csv"] for r in matched)
per_seed = load_metric_csvs(csv_paths)
agg = aggregate_per_seed(per_seed)

all_metrics = list(use_case_cfg["criteria"]["metrics_for_evaluation"])
ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])
default_weights = dict(use_case_cfg["criteria"]["ranking_weights"])


# ---- Sidebar: metric picker + weight sliders ------------------------
with st.sidebar:
    st.header("3. Metrics in play")
    st.caption("These metrics participate in BOTH the radar and the live score.")
    selected_metrics = st.multiselect(
        "Metrics",
        options=all_metrics,
        default=[m for m in all_metrics if default_weights.get(m, 0) > 0] or all_metrics,
    )
    if not selected_metrics:
        st.warning("Select at least one metric.")
        st.stop()

    st.header("4. Ranking weights")
    weights: dict[str, float] = {}
    for m in selected_metrics:
        weights[m] = st.slider(
            m, min_value=0.0, max_value=1.0,
            value=float(default_weights.get(m, 0.0)) or 1.0 / len(selected_metrics),
            step=0.05,
        )
    if sum(weights.values()) == 0:
        st.warning("All weights are zero; defaulting to a uniform mixture.")
        weights = {m: 1.0 / len(selected_metrics) for m in selected_metrics}


metrics_df = aggregated_to_metrics_df(agg, all_metrics)
ranked_df, _, best = rerank(
    metrics_df,
    ranking_criteria=ranking_criteria,
    ranking_weights=weights,
    metrics_for_evaluation=selected_metrics,
)


# ======================================================================
# Header banner: what you're looking at
# ======================================================================
banner_cols = st.columns(4)
banner_cols[0].metric("Use case", use_case)
banner_cols[1].metric("Reward variant", reward_variant)
banner_cols[2].metric("Model", model.upper())
banner_cols[3].metric("Seeds aggregated", len(seeds_present))
if sample_size:
    st.caption(f"All runs at sample_size = {sample_size}.")
st.markdown("---")


# ======================================================================
# Compromise function formulas (paper Table 1)
# ======================================================================
with st.expander("Compromise function definitions  (paper Table 1)"):
    st.markdown(
        "Each compromise rule $C_j$ is defined by a score map "
        r"$\Phi_j: [0,1]^{|\mathcal{I}|} \to \mathbb{R}$ that aggregates the "
        "actor-specific expected rewards $E_{i,a}$ for each action $a$."
    )
    for name, (latex, description) in COMPROMISE_FORMULAS.items():
        st.markdown(f"**{name}**")
        st.latex(latex)
        st.caption(description)


with st.expander(f"Stakeholder reward functions  --  {use_case} ({reward_variant} variant)"):
    if use_case == "lending":
        st.markdown(
            r"For each applicant context, the reward of stakeholder $i$ is the "
            r"product of a **base lookup table** $R^{i}_{\mathrm{base}}(a, o, T)$ "
            r"(indexed by action $a$, repayment outcome $o$, applicant type $T$) "
            r"and a context-dependent adjustment in the loan amount $L$ and "
            r"interest rate $R$, plus i.i.d. uniform noise clipped to $[0, 1]$."
        )
        st.markdown("**Adjustment factors**")
        st.latex(
            r"\ell(L) = \mathrm{clip}\!\left(\frac{L}{10{,}000},\, 0.5,\, 1.5\right), \quad "
            r"\rho(R) = \mathrm{clip}\!\left(R,\, 0.05,\, 0.25\right), \quad "
            r"\xi \sim \mathrm{Unif}(-0.05, 0.05)"
        )

        try:
            structures = RewardCalculator.get_structures_for_variant(reward_variant)
        except ValueError:
            structures = None
            st.warning(f"Unknown reward variant '{reward_variant}' -- showing formulas only.")

        for actor in ("Bank", "Applicant", "Regulatory"):
            st.markdown(f"#### {actor}")
            latex, descr = LENDING_REWARD_FORMULAS[actor]
            st.latex(latex)
            st.caption(descr)

            if structures is None:
                continue
            base_cols = st.columns(2)
            for col, (group_id, group_label) in zip(
                base_cols,
                [(0, "Non-vulnerable applicants  (T = 0)"),
                 (1, "Vulnerable applicants  (T = 1)")],
            ):
                with col:
                    st.markdown(f"**Base reward table** -- {group_label}")
                    df = _lending_base_reward_table(structures, actor, group_id)
                    styled = (
                        df.style
                        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                        .format("{:.2f}")
                    )
                    st.dataframe(styled, use_container_width=True)

    else:  # health
        st.markdown(
            "Each actor's reward is a closed-form mapping from the observed "
            r"cognitive outcome $o$, the chosen action $a$ "
            r"($\mathrm{A}$ = treatment, $\mathrm{C}$ = control), and the "
            r"demographic attribute $x_{23}$, plus i.i.d. uniform noise "
            r"clipped to $[0, 1]$. Per-action statistics "
            r"$o_{\min,a}, o_{\max,a}$ are computed from the training data."
        )
        for actor, (latex, descr) in HEALTH_REWARD_FORMULAS.items():
            st.markdown(f"#### {actor}")
            st.latex(latex)
            st.caption(descr)


# ======================================================================
# Radar + ranking table
# ======================================================================
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.subheader("Performance radar (mean across seeds)")
    st.caption(
        "Each axis is normalized to [0, 1] with 1 = best, using the same rule as the score. "
        "Hover any vertex for the mean and SEM across seeds."
    )

    available_actors = list(metrics_df["Actor/Criterion"].unique())
    default_actors = [a for a in available_actors
                      if a in {"Oracle", "Bank", "Applicant", "Maximin",
                               "Nash Bargaining", "Proportional Fairness",
                               "Parent", "Healthcare_Provider", "Policy_Maker"}]
    selected_actors = st.multiselect(
        "Actors / decision criteria to plot",
        options=available_actors,
        default=default_actors or available_actors[:5],
    )

    radar_norm = normalize_for_display(metrics_df, selected_metrics, ranking_criteria)
    fig = go.Figure()
    for actor in selected_actors:
        row_norm = radar_norm[radar_norm["Actor/Criterion"] == actor].iloc[0]
        values = [float(row_norm[m]) for m in selected_metrics]
        # Hover text: show mean +/- SEM in the raw units
        row_agg = agg[agg["Actor/Criterion"] == actor].iloc[0] if (agg["Actor/Criterion"] == actor).any() else None
        hover_text = []
        for m in selected_metrics:
            mean_v = row_agg[f"{m}_mean"] if row_agg is not None and f"{m}_mean" in row_agg else float("nan")
            sem_v = row_agg[f"{m}_sem"] if row_agg is not None and f"{m}_sem" in row_agg else float("nan")
            hover_text.append(f"{m}<br>raw mean: {mean_v:.4f}<br>±SEM: {sem_v:.4f}")
        # Close the loop
        values.append(values[0])
        hover_text.append(hover_text[0])
        theta = list(selected_metrics) + [selected_metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=theta, fill="toself",
            name=str(actor),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra>" + str(actor) + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        showlegend=True,
        # Legend below the chart (horizontal) so the polar plot owns the full
        # column width; otherwise the leftmost axis label gets clipped.
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        height=620,
        # Generous side margins so long axis labels (e.g. Conditional_Outcome_Parity,
        # Total_Profit) have room to render without being clipped.
        margin=dict(l=90, r=90, t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Live ranking (mean + SEM)")
    st.metric("Best actor / criterion under current weights & metrics", value=str(best))

    # Build a compact display table:
    # Actor | Weighted Normalized-Sum | <each selected metric mean (sem)>
    display_rows = []
    for _, ranked_row in ranked_df.iterrows():
        actor = ranked_row["Actor/Criterion"]
        ar = agg[agg["Actor/Criterion"] == actor]
        row = {
            "Actor/Criterion": actor,
            "Weighted Normalized-Sum": round(float(ranked_row["Weighted Normalized-Sum"]), 4),
        }
        for m in selected_metrics:
            if not ar.empty and f"{m}_mean" in ar.columns:
                mu = ar.iloc[0][f"{m}_mean"]
                sem = ar.iloc[0][f"{m}_sem"]
                row[m] = f"{mu:.3f} ± {sem:.3f}"
            else:
                row[m] = "n/a"
        display_rows.append(row)
    table = pd.DataFrame(display_rows)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    st.dataframe(table, use_container_width=True, hide_index=True)


st.markdown("---")


# ======================================================================
# Trade-off explorer (the user's "metric pair table" feature)
# ======================================================================
st.subheader("Trade-off explorer  -- two-metric weight grid")
st.caption(
    "Pick two metrics and a grid of weights. For each row, the table shows "
    "which compromise function (or actor) wins under that 2-metric mix. "
    "Useful for ablation tables in the paper."
)

tcol_a, tcol_b, tcol_c, tcol_d = st.columns([1, 1, 1, 1])
with tcol_a:
    metric_a = st.selectbox("Metric A", options=all_metrics,
                            index=all_metrics.index("Accuracy") if "Accuracy" in all_metrics else 0,
                            key="metric_a")
with tcol_b:
    metric_b_options = [m for m in all_metrics if m != metric_a]
    default_b = "Demographic_Parity" if "Demographic_Parity" in metric_b_options else metric_b_options[0]
    metric_b = st.selectbox("Metric B", options=metric_b_options,
                            index=metric_b_options.index(default_b), key="metric_b")
with tcol_c:
    n_steps = st.slider("Weight grid steps", min_value=3, max_value=21, value=5, step=2)
with tcol_d:
    restrict_compromise = st.checkbox(
        "Only compromise rules (exclude single-stakeholders)",
        value=True,
        help=(
            "When checked, only Maximin / Nash Bargaining / NSW / "
            "Kalai-Smorodinsky / Compromise Programming / Proportional Fairness "
            "compete. When unchecked, single-stakeholder actors (Bank, Applicant, "
            f"...) are also considered. Oracle is always excluded -- it's the "
            "idealized upper bound and would dominate by construction."
        ),
    )

weight_grid = np.linspace(0.0, 1.0, n_steps).tolist()
actors_filter = COMPROMISE_RULES if restrict_compromise else None
trade_table = build_tradeoff_table(
    metrics_df, metric_a, metric_b, weight_grid, ranking_criteria,
    actors_filter=actors_filter,
)
st.caption(
    f"Excluded from the table regardless of the checkbox: {sorted(TRADEOFF_EXCLUDE)}."
)
st.dataframe(trade_table, use_container_width=True, hide_index=True)

# Mini-summary: which compromise wins the most often?
if "Best compromise" in trade_table.columns:
    wins = trade_table["Best compromise"].value_counts()
    if not wins.empty:
        leader = wins.index[0]
        st.caption(
            f"Across the {len(weight_grid)} weight combinations, "
            f"**{leader}** is the most frequent winner ({wins.iloc[0]}/{len(weight_grid)})."
        )

st.markdown("---")


# ======================================================================
# Underlying model performance (aggregated)
# ======================================================================
st.subheader("Underlying model performance (mean across seeds)")

# Outcome metric: Accuracy for classification, MAE for causal regression
outcome_summaries = []
reward_mses_per_seed = {}
hparams_outcome = set()
hparams_reward = set()
for r in matched:
    s = r["summary"]
    o = s.get("outcome_model") or {}
    if isinstance(o.get("value"), (int, float)):
        outcome_summaries.append({
            "seed": r["seed"],
            "metric": o.get("metric"),
            "value": float(o["value"]),
        })
    for actor, mse in (s.get("reward_models_mse_per_actor") or {}).items():
        reward_mses_per_seed.setdefault(actor, []).append(float(mse))
    hparams_outcome.add(json.dumps(s.get("suggested_params_outcome"), sort_keys=True))
    hparams_reward.add(json.dumps(s.get("suggested_params_reward"), sort_keys=True))

perf_cols = st.columns(4)
if outcome_summaries:
    metric_name = outcome_summaries[0]["metric"] or "score"
    vals = np.array([o["value"] for o in outcome_summaries], dtype=float)
    mean_v = vals.mean()
    sem_v = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    perf_cols[0].metric(f"Outcome {metric_name}",
                        value=f"{mean_v:.4f}",
                        delta=f"±{sem_v:.4f} SEM")
perf_cols[1].metric("Seeds aggregated", len(matched))
perf_cols[2].metric("Reward variant", reward_variant)
perf_cols[3].metric("CV splits", matched[0]["cv_splits"])

if reward_mses_per_seed:
    st.markdown("**Reward-model test MSE per actor**  (mean ± SEM across seeds)")
    rows = []
    for actor, mses in sorted(reward_mses_per_seed.items()):
        arr = np.asarray(mses, dtype=float)
        sem_v = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append({"Actor": actor, "Mean MSE": float(arr.mean()), "SEM": sem_v, "n": len(arr)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Selected hyperparameters across the seeds"):
    st.json({
        "outcome_model (unique sets)": [json.loads(h) for h in hparams_outcome if h != "null"],
        "reward_models (unique sets)": [json.loads(h) for h in hparams_reward if h != "null"],
    })


# ======================================================================
# Comparison across all (reward_variant, model) buckets: Dirichlet ablation
# ======================================================================
st.markdown("---")
st.subheader("Cross-bucket comparison: which compromise rule is most robust?")
st.caption(
    "A Dirichlet weight sweep is run for **every** (reward_variant × model) bucket present in "
    "your `results/`, sharing the same random weight samples across buckets so the comparison is fair. "
    "Bars show the share of weight configurations where each rule is the consensus winner. "
    "Reference baselines (Oracle / Random / Outcome_Pred_Model / Outcome_Maxim) are excluded."
)


@st.cache_data
def _dirichlet_across_buckets(results_root_str: str, use_case: str, *,
                              n_samples: int, seed: int) -> dict:
    """For each (reward_variant, model) bucket, run an identical Dirichlet weight
    sweep on its mean-across-seeds metrics, and return a dict of per-bucket
    winner-share series."""
    runs = discover_runs(results_root_str, use_case, sample_size_only=10000)
    if not runs:
        return {}
    buckets: dict[tuple[str, str], list[str]] = {}
    for r in runs:
        buckets.setdefault((r["reward_variant"], r["model"]), []).append(r["csv"])

    use_case_cfg = _load_use_case_yaml(use_case)
    ranking_criteria = dict(use_case_cfg["criteria"]["ranking_criteria"])
    metrics = list(use_case_cfg["criteria"]["metrics_for_evaluation"])

    weight_configs = sweep_dirichlet(metrics, n_samples=n_samples, alpha=1.0, seed=seed)
    exclude = set(ABLATE_EXCLUDE)

    out = {}
    for (rv, model), csvs in buckets.items():
        per_seed_dfs = []
        for p in csvs:
            df = pd.read_csv(p)
            drop = [c for c in df.columns if any(k in c for k in (" Normalized", " Rank", "Weighted Normalized-Sum"))]
            per_seed_dfs.append(df.drop(columns=drop, errors="ignore"))
        if not per_seed_dfs:
            continue
        long_df = ablate_to_long(per_seed_dfs, weight_configs, ranking_criteria, metrics, exclude)
        winners = long_df[long_df["is_consensus_winner"]]
        counts = winners["Actor/Criterion"].value_counts()
        stability = winners.groupby("Actor/Criterion")["consensus_winner_stability"].mean()
        out[(rv, model)] = {
            "counts": counts,
            "stability": stability,
            "n_configs": int(len(weight_configs)),
            "n_seeds": len(csvs),
        }
    return out


col_ctrl, col_info = st.columns([1, 3])
with col_ctrl:
    n_samples = st.slider("Dirichlet samples", min_value=100, max_value=1000, value=300, step=100,
                          help="More samples → tighter winner-share estimates but slower computation.")
    sweep_seed = st.number_input("Sweep seed", value=0, min_value=0, max_value=2_000_000_000, step=1,
                                 help="Random seed for the Dirichlet samples. All buckets share the same seed.")

with col_info:
    st.info(
        "The same `n` samples are drawn from the simplex and replayed against every bucket's "
        "(mean-across-seeds) metrics. So differences in the bars below reflect bucket effects, "
        "not weight-sampling noise."
    )

per_bucket = _dirichlet_across_buckets(
    str(results_root), use_case, n_samples=int(n_samples), seed=int(sweep_seed),
)

if len(per_bucket) < 2:
    st.warning(
        f"Only {len(per_bucket)} bucket(s) discovered under `results/{use_case}/`. "
        "The cross-bucket comparison needs at least two; train more configurations to enable it."
    )
else:
    # Build a grouped bar chart: y = actors, x = winner share %, one bar per bucket.
    bucket_keys = sorted(per_bucket.keys())  # ((reward, model), …)
    bucket_labels = [f"{rv} / {model}" for (rv, model) in bucket_keys]
    palette = list(_plt.get_cmap("tab10").colors)
    bucket_colors = [palette[i % len(palette)] for i in range(len(bucket_keys))]

    # Union of actors across buckets, ordered by total wins (descending) so the
    # most-frequent winner sits at the top.
    union_counts: dict[str, int] = {}
    for k in bucket_keys:
        for a, v in per_bucket[k]["counts"].items():
            union_counts[a] = union_counts.get(a, 0) + int(v)
    actors_ordered = sorted(union_counts, key=lambda a: -union_counts[a])

    fig = Figure(figsize=(11, max(4.5, 0.42 * len(actors_ordered) + 1.6)))
    ax = fig.subplots()
    ax.grid(False)
    y = np.arange(len(actors_ordered))
    n_buckets = len(bucket_keys)
    bar_h = 0.78 / n_buckets

    for i, (k, label) in enumerate(zip(bucket_keys, bucket_labels)):
        info = per_bucket[k]
        total = info["n_configs"]
        shares = np.array([info["counts"].get(a, 0) / max(total, 1) * 100 for a in actors_ordered])
        offset = (i - (n_buckets - 1) / 2) * bar_h
        ax.barh(
            y + offset, shares, height=bar_h,
            color=bucket_colors[i],
            edgecolor="white", linewidth=0.6, alpha=0.92,
            label=f"{label}  (n_seeds={info['n_seeds']})",
        )
        max_share = max(shares) if len(shares) else 0
        for j, share in enumerate(shares):
            if share >= max_share * 0.06:
                ax.text(share + max_share * 0.008, y[j] + offset,
                        f"{share:.0f}%", va="center", ha="left",
                        fontsize=8.5, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(actors_ordered)
    ax.invert_yaxis()
    ax.set_xlabel("Share of weight configurations won (%)", fontsize=11)
    ax.set_title(f"{use_case.title()} -- cross-bucket Dirichlet winner share  "
                 f"({n_samples} configs over the {len(use_case_cfg['criteria']['metrics_for_evaluation'])}-metric simplex)",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.25, axis="x", linestyle="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=True, edgecolor="lightgray", title="Bucket")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Companion table with stability and counts so the user can dig in.
    with st.expander("Per-bucket numerical breakdown"):
        tbl_rows = []
        for k, label in zip(bucket_keys, bucket_labels):
            info = per_bucket[k]
            for actor in actors_ordered:
                wins = int(info["counts"].get(actor, 0))
                share = wins / max(info["n_configs"], 1) * 100
                stab = float(info["stability"].get(actor, float("nan")))
                tbl_rows.append({
                    "Bucket": label,
                    "Actor / criterion": actor,
                    "Wins": wins,
                    "Share (%)": round(share, 1),
                    "Mean stability": round(stab, 2) if not np.isnan(stab) else None,
                    "n_seeds": info["n_seeds"],
                })
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)


st.markdown(
    """
    ---
    **Accountability note.** This dashboard is *read-only* over training
    artefacts. It uses the same normalization as the original framework, so
    any ranking shown here can be reproduced bit-for-bit by rerunning
    `main.py` with the corresponding `ranking_weights.*` overrides.
    Aggregation is unweighted across the seeds matching the chosen
    configuration; SEM uses Bessel-corrected std divided by sqrt(n_seeds).
    """
)
