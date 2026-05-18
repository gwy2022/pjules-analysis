from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


MODEL_LABELS = {
    "Daily_Mean_GPP_pmodel": "P-JULES",
    "Daily_Mean_GPP_multi": "JULES multi-layer",
    "Daily_Mean_GPP_multi13": "JULES 13-PFT",
    "Daily_Mean_LE_pmodel": "P-JULES",
    "Daily_Mean_LE_multi": "JULES multi-layer",
    "Daily_Mean_LE_multi13": "JULES 13-PFT",
}


def density_scatter(
    ax,
    data: pd.DataFrame,
    *,
    obs_col: str,
    model_col: str,
    xlabel: str,
    ylabel: str,
    limit: float | None = None,
):
    tmp = data[[obs_col, model_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None

    x = tmp[obs_col].to_numpy()
    y = tmp[model_col].to_numpy()
    xy = np.vstack([x, y])
    try:
        density = gaussian_kde(xy)(xy)
    except Exception:
        density = np.ones_like(x)
    order = density.argsort()
    x, y, density = x[order], y[order], density[order]

    scatter = ax.scatter(x, y, c=density, s=12, cmap="viridis", alpha=0.75, edgecolors="none")
    max_value = limit or max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.05
    ax.plot([0, max_value], [0, max_value], color="0.35", lw=1.0, ls="--")
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(tmp) > 1:
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        pred = model.predict(x.reshape(-1, 1))
        rmse = mean_squared_error(x, y, squared=False)
        text = (
            f"n = {len(tmp)}\n"
            f"R2 = {r2_score(x, y):.2f}\n"
            f"RMSE = {rmse:.2f}\n"
            f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    return scatter


def plot_gpp_scatter_matrix(
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    model_cols = [
        "Daily_Mean_GPP_pmodel",
        "Daily_Mean_GPP_multi",
        "Daily_Mean_GPP_multi13",
    ]
    fig, axes = plt.subplots(len(model_cols), 2, figsize=(10, 13), constrained_layout=True)
    last = None
    for row, model_col in enumerate(model_cols):
        for col, (label, frame, limit) in enumerate([
            ("Yearly GPP", yearly, 15),
            ("Monthly GPP", monthly, 20),
        ]):
            ax = axes[row, col]
            last = density_scatter(
                ax,
                frame,
                obs_col="Daily_Mean_GPP_obs",
                model_col=model_col,
                xlabel="Observed GPP (g C m-2 d-1)",
                ylabel=f"{MODEL_LABELS.get(model_col, model_col)} GPP",
                limit=limit,
            )
            if row == 0:
                ax.set_title(label, fontweight="bold")
    if last is not None:
        fig.colorbar(last, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, label="Point density")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_binned_metric(
    stats: pd.DataFrame,
    *,
    metric: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for model, sub in stats.groupby("model"):
        sub = sub.sort_values("bin_center")
        ax.plot(
            sub["bin_center"],
            sub[metric],
            marker="o",
            lw=1.7,
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_cycle(
    monthly: pd.DataFrame,
    *,
    variable: str,
    model_cols: list[str],
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    obs_col = f"Daily_Mean_{variable}_obs"
    cycle = monthly.groupby("Month")[[obs_col] + model_cols].mean(numeric_only=True).reset_index()
    ax.plot(cycle["Month"], cycle[obs_col], color="black", lw=2.2, label="Observed")
    for col in model_cols:
        ax.plot(cycle["Month"], cycle[col], lw=1.8, marker="o", label=MODEL_LABELS.get(col, col))
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{variable} daily mean")
    ax.set_xticks(range(1, 13))
    ax.legend(frameon=False, ncols=2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
