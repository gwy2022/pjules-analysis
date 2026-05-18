#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jules_gmd_analysis.config import ensure_output_dirs, load_config
from jules_gmd_analysis.plotting import plot_binned_metric, plot_gpp_scatter_matrix, plot_monthly_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make publication-ready analysis figures.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    processed = cfg.resolve_path("processed_dir")
    tables = cfg.resolve_path("tables_dir")
    figures = cfg.resolve_path("figures_dir")
    dpi = int(cfg.figures.get("dpi", 300))

    monthly = pd.read_csv(processed / "monthly_analysis.csv", parse_dates=["Time"])
    yearly = pd.read_csv(processed / "yearly_analysis.csv", parse_dates=["Time"])

    plot_gpp_scatter_matrix(
        yearly,
        monthly,
        out_path=figures / "gpp_scatter_yearly_monthly.png",
        dpi=dpi,
    )

    gpp_models = [
        "Daily_Mean_GPP_pmodel",
        "Daily_Mean_GPP_multi",
        "Daily_Mean_GPP_multi13",
    ]
    plot_monthly_cycle(
        monthly,
        variable="GPP",
        model_cols=gpp_models,
        out_path=figures / "gpp_monthly_cycle.png",
        dpi=dpi,
    )

    for table, x_label, output in [
        ("gpp_rmse_by_observed_gpp_bin.csv", "Observed GPP bin center", "gpp_rmse_by_observed_gpp_bin.png"),
        ("gpp_rmse_by_lai_bin.csv", "LAI bin center", "gpp_rmse_by_lai_bin.png"),
        ("gpp_rmse_by_fsmc_bin.csv", "FSmc bin center", "gpp_rmse_by_fsmc_bin.png"),
    ]:
        path = tables / table
        if path.exists():
            stats = pd.read_csv(path)
            plot_binned_metric(
                stats,
                metric="rmse",
                xlabel=x_label,
                ylabel="RMSE",
                out_path=figures / output,
                dpi=dpi,
            )

    print(f"Wrote figures to {figures}")


if __name__ == "__main__":
    main()
