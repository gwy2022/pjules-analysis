#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jules_gmd_analysis.config import ensure_output_dirs, load_config
from jules_gmd_analysis.metrics import add_ai_classes, add_season, binned_error_stats, model_skill_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute GPP and LE model evaluation metrics.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    processed = cfg.resolve_path("processed_dir")
    tables = cfg.resolve_path("tables_dir")
    monthly = pd.read_csv(processed / "monthly_analysis.csv", parse_dates=["Time"])
    yearly = pd.read_csv(processed / "yearly_analysis.csv", parse_dates=["Time"])
    site_ai = pd.read_csv(cfg.resolve_path("site_ai"))

    gpp_models = [
        "Daily_Mean_GPP_pmodel",
        "Daily_Mean_GPP_multi",
        "Daily_Mean_GPP_multi13",
    ]
    le_models = [
        "Daily_Mean_LE_pmodel",
        "Daily_Mean_LE_multi",
        "Daily_Mean_LE_multi13",
    ]

    monthly_ai = add_season(add_ai_classes(monthly, site_ai))
    yearly_ai = add_ai_classes(yearly, site_ai)

    model_skill_table(
        monthly_ai, obs_col="Daily_Mean_GPP_obs", model_cols=gpp_models
    ).to_csv(tables / "gpp_monthly_metrics.csv", index=False)
    model_skill_table(
        yearly_ai, obs_col="Daily_Mean_GPP_obs", model_cols=gpp_models
    ).to_csv(tables / "gpp_yearly_metrics.csv", index=False)
    model_skill_table(
        monthly_ai, obs_col="Daily_Mean_GPP_obs", model_cols=gpp_models, group_cols=["AI_class"]
    ).to_csv(tables / "gpp_monthly_metrics_by_ai.csv", index=False)
    model_skill_table(
        monthly_ai, obs_col="Daily_Mean_GPP_obs", model_cols=gpp_models, group_cols=["Season"]
    ).to_csv(tables / "gpp_monthly_metrics_by_season.csv", index=False)

    available_le = [col for col in le_models if col in monthly_ai.columns]
    if available_le and "Daily_Mean_LE_obs" in monthly_ai.columns:
        model_skill_table(
            monthly_ai, obs_col="Daily_Mean_LE_obs", model_cols=available_le
        ).to_csv(tables / "le_monthly_metrics.csv", index=False)

    binned_error_stats(
        monthly_ai,
        obs_col="Daily_Mean_GPP_obs",
        model_cols=gpp_models,
        by_col="Daily_Mean_GPP_obs",
        bin_width=2.0,
    ).to_csv(tables / "gpp_rmse_by_observed_gpp_bin.csv", index=False)

    if "LAI" in monthly_ai:
        binned_error_stats(
            monthly_ai,
            obs_col="Daily_Mean_GPP_obs",
            model_cols=gpp_models,
            by_col="LAI",
            bin_width=0.5,
        ).to_csv(tables / "gpp_rmse_by_lai_bin.csv", index=False)

    if "Daily_Mean_fsmc_pmodel" in monthly_ai:
        binned_error_stats(
            monthly_ai,
            obs_col="Daily_Mean_GPP_obs",
            model_cols=gpp_models,
            by_col="Daily_Mean_fsmc_pmodel",
            bin_width=0.05,
        ).to_csv(tables / "gpp_rmse_by_fsmc_bin.csv", index=False)

    print(f"Wrote metrics to {tables}")


if __name__ == "__main__":
    main()
