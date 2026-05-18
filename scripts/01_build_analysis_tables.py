#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jules_gmd_analysis.config import ensure_output_dirs, load_config
from jules_gmd_analysis.io import load_standard_inputs, read_site_list
from jules_gmd_analysis.processing import (
    aggregate_period,
    apply_gpp_conversions,
    daily_dict_to_dataframe,
    extract_daily_site_data,
    merge_daily_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily/monthly/yearly analysis tables.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    site_column = cfg.site_selection.get("site_column", "Site")
    site_info = read_site_list(cfg.resolve_path("site_metadata"), site_column=site_column)
    site_names = site_info[site_column].dropna().astype(str).tolist()
    exclude = set(cfg.site_selection.get("exclude_sites", []))
    site_names = [site for site in site_names if site not in exclude]

    datasets = load_standard_inputs(cfg, site_names)
    shifts = cfg.time_alignment

    obs_daily = extract_daily_site_data(
        datasets["obs"], site_names, shift_minutes=shifts.get("obs_shift_minutes", 0)
    )
    pmodel_daily = extract_daily_site_data(
        datasets["pmodel"], site_names, shift_minutes=shifts.get("pmodel_shift_minutes", 30), include_fsmc=True
    )
    multi_daily = extract_daily_site_data(
        datasets["multi"], site_names, shift_minutes=shifts.get("multilayer_shift_minutes", 30)
    )
    multi13_daily = extract_daily_site_data(
        datasets["multi13"], site_names, shift_minutes=shifts.get("multi13_shift_minutes", 30)
    )
    lai_daily = extract_daily_site_data(
        datasets["met"], site_names, shift_minutes=shifts.get("met_shift_minutes", 0), include_lai=True
    )

    obs_df = daily_dict_to_dataframe(obs_daily, "obs")
    pmodel_df = daily_dict_to_dataframe(pmodel_daily, "pmodel")
    multi_df = daily_dict_to_dataframe(multi_daily, "multi")
    multi13_df = daily_dict_to_dataframe(multi13_daily, "multi13")
    lai_df = daily_dict_to_dataframe(lai_daily, "met")

    daily = merge_daily_tables(obs_df, pmodel_df, multi_df, multi13_df, lai_df)
    daily = apply_gpp_conversions(
        daily,
        obs_factor=cfg.unit_conversions.get("obs_gpp_factor", 1.0),
        model_factor=cfg.unit_conversions.get("model_gpp_factor", 1.0),
    )

    processed = cfg.resolve_path("processed_dir")
    daily.to_csv(processed / "daily_analysis.csv", index=False)
    aggregate_period(daily, "monthly").to_csv(processed / "monthly_analysis.csv", index=False)
    aggregate_period(daily, "yearly").to_csv(processed / "yearly_analysis.csv", index=False)
    print(f"Wrote analysis tables to {processed}")


if __name__ == "__main__":
    main()
