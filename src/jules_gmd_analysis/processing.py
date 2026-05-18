from __future__ import annotations

import numpy as np
import pandas as pd


def _site_coordinates(ds) -> tuple[float, float]:
    lat = float(ds["latitude"].load().values.flat[0]) if "latitude" in ds else np.nan
    lon = float(ds["longitude"].load().values.flat[0]) if "longitude" in ds else np.nan
    return lat, lon


def _to_series(ds, var: str, *, soil_layer: int | None = None) -> pd.DataFrame | None:
    if var not in ds:
        return None
    arr = ds[var]
    if soil_layer is not None and "soil" in arr.dims:
        arr = arr.isel(soil=soil_layer)
    arr = arr.load()
    return pd.DataFrame({
        "time": pd.to_datetime(arr["time"].values.flatten()),
        var: arr.values.flatten(),
    })


def _daily_mean(frame: pd.DataFrame, value_col: str, shift_minutes: int) -> pd.Series:
    data = frame.copy()
    data["time"] = pd.to_datetime(data["time"])
    if shift_minutes:
        data["time"] = data["time"] - pd.Timedelta(minutes=shift_minutes)
    data["date"] = data["time"].dt.floor("D")
    return data.groupby("date")[value_col].mean()


def extract_daily_site_data(
    datasets_by_site: dict,
    site_names: list[str],
    *,
    shift_minutes: int,
    include_fsmc: bool = False,
    include_lai: bool = False,
) -> dict[str, list[dict]]:
    """Extract daily GPP, latent heat, FSmc, and/or LAI from xarray datasets."""
    results: dict[str, list[dict]] = {}
    for site, datasets in datasets_by_site.items():
        if site not in site_names:
            continue
        records = []
        for ds in datasets:
            try:
                lat, lon = _site_coordinates(ds)
                series = {}

                gpp = _to_series(ds, "GPP")
                if gpp is not None:
                    gpp.loc[gpp["GPP"] < 0, "GPP"] = 0
                    series["daily_GPP"] = _daily_mean(gpp, "GPP", shift_minutes)

                le = _to_series(ds, "Qle")
                if le is not None:
                    series["daily_LE"] = _daily_mean(le, "Qle", shift_minutes)

                if include_fsmc:
                    fsmc = _to_series(ds, "fsmc", soil_layer=0)
                    if fsmc is not None:
                        series["daily_fsmc"] = _daily_mean(fsmc, "fsmc", shift_minutes)

                if include_lai:
                    lai = _to_series(ds, "LAI")
                    if lai is not None:
                        series["daily_LAI"] = _daily_mean(lai, "LAI", shift_minutes)

                if not series:
                    continue

                common_dates = None
                for value in series.values():
                    common_dates = value.index if common_dates is None else common_dates.intersection(value.index)
                if common_dates is None or len(common_dates) == 0:
                    continue

                item = {
                    "time": pd.to_datetime(common_dates),
                    "longitude": lon,
                    "latitude": lat,
                }
                for key, value in series.items():
                    item[key] = value.loc[common_dates].values
                records.append(item)
            except Exception as exc:
                print(f"[WARN] Skipping {site}: {exc}")
        results[site] = records
    return results


def extract_halfhourly_site_data(
    datasets_by_site: dict,
    site_names: list[str],
    *,
    shift_minutes: int,
    include_fsmc: bool = False,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for site, datasets in datasets_by_site.items():
        if site not in site_names:
            continue
        records = []
        for ds in datasets:
            try:
                lat, lon = _site_coordinates(ds)
                gpp = _to_series(ds, "GPP")
                le = _to_series(ds, "Qle")
                if gpp is None or le is None:
                    continue
                gpp.loc[gpp["GPP"] < 0, "GPP"] = 0

                def prep(frame, var):
                    out = frame.rename(columns={var: var}).copy()
                    out["time"] = pd.to_datetime(out["time"]).dt.round("30min")
                    if shift_minutes:
                        out["time"] = out["time"] - pd.Timedelta(minutes=shift_minutes)
                    return out

                merged = prep(gpp, "GPP").merge(prep(le, "Qle"), on="time", how="inner")
                if include_fsmc:
                    fsmc = _to_series(ds, "fsmc", soil_layer=0)
                    if fsmc is not None:
                        merged = merged.merge(prep(fsmc, "fsmc"), on="time", how="inner")
                records.append({
                    "time": merged["time"].values,
                    "hh_GPP": merged["GPP"].values,
                    "hh_LE": merged["Qle"].values,
                    "hh_fsmc": merged["fsmc"].values if "fsmc" in merged else np.full(len(merged), np.nan),
                    "longitude": lon,
                    "latitude": lat,
                })
            except Exception as exc:
                print(f"[WARN] Skipping {site}: {exc}")
        results[site] = records
    return results


def daily_dict_to_dataframe(data: dict, suffix: str) -> pd.DataFrame:
    rows = []
    for site, items in data.items():
        for item in items:
            frame = pd.DataFrame({"Time": pd.to_datetime(item["time"])})
            for key, value in item.items():
                if key.startswith("daily_"):
                    frame[f"{key}_{suffix}"] = value
            frame["Site"] = site
            frame["latitude"] = item.get("latitude", np.nan)
            frame["longitude"] = item.get("longitude", np.nan)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_daily_tables(
    obs_df: pd.DataFrame,
    pmodel_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    multi13_df: pd.DataFrame,
    lai_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = obs_df.copy()
    for table in [pmodel_df, multi_df, multi13_df]:
        if not table.empty:
            merged = merged.merge(table, on=["Site", "Time"], how="inner", suffixes=("", "_drop"))
            merged = merged[[c for c in merged.columns if not c.endswith("_drop")]]
    if lai_df is not None and not lai_df.empty:
        merged = merged.merge(lai_df[["Site", "Time", "daily_LAI_met"]], on=["Site", "Time"], how="left")
        merged = merged.rename(columns={"daily_LAI_met": "LAI"})
    return merged


def apply_gpp_conversions(df: pd.DataFrame, *, obs_factor: float, model_factor: float) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "daily_GPP_obs":
            out[col] = out[col] * obs_factor
        elif col.startswith("daily_GPP_"):
            out[col] = out[col] * model_factor
    return out


def aggregate_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"])
    if period == "monthly":
        out["Period"] = out["Time"].dt.to_period("M").dt.to_timestamp()
    elif period == "yearly":
        out["Period"] = out["Time"].dt.to_period("Y").dt.to_timestamp()
    else:
        raise ValueError("period must be 'monthly' or 'yearly'")

    value_cols = [c for c in out.columns if c.startswith("daily_") or c == "LAI"]
    group_cols = ["Site", "Period"]
    keep_cols = [c for c in ["latitude", "longitude"] if c in out.columns]
    grouped = out.groupby(group_cols, as_index=False)[value_cols + keep_cols].mean()
    grouped = grouped.rename(columns={
        "Period": "Time",
        "daily_GPP_obs": "Daily_Mean_GPP_obs",
        "daily_GPP_pmodel": "Daily_Mean_GPP_pmodel",
        "daily_GPP_multi": "Daily_Mean_GPP_multi",
        "daily_GPP_multi13": "Daily_Mean_GPP_multi13",
        "daily_LE_obs": "Daily_Mean_LE_obs",
        "daily_LE_pmodel": "Daily_Mean_LE_pmodel",
        "daily_LE_multi": "Daily_Mean_LE_multi",
        "daily_LE_multi13": "Daily_Mean_LE_multi13",
        "daily_fsmc_pmodel": "Daily_Mean_fsmc_pmodel",
    })
    grouped["Month"] = pd.to_datetime(grouped["Time"]).dt.month
    grouped["Year"] = pd.to_datetime(grouped["Time"]).dt.year
    return grouped


def merge_halfhourly_datasets(obs_data, pmodel_data, multi_data, multi13_data) -> pd.DataFrame:
    records = []
    for site in obs_data:
        obs_items = obs_data.get(site, [])
        if not obs_items:
            continue
        for idx, obs_item in enumerate(obs_items):
            obs = pd.DataFrame({
                "Time": pd.to_datetime(obs_item["time"]).round("min"),
                "GPP_obs": obs_item.get("hh_GPP"),
                "LE_obs": obs_item.get("hh_LE"),
                "latitude": obs_item.get("latitude"),
                "longitude": obs_item.get("longitude"),
            })

            def model_frame(items, col):
                if idx >= len(items):
                    return None
                item = items[idx]
                return pd.DataFrame({
                    "Time": pd.to_datetime(item["time"]).round("min"),
                    col: item.get("hh_GPP"),
                })

            merged = obs
            for items, col in [
                (pmodel_data.get(site, []), "GPP_pmodel"),
                (multi_data.get(site, []), "GPP_multi"),
                (multi13_data.get(site, []), "GPP_multi13"),
            ]:
                frame = model_frame(items, col)
                if frame is not None:
                    merged = merged.merge(frame, on="Time", how="left")
            merged["Site"] = site
            records.append(merged)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
