from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def model_skill_table(
    df: pd.DataFrame,
    *,
    obs_col: str,
    model_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_cols, key))
        for model_col in model_cols:
            tmp = sub[[obs_col, model_col]].dropna()
            if len(tmp) < 2:
                continue
            obs = tmp[obs_col].to_numpy()
            pred = tmp[model_col].to_numpy()
            rmse = mean_squared_error(obs, pred, squared=False)
            mae = mean_absolute_error(obs, pred)
            bias = float(np.mean(pred - obs))
            rows.append({
                **group_values,
                "model": model_col,
                "n": len(tmp),
                "mean_obs": float(np.mean(obs)),
                "mean_model": float(np.mean(pred)),
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "nrmse": rmse / float(np.mean(obs)) if np.mean(obs) != 0 else np.nan,
                "r2": r2_score(obs, pred),
                "r": float(np.corrcoef(obs, pred)[0, 1]),
            })
    return pd.DataFrame(rows)


def classify_ai(ai: float) -> str:
    if pd.isna(ai):
        return "Unknown"
    if ai < 2:
        return "Humid (AI < 2)"
    if ai < 5:
        return "Semi-arid (2 <= AI < 5)"
    return "Arid (AI >= 5)"


def add_ai_classes(df: pd.DataFrame, site_ai: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Site", "AI", "Latitude", "latitude"] if c in site_ai.columns]
    out = df.merge(site_ai[cols].drop_duplicates("Site"), on="Site", how="left")
    if "AI" in out:
        out["AI_class"] = out["AI"].apply(classify_ai)
    lat_col = "Latitude" if "Latitude" in out else "latitude"
    if lat_col in out:
        out["Hemisphere"] = np.where(out[lat_col] >= 0, "Northern", "Southern")
    return out


def add_season(df: pd.DataFrame, *, hemisphere_aware: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["Month"] = pd.to_datetime(out["Time"]).dt.month
    north = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    south = {
        12: "JJA", 1: "JJA", 2: "JJA",
        3: "SON", 4: "SON", 5: "SON",
        6: "DJF", 7: "DJF", 8: "DJF",
        9: "MAM", 10: "MAM", 11: "MAM",
    }
    if hemisphere_aware and "Hemisphere" in out:
        out["Season"] = [
            (south if hemi == "Southern" else north)[month]
            for month, hemi in zip(out["Month"], out["Hemisphere"])
        ]
    else:
        out["Season"] = out["Month"].map(north)
    return out


def binned_error_stats(
    df: pd.DataFrame,
    *,
    obs_col: str,
    model_cols: list[str],
    by_col: str,
    bin_width: float,
    min_count: int = 30,
) -> pd.DataFrame:
    tmp = df[[obs_col, by_col] + model_cols].dropna(subset=[obs_col, by_col]).copy()
    max_value = float(np.nanmax(tmp[by_col]))
    bins = np.arange(0, max_value + bin_width, bin_width)
    tmp["bin"] = pd.cut(tmp[by_col], bins=bins, include_lowest=True)
    rows = []
    for model_col in model_cols:
        for bin_label, sub in tmp.groupby("bin", observed=True):
            pair = sub[[obs_col, model_col]].dropna()
            if len(pair) < min_count:
                continue
            obs = pair[obs_col].to_numpy()
            pred = pair[model_col].to_numpy()
            rows.append({
                "model": model_col,
                "bin": str(bin_label),
                "bin_center": float(bin_label.mid),
                "n": len(pair),
                "bias": float(np.mean(pred - obs)),
                "mae": mean_absolute_error(obs, pred),
                "rmse": mean_squared_error(obs, pred, squared=False),
            })
    return pd.DataFrame(rows)
