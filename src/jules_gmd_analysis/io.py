from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import xarray as xr


def read_site_list(path: str | Path, site_column: str = "Site") -> pd.DataFrame:
    sites = pd.read_csv(path)
    if site_column not in sites.columns:
        raise KeyError(f"Missing site column {site_column!r} in {path}")
    return sites


def discover_netcdf_files(
    root: str | Path,
    site_names: list[str],
    *,
    layout: str,
    pattern_template: str,
) -> dict[str, list[Path]]:
    """Find NetCDF files by site.

    Parameters
    ----------
    layout
        Either "flat" for files directly under root, or "site_subdirs" for one
        directory per site.
    pattern_template
        Regex template that can use the named field {site}.
    """
    root = Path(root).expanduser()
    out: dict[str, list[Path]] = {}
    for site in site_names:
        search_dir = root if layout == "flat" else root / site
        regex = re.compile(pattern_template.format(site=re.escape(site)), re.IGNORECASE)
        if not search_dir.exists():
            out[site] = []
            continue
        out[site] = sorted(
            path for path in search_dir.iterdir()
            if path.is_file() and regex.match(path.name)
        )
    return out


def open_site_datasets(files_by_site: dict[str, list[Path]]) -> dict[str, list[xr.Dataset]]:
    datasets: dict[str, list[xr.Dataset]] = {}
    for site, files in files_by_site.items():
        opened = []
        for path in files:
            try:
                opened.append(xr.open_dataset(path))
            except Exception as exc:
                print(f"[WARN] Could not open {path}: {exc}")
        datasets[site] = opened
    return datasets


def load_standard_inputs(cfg, site_names: list[str]) -> dict[str, dict[str, list[xr.Dataset]]]:
    """Load the NetCDF groups used by the analysis."""
    model_pattern = r"local_{site}_.*{site}\.nc$"
    flat_pattern = r"{site}_.*\.nc$"

    groups = {
        "met": (
            cfg.resolve_path("met_root"),
            "flat",
            flat_pattern,
        ),
        "obs": (
            cfg.resolve_path("obs_root"),
            "site_subdirs",
            model_pattern,
        ),
        "pmodel": (
            cfg.resolve_path("pmodel_root"),
            "site_subdirs",
            model_pattern,
        ),
        "multi": (
            cfg.resolve_path("multilayer_root"),
            "site_subdirs",
            model_pattern,
        ),
        "multi13": (
            cfg.resolve_path("multi13_root"),
            "site_subdirs",
            model_pattern,
        ),
    }

    loaded = {}
    for name, (root, layout, pattern) in groups.items():
        files = discover_netcdf_files(
            root,
            site_names,
            layout=layout,
            pattern_template=pattern,
        )
        loaded[name] = open_site_datasets(files)
    return loaded
