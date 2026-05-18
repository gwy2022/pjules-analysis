from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AnalysisConfig:
    raw: dict
    config_path: Path
    project_dir: Path

    @property
    def paths(self) -> dict:
        return self.raw["paths"]

    @property
    def site_selection(self) -> dict:
        return self.raw.get("site_selection", {})

    @property
    def time_alignment(self) -> dict:
        return self.raw.get("time_alignment", {})

    @property
    def unit_conversions(self) -> dict:
        return self.raw.get("unit_conversions", {})

    @property
    def figures(self) -> dict:
        return self.raw.get("figures", {})

    def resolve_path(self, key: str) -> Path:
        value = Path(self.paths[key]).expanduser()
        if value.is_absolute():
            return value
        return (self.project_dir / value).resolve()


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    project_dir = config_path.parents[1]
    return AnalysisConfig(raw=raw, config_path=config_path, project_dir=project_dir)


def ensure_output_dirs(cfg: AnalysisConfig) -> None:
    for key in ["processed_dir", "tables_dir", "figures_dir"]:
        cfg.resolve_path(key).mkdir(parents=True, exist_ok=True)
