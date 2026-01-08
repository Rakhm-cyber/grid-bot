from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .config import Config
from .pipeline import run_analysis


def run_pair_task(
    cfg_dict: Dict[str, Any],
    s1: str,
    s2: str,
    start: str | None,
    end: str | None,
    progress: bool = False,
) -> None:
    base_cfg = Config(**cfg_dict)
    pair_cfg = replace(
        base_cfg,
        symbol1=s1,
        symbol2=s2,
        output_path=str(Path(base_cfg.output_path) / f"{s1}_{s2}"),
    )
    run_analysis(pair_cfg, start=start, end=end, progress=progress)
