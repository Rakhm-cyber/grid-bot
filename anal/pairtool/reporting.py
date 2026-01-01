from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ReportPaths:
    root: Path
    plots: Path
    result_json: Optional[Path] = None
    aligned_csv: Optional[Path] = None
    rolling_csv: Optional[Path] = None
    breaks_csv: Optional[Path] = None
    html_report: Optional[Path] = None


def ensure_dirs(base: Path) -> ReportPaths:
    base.mkdir(parents=True, exist_ok=True)
    plots = base / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return ReportPaths(root=base, plots=plots)


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True)


def plot_series(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["log_y"], label="log_y")
    ax.plot(df.index, df["log_x"], label="log_x")
    ax.set_title("Log Prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spread(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["spread"], label="spread")
    rolling = df["spread"].rolling(100).mean()
    ax.plot(df.index, rolling, label="rolling_mean")
    ax.set_title("Spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_zscore(df: pd.DataFrame, path: Path) -> None:
    spread = df["spread"]
    z = (spread - spread.mean()) / spread.std(ddof=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, z, label="zscore")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Spread Z-Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rolling(df: pd.DataFrame, path: Path, column: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[column], label=column)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_chow(breaks: pd.DataFrame, path: Path) -> None:
    if breaks.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(breaks["break_time"], breaks["p_value"], label="p_value")
    ax.set_title("Chow Test p-values")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_html_report(path: Path, title: str, summary: Dict[str, str]) -> None:
    rows = "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items()])
    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h1>{title}</h1>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Key</th><th>Value</th></tr>
{rows}
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
