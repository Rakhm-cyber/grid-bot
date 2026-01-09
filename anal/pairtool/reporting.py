from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


def _fmt(val: Any) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def write_html_report_from_result(path: Path, title: str, result: Dict[str, Any]) -> None:
    meta = result.get("metadata", {})
    tests = result.get("static_tests", {})
    spread = result.get("spread_model", {})
    mean_rev = result.get("mean_reversion", {})
    rolling = result.get("rolling", {}).get("summary", {})
    breaks = result.get("breaks", {}).get("chow", {})
    trade = result.get("tradeability", {})

    def table(rows: Dict[str, Any]) -> str:
        return "\n".join([f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>" for k, v in rows.items()])

    corr = tests.get("correlations", {})
    station = tests.get("stationarity", {})
    coint = tests.get("cointegration", {})
    norm = tests.get("normality", {})
    caus = tests.get("causality", {})

    html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h1>{title}</h1>

<h2>Summary</h2>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "symbols": "/".join(meta.get("symbols", [])),
    "timeframe": meta.get("timeframe"),
    "nobs": meta.get("nobs"),
    "score": trade.get("score"),
    "decision": trade.get("decision"),
})}
</table>

<h2>What was computed and why</h2>
<p>We test whether two assets move together (correlations), whether their prices or spread are stable
over time (stationarity), and whether a linear combination is mean-reverting (cointegration and
half-life). These checks help decide if a pair is suitable for mean-reversion strategies.</p>

<h2>Spread model</h2>
<p>Spread is computed as: spread = log(price_y) - (alpha + beta * log(price_x)).</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "alpha": spread.get("alpha"),
    "beta": spread.get("beta"),
    "method": spread.get("method"),
    "include_intercept": spread.get("include_intercept"),
})}
</table>

<h2>Correlations (on returns)</h2>
<p>Measures short-term co-movement. Higher absolute values mean stronger linear/monotonic link.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({f"{k}.value": v.get("value") for k, v in corr.items() if isinstance(v, dict)})}
</table>

<h2>Stationarity</h2>
<p>Stationarity means statistical properties are stable over time. For ADF, p &lt; 0.05 suggests stationarity.
For KPSS, p &gt; 0.05 suggests stationarity.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "spread.adf_p": station.get("spread", {}).get("adf", {}).get("p_value"),
    "spread.kpss_p": station.get("spread", {}).get("kpss", {}).get("p_value"),
    "returns_y.adf_p": station.get("returns_y", {}).get("adf", {}).get("p_value"),
    "returns_x.adf_p": station.get("returns_x", {}).get("adf", {}).get("p_value"),
})}
</table>

<h2>Cointegration (Engle-Granger)</h2>
<p>If residual ADF p-value &lt; 0.05, the pair is likely cointegrated (mean-reverting spread).</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "beta": coint.get("engle_granger", {}).get("beta"),
    "alpha": coint.get("engle_granger", {}).get("alpha"),
    "resid_adf_p": coint.get("engle_granger", {}).get("resid_adf_p"),
})}
</table>

<h2>Mean reversion diagnostics</h2>
<p>Half-life estimates how many bars it takes for the spread to revert by half.
Hurst &lt; 0.5 suggests mean reversion.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "half_life": mean_rev.get("half_life"),
    "hurst": mean_rev.get("hurst"),
})}
</table>

<h2>Normality checks</h2>
<p>Normality tests indicate whether spread/returns behave like a normal distribution. This is mainly diagnostic.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({k: v.get("p_value") for k, v in norm.items() if isinstance(v, dict)})}
</table>

<h2>Granger causality (returns)</h2>
<p>Low p-values at certain lags suggest one series may help predict the other (not necessarily causal in practice).</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({f"lag_{k}": v.get("p_value") for k, v in caus.get("granger", {}).items() if isinstance(v, dict)})}
</table>

<h2>Rolling stability</h2>
<p>Rolling tests show if the relationship is stable over time. Higher share of windows with ADF p &lt; 0.05
indicates more consistent mean reversion.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "windows": rolling.get("windows"),
    "adf_p_ok_pct": rolling.get("adf_p_ok_pct"),
})}
</table>

<h2>Structural breaks (Chow)</h2>
<p>Lower p-values suggest a potential regime change at that time.</p>
<table border="1" cellpadding="6" cellspacing="0">
{table({
    "enabled": breaks.get("enabled"),
    "top_breaks": len(breaks.get("top_p_values", [])) if isinstance(breaks, dict) else 0,
})}
</table>

<h2>Notes</h2>
<ul>
<li>This report is for research. It does not execute trades.</li>
<li>Statistical tests can be unstable on short samples or during regime shifts.</li>
</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
