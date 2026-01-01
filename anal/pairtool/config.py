from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _deep_get(dct: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur: Any = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class Config:
    symbol1: str = "BTCUSDT"
    symbol2: str = "ETHUSDT"
    timeframe: str = "1m"
    data_source: str = "csv"
    input_path: str = "data/binance"
    output_path: str = "anal/output"

    price_field: str = "close"
    use_log_prices: bool = True
    winsorize_enabled: bool = True
    winsorize_limits: Tuple[float, float] = (0.001, 0.999)
    outliers_method: str = "zscore"
    outliers_threshold: float = 6.0
    missing_strategy: str = "drop"
    missing_ffill_limit: Optional[int] = None

    correlations: List[str] = field(default_factory=lambda: ["pearson", "spearman", "kendall"])
    stationarity: List[str] = field(default_factory=lambda: ["adf", "kpss", "pp"])
    cointegration_engle_granger: bool = True
    cointegration_johansen: bool = False
    normality: List[str] = field(default_factory=lambda: ["jarque_bera", "anderson_darling", "ks"])
    granger_enabled: bool = True
    granger_max_lag: int = 10

    hedge_method: str = "ols"
    include_intercept: bool = True

    rolling_window: int = 720
    rolling_step: int = 1
    rolling_min_periods: int = 500
    rolling_corr_threshold: Optional[float] = None

    chow_enabled: bool = True
    chow_grid_points: int = 20
    bai_perron_enabled: bool = False

    output_formats: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    plots: bool = True
    report_title: str = "Pair Analytics Report"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        return cls(
            symbol1=_deep_get(raw, ["symbols", "symbol1"], "BTCUSDT"),
            symbol2=_deep_get(raw, ["symbols", "symbol2"], "ETHUSDT"),
            timeframe=raw.get("timeframe", "1m"),
            data_source=raw.get("data_source", "csv"),
            input_path=_deep_get(raw, ["paths", "input"], "data/binance"),
            output_path=_deep_get(raw, ["paths", "output"], "anal/output"),
            price_field=_deep_get(raw, ["preprocessing", "price_field"], "close"),
            use_log_prices=_deep_get(raw, ["preprocessing", "use_log_prices"], True),
            winsorize_enabled=_deep_get(raw, ["preprocessing", "winsorize", "enabled"], True),
            winsorize_limits=tuple(_deep_get(raw, ["preprocessing", "winsorize", "limits"], [0.001, 0.999])),
            outliers_method=_deep_get(raw, ["preprocessing", "outliers", "method"], "zscore"),
            outliers_threshold=float(_deep_get(raw, ["preprocessing", "outliers", "threshold"], 6.0)),
            missing_strategy=_deep_get(raw, ["preprocessing", "missing", "strategy"], "drop"),
            missing_ffill_limit=_deep_get(raw, ["preprocessing", "missing", "ffill_limit"], None),
            correlations=_deep_get(raw, ["tests", "correlations"], ["pearson", "spearman", "kendall"]),
            stationarity=_deep_get(raw, ["tests", "stationarity"], ["adf", "kpss", "pp"]),
            cointegration_engle_granger=_deep_get(raw, ["tests", "cointegration", "engle_granger"], True),
            cointegration_johansen=_deep_get(raw, ["tests", "cointegration", "johansen"], False),
            normality=_deep_get(raw, ["tests", "normality"], ["jarque_bera", "anderson_darling", "ks"]),
            granger_enabled=_deep_get(raw, ["tests", "causality", "granger", "enabled"], True),
            granger_max_lag=int(_deep_get(raw, ["tests", "causality", "granger", "max_lag"], 10)),
            hedge_method=_deep_get(raw, ["spread", "hedge_method"], "ols"),
            include_intercept=_deep_get(raw, ["spread", "include_intercept"], True),
            rolling_window=int(_deep_get(raw, ["rolling", "window"], 720)),
            rolling_step=int(_deep_get(raw, ["rolling", "step"], 1)),
            rolling_min_periods=int(_deep_get(raw, ["rolling", "min_periods"], 500)),
            rolling_corr_threshold=_deep_get(raw, ["rolling", "corr_threshold"], None),
            chow_enabled=_deep_get(raw, ["breaks", "chow", "enabled"], True),
            chow_grid_points=int(_deep_get(raw, ["breaks", "chow", "grid_points"], 20)),
            bai_perron_enabled=_deep_get(raw, ["breaks", "bai_perron"], False),
            output_formats=_deep_get(raw, ["reporting", "output_formats"], ["json", "csv", "html"]),
            plots=_deep_get(raw, ["reporting", "plots"], True),
            report_title=_deep_get(raw, ["reporting", "title"], "Pair Analytics Report"),
        )
