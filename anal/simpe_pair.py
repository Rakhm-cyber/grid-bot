from anal.pairtool.pipeline import run_analysis
from anal.pairtool.config import Config


def main() -> None:
    cfg = Config(
        symbol1="BTCUSDT",
        symbol2="ETHUSDT",
        timeframe="1m",
        data_source="csv",
        input_path="data/binance",
        output_path="anal/output",
    )
    run_analysis(cfg)


if __name__ == "__main__":
    main()
