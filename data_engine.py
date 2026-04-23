from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataEngineConfig:
    cache_dir: Path = Path("cache")
    ttl_seconds: int = 3600
    period: str = "2y"
    interval: str = "1d"
    max_retries: int = 3


class DataEngine:
    def __init__(self, config: DataEngineConfig | None = None) -> None:
        self.config = config or DataEngineConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.config.cache_dir / f"{ticker.upper()}_{self.config.period}_{self.config.interval}.parquet"

    def _meta_path(self, ticker: str) -> Path:
        return self.config.cache_dir / f"{ticker.upper()}_{self.config.period}_{self.config.interval}.meta.json"

    def _is_cache_valid(self, ticker: str) -> bool:
        cp = self._cache_path(ticker)
        mp = self._meta_path(ticker)
        if not cp.exists() or not mp.exists():
            return False
        meta = json.loads(mp.read_text())
        created = datetime.fromisoformat(meta["created_at"])
        return (datetime.now(timezone.utc) - created).total_seconds() <= self.config.ttl_seconds

    def _load_cache(self, ticker: str) -> pd.DataFrame:
        df = pd.read_parquet(self._cache_path(ticker))
        df.index = pd.to_datetime(df.index, utc=True)
        now = datetime.now(timezone.utc)
        return df.loc[df.index <= now]

    def _save_cache(self, ticker: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._cache_path(ticker))
        self._meta_path(ticker).write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat()}))

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        if self._is_cache_valid(ticker):
            return self._load_cache(ticker)

        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw = yf.download(
                    ticker,
                    period=self.config.period,
                    interval=self.config.interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if raw.empty:
                    raise ValueError(f"No data returned for {ticker}")
                raw = raw.rename(columns=str.lower)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0).str.lower()
                raw.index = pd.to_datetime(raw.index, utc=True)
                now = datetime.now(timezone.utc)
                clean = raw.loc[raw.index <= now, ["open", "high", "low", "close", "volume"]].copy()
                self._save_cache(ticker, clean)
                return clean
            except Exception as exc:  # graceful retry
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"Failed to fetch {ticker} after {self.config.max_retries} attempts") from last_exc


if __name__ == "__main__":
    engine = DataEngine()
    df = engine.fetch_ohlcv("AAPL")
    print(df.tail())
    print(f"Rows fetched: {len(df)}; date range: {df.index.min()} -> {df.index.max()}")
