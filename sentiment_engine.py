from __future__ import annotations

import json
import math
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


@dataclass(frozen=True)
class SentimentRecord:
    title: str
    published_at_utc: datetime
    market_close_utc: datetime
    score: float


@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tok)


class SentimentEngine:
    def __init__(self, cache_dir: str | Path = "cache/sentiment") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = _get_finbert_pipeline()

    @staticmethod
    def _to_market_close_utc(ts_utc: datetime) -> datetime:
        et = ts_utc.astimezone(ZoneInfo("America/New_York"))
        close_et = datetime(et.year, et.month, et.day, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        return close_et.astimezone(timezone.utc)

    @staticmethod
    def _label_to_signed_score(label: str, prob: float) -> float:
        lab = label.lower()
        if "positive" in lab:
            return prob
        if "negative" in lab:
            return -prob
        return 0.0

    def _cache_path(self, ticker: str, as_of: datetime) -> Path:
        return self.cache_dir / f"{ticker.upper()}_{as_of.date().isoformat()}.json"

    def get_headline_sentiment(self, ticker: str, as_of: datetime | None = None) -> dict:
        as_of = as_of or datetime.now(timezone.utc)
        cp = self._cache_path(ticker, as_of)
        if cp.exists():
            return json.loads(cp.read_text())

        news = yf.Ticker(ticker).news or []
        records: list[SentimentRecord] = []
        for item in news:
            title = item.get("title", "")
            pub_ts = item.get("providerPublishTime")
            if not title or not pub_ts:
                continue
            pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            pred = self.pipe(title)[0]
            signed = self._label_to_signed_score(pred["label"], float(pred["score"]))
            records.append(
                SentimentRecord(
                    title=title,
                    published_at_utc=pub_dt,
                    market_close_utc=self._to_market_close_utc(pub_dt),
                    score=signed,
                )
            )

        half_life_days = 3.0
        weighted_sum, total_weight = 0.0, 0.0
        for r in records:
            age_days = (as_of - r.market_close_utc).total_seconds() / 86400
            weight = math.exp(-math.log(2) * max(age_days, 0) / half_life_days)
            weighted_sum += r.score * weight
            total_weight += weight
        aggregate = weighted_sum / total_weight if total_weight else 0.0

        payload = {
            "ticker": ticker.upper(),
            "as_of": as_of.isoformat(),
            "aggregate_score": aggregate,
            "headlines": [
                {
                    "title": r.title,
                    "published_at_utc": r.published_at_utc.isoformat(),
                    "market_close_utc": r.market_close_utc.isoformat(),
                    "score": r.score,
                }
                for r in records
            ],
        }
        cp.write_text(json.dumps(payload, indent=2))
        return payload


if __name__ == "__main__":
    engine = SentimentEngine()
    out = engine.get_headline_sentiment("MSFT")
    print("Aggregate sentiment:", out["aggregate_score"])
    print("Headlines analyzed:", len(out["headlines"]))
