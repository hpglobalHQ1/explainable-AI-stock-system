from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base, get_session


class CapitalLimitError(RuntimeError):
    pass


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    buy_price: Mapped[float] = mapped_column(Float, nullable=False)
    buy_date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str] = mapped_column(String, nullable=False, default="")


@dataclass
class PositionInput:
    ticker: str
    quantity: float
    buy_price: float
    buy_date: date | None = None
    notes: str = ""


@dataclass
class PositionData:
    id: int
    ticker: str
    quantity: float
    buy_price: float
    buy_date: date
    notes: str


class PortfolioEngine:
    def __init__(self, monthly_investment_limit: float = 50_000.0) -> None:
        self.monthly_investment_limit = monthly_investment_limit

    def add_position(self, pos: PositionInput) -> int:
        d = pos.buy_date or datetime.utcnow().date()
        spend = pos.quantity * pos.buy_price
        month_spend = 0.0
        with get_session() as s:
            from datetime import date as date_type
            today_first = datetime.utcnow().date().replace(day=1)
            rows = s.query(Position).filter(Position.buy_date >= today_first).all()
            month_spend = sum(r.quantity * r.buy_price for r in rows)
            if month_spend + spend > self.monthly_investment_limit:
                raise CapitalLimitError("Monthly investment limit exceeded")
            row = Position(
                ticker=pos.ticker.upper(),
                quantity=pos.quantity,
                buy_price=pos.buy_price,
                buy_date=d,
                notes=pos.notes,
            )
            s.add(row)
            s.flush()
            return int(row.id)

    def remove_position(self, position_id: int) -> bool:
        with get_session() as s:
            row = s.get(Position, position_id)
            if not row:
                return False
            s.delete(row)
            return True

    def list_positions(self) -> list[PositionData]:
        with get_session() as s:
            rows = s.query(Position).order_by(Position.id).all()
            return [
                PositionData(
                    id=row.id,
                    ticker=row.ticker,
                    quantity=row.quantity,
                    buy_price=row.buy_price,
                    buy_date=row.buy_date,
                    notes=row.notes,
                )
                for row in rows
            ]

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for p in self.list_positions():
            total += p.quantity * current_prices.get(p.ticker, p.buy_price)
        return total

    def get_pnl_report(self, current_prices: dict[str, float]) -> pd.DataFrame:
        rows = []
        for p in self.list_positions():
            cur = current_prices.get(p.ticker, p.buy_price)
            cost = p.quantity * p.buy_price
            val = p.quantity * cur
            rows.append(
                {
                    "id": p.id,
                    "ticker": p.ticker,
                    "quantity": p.quantity,
                    "buy_price": p.buy_price,
                    "current_price": cur,
                    "cost_basis": cost,
                    "market_value": val,
                    "pnl": val - cost,
                }
            )
        return pd.DataFrame(rows)

    def get_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        returns = price_data.pct_change().dropna(how="any")
        return returns.corr()

    def get_concentration_risk(self, current_prices: dict[str, float]) -> dict:
        pnl = self.get_pnl_report(current_prices)
        if pnl.empty:
            return {"triggered": False, "max_weight": 0.0, "ticker": None}
        total = pnl["market_value"].sum()
        pnl["weight"] = pnl["market_value"] / total
        max_row = pnl.loc[pnl["weight"].idxmax()]
        return {
            "triggered": bool(max_row["weight"] > 0.25),
            "max_weight": float(max_row["weight"]),
            "ticker": str(max_row["ticker"]),
        }


if __name__ == "__main__":
    from db.database import init_db

    init_db()
    pe = PortfolioEngine(monthly_investment_limit=100_000)
    pid = pe.add_position(PositionInput(ticker="AAPL", quantity=10, buy_price=180.0))
    print("Added ID:", pid)
    print(pe.get_pnl_report({"AAPL": 195.0}))
