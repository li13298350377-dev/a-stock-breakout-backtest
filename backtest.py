"""单持仓事件驱动回测。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import (
    COMMISSION_RATE,
    INITIAL_CASH,
    LOT_SIZE,
    MA_EXIT_WINDOW,
    MAX_CONSECUTIVE_LOSSES,
    MAX_HOLD_DAYS_WITHOUT_PROFIT,
    MIN_COMMISSION,
    PROFIT_TARGET_PCT,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    STOP_LOSS_PCT,
    TRAILING_ACTIVATE_PCT,
    TRAILING_DRAWDOWN_PCT,
)
from signals import add_buy_signals


@dataclass
class Position:
    """当前持仓。"""
    code: str
    name: str
    shares: int
    buy_price: float
    buy_date: pd.Timestamp
    hold_days: int = 0
    max_close: float = 0.0


def buy_cost(price: float, shares: int) -> float:
    """买入总成本：成交金额 + 佣金。"""
    amount = price * shares
    return amount + max(amount * COMMISSION_RATE, MIN_COMMISSION)


def sell_cash(price: float, shares: int) -> float:
    """卖出到账现金：成交金额 - 佣金 - 印花税。"""
    amount = price * shares
    return amount - max(amount * COMMISSION_RATE, MIN_COMMISSION) - amount * STAMP_TAX_RATE


def run_backtest(stock_data: dict[str, pd.DataFrame], names: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行回测：每天最多持有一只股票，按信号次日开盘交易。"""
    prepared = {code: add_buy_signals(df) for code, df in stock_data.items() if not df.empty}
    all_dates = sorted({d for df in prepared.values() for d in df["日期"].tolist()})
    rows = {code: df.set_index("日期") for code, df in prepared.items()}

    cash = float(INITIAL_CASH)
    position: Position | None = None
    consecutive_losses = 0
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for current_date in all_dates:
        # 先处理持仓的卖出信号，卖出统一在次日开盘价执行。
        if position and current_date in rows[position.code].index:
            row = rows[position.code].loc[current_date]
            position.hold_days += 1
            position.max_close = max(position.max_close, float(row["收盘"]))
            pnl_pct = float(row["收盘"]) / position.buy_price * 100 - 100
            drawdown_from_high = float(row["收盘"]) / position.max_close * 100 - 100

            reason = None
            if pnl_pct <= STOP_LOSS_PCT:
                reason = "亏损达到 8% 止损"
            elif float(row["收盘"]) < float(row[f"ma{MA_EXIT_WINDOW}"]):
                reason = "收盘跌破 10 日均线"
            elif position.hold_days >= MAX_HOLD_DAYS_WITHOUT_PROFIT and pnl_pct <= PROFIT_TARGET_PCT:
                reason = "5 个交易日未盈利超过 8%"
            elif pnl_pct >= TRAILING_ACTIVATE_PCT and drawdown_from_high <= -TRAILING_DRAWDOWN_PCT:
                reason = "盈利超过 20% 后回撤 10%"

            if reason and pd.notna(row.get("next_open")):
                exec_price = float(row["next_open"]) * (1 - SLIPPAGE_RATE)
                proceeds = sell_cash(exec_price, position.shares)
                pnl = proceeds - buy_cost(position.buy_price, position.shares)
                cash += proceeds
                consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                trades.append(
                    {
                        "date": row["日期"] if "日期" in row else current_date,
                        "exec_date": None,
                        "code": position.code,
                        "name": position.name,
                        "side": "SELL",
                        "price": round(exec_price, 3),
                        "shares": position.shares,
                        "reason": reason,
                        "pnl": round(pnl, 2),
                        "cash": round(cash, 2),
                    }
                )
                position = None

        # 空仓且未触发连续亏损暂停时，按当日信号选择成交额最高标的，次日开盘买入。
        if position is None and consecutive_losses < MAX_CONSECUTIVE_LOSSES:
            candidates = []
            for code, df_by_date in rows.items():
                if current_date not in df_by_date.index:
                    continue
                row = df_by_date.loc[current_date]
                if bool(row.get("buy_signal", False)):
                    candidates.append((float(row["成交额"]), code, row))
            if candidates:
                _, code, row = max(candidates, key=lambda x: x[0])
                exec_price = float(row["next_open"]) * (1 + SLIPPAGE_RATE)
                shares = int(cash // (exec_price * LOT_SIZE)) * LOT_SIZE
                if shares >= LOT_SIZE and buy_cost(exec_price, shares) <= cash:
                    cash -= buy_cost(exec_price, shares)
                    position = Position(
                        code=code,
                        name=names.get(code, ""),
                        shares=shares,
                        buy_price=exec_price,
                        buy_date=current_date,
                        max_close=float(row["收盘"]),
                    )
                    trades.append(
                        {
                            "date": current_date,
                            "exec_date": None,
                            "code": code,
                            "name": names.get(code, ""),
                            "side": "BUY",
                            "price": round(exec_price, 3),
                            "shares": shares,
                            "reason": "强势突破，次日开盘买入",
                            "pnl": 0,
                            "cash": round(cash, 2),
                        }
                    )

        market_value = 0.0
        if position and current_date in rows[position.code].index:
            market_value = float(rows[position.code].loc[current_date]["收盘"]) * position.shares
        equity_rows.append({"date": current_date, "cash": round(cash, 2), "market_value": round(market_value, 2), "equity": round(cash + market_value, 2)})

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)
