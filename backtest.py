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

TRADE_COLUMNS = [
    "signal_date",
    "exec_date",
    "code",
    "name",
    "side",
    "buy_price",
    "sell_price",
    "shares",
    "buy_value",
    "sell_value",
    "buy_commission",
    "sell_commission",
    "stamp_tax",
    "pnl",
    "pnl_pct",
    "cash_after_trade",
    "reason",
]


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


def trade_value(price: float, shares: int) -> float:
    """成交金额：成交价格 * 股数。"""
    return price * shares


def commission(value: float) -> float:
    """交易佣金：按成交金额比例计算，受最低佣金约束。"""
    return max(value * COMMISSION_RATE, MIN_COMMISSION)


def buy_cost(price: float, shares: int) -> float:
    """买入总成本：成交金额 + 佣金。"""
    amount = trade_value(price, shares)
    return amount + commission(amount)


def sell_cash(price: float, shares: int) -> float:
    """卖出到账现金：成交金额 - 佣金 - 印花税。"""
    amount = trade_value(price, shares)
    return amount - commission(amount) - amount * STAMP_TAX_RATE


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
                sell_value = trade_value(exec_price, position.shares)
                sell_commission = commission(sell_value)
                stamp_tax = sell_value * STAMP_TAX_RATE
                proceeds = sell_value - sell_commission - stamp_tax
                buy_total_cost = buy_cost(position.buy_price, position.shares)
                pnl = proceeds - buy_total_cost
                pnl_pct = pnl / buy_total_cost * 100
                cash += proceeds
                consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                trades.append(
                    {
                        "signal_date": current_date,
                        "exec_date": row["next_date"],
                        "code": position.code,
                        "name": position.name,
                        "side": "SELL",
                        "sell_price": round(exec_price, 3),
                        "shares": position.shares,
                        "sell_value": round(sell_value, 2),
                        "sell_commission": round(sell_commission, 2),
                        "stamp_tax": round(stamp_tax, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "cash_after_trade": round(cash, 2),
                        "reason": reason,
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
                    buy_value = trade_value(exec_price, shares)
                    buy_commission = commission(buy_value)
                    cash -= buy_value + buy_commission
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
                            "signal_date": current_date,
                            "exec_date": row["next_date"],
                            "code": code,
                            "name": names.get(code, ""),
                            "side": "BUY",
                            "buy_price": round(exec_price, 3),
                            "shares": shares,
                            "buy_value": round(buy_value, 2),
                            "buy_commission": round(buy_commission, 2),
                            "cash_after_trade": round(cash, 2),
                            "reason": "强势突破，次日开盘买入",
                        }
                    )

        market_value = 0.0
        if position and current_date in rows[position.code].index:
            market_value = float(rows[position.code].loc[current_date]["收盘"]) * position.shares
        equity_rows.append({"date": current_date, "cash": round(cash, 2), "market_value": round(market_value, 2), "equity": round(cash + market_value, 2)})

    return pd.DataFrame(trades, columns=TRADE_COLUMNS), pd.DataFrame(equity_rows)
