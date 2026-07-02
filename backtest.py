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
    MAX_NEXT_OPEN_GAP,
    MIN_COMMISSION,
    MIN_NEXT_OPEN_GAP,
    PROFIT_TARGET_PCT,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    STOP_LOSS_PCT,
    TRAILING_ACTIVATE_PCT,
    TRAILING_DRAWDOWN_PCT,
)
from signals import add_buy_signals

DIAGNOSTIC_COLUMNS = [
    "code",
    "name",
    "history_rows",
    "first_date",
    "last_date",
    "buy_signal_count",
    "total_signal_events",
    "unexplained_signal_count",
    "executed_buy_count",
    "sell_count",
    "skipped_high_gap_count",
    "skipped_cash_count",
    "skipped_position_count",
    "total_pnl",
    "final_cash_after_stock",
]




@dataclass(frozen=True)
class BacktestRules:
    """可配置的买入过滤与卖出风控阈值。"""
    max_next_open_gap: float = MAX_NEXT_OPEN_GAP
    min_next_open_gap: float | None = MIN_NEXT_OPEN_GAP
    stop_loss_pct: float = STOP_LOSS_PCT
    max_hold_days_without_profit: int = MAX_HOLD_DAYS_WITHOUT_PROFIT
    profit_target_pct: float = PROFIT_TARGET_PCT
    trailing_activate_pct: float = TRAILING_ACTIVATE_PCT
    trailing_drawdown_pct: float = TRAILING_DRAWDOWN_PCT

SIGNAL_EVENT_COLUMNS = [
    "signal_date",
    "next_date",
    "code",
    "name",
    "signal_close",
    "next_open",
    "open_gap_pct",
    "action",
    "reason",
    "cash_before",
    "position_before",
    "shares",
    "price",
]


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


def _build_diagnostics(prepared: dict[str, pd.DataFrame], names: dict[str, str]) -> dict[str, dict]:
    """初始化每只股票的诊断计数。"""
    diagnostics: dict[str, dict] = {}
    for code, df in prepared.items():
        first_date = df["日期"].iloc[0] if not df.empty else None
        last_date = df["日期"].iloc[-1] if not df.empty else None
        skipped_high_gap = df.get("buy_signal_before_gap", pd.Series(dtype=bool)) & ~df.get(
            "buy_signal", pd.Series(dtype=bool)
        )
        diagnostics[code] = {
            "code": code,
            "name": names.get(code, ""),
            "history_rows": len(df),
            "first_date": first_date,
            "last_date": last_date,
            "buy_signal_count": int(
                df.get("buy_signal_before_gap", pd.Series(dtype=bool)).sum()
            ),
            "total_signal_events": int(
                df.get("buy_signal_rule", pd.Series(dtype=bool)).sum()
            ),
            "unexplained_signal_count": 0,
            "executed_buy_count": 0,
            "sell_count": 0,
            "skipped_high_gap_count": int(skipped_high_gap.sum()),
            "skipped_cash_count": 0,
            "skipped_position_count": 0,
            "total_pnl": 0.0,
            "final_cash_after_stock": None,
        }
    return diagnostics


def run_backtest(
    stock_data: dict[str, pd.DataFrame],
    names: dict[str, str],
    pause_after_consecutive_losses: bool = True,
    rules: BacktestRules | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行回测：每天最多持有一只股票，按信号次日开盘交易。

    pause_after_consecutive_losses 为 False 时，仅用于研究诊断，忽略连续亏损暂停买入规则。
    """
    rules = rules or BacktestRules()
    prepared = {
        code: add_buy_signals(
            df,
            max_next_open_gap=rules.max_next_open_gap,
            min_next_open_gap=rules.min_next_open_gap,
        )
        for code, df in stock_data.items()
        if not df.empty
    }
    diagnostics = _build_diagnostics(prepared, names)
    all_dates = sorted({d for df in prepared.values() for d in df["日期"].tolist()})
    rows = {code: df.set_index("日期") for code, df in prepared.items()}

    cash = float(INITIAL_CASH)
    position: Position | None = None
    consecutive_losses = 0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    signal_events: list[dict] = []
    explained_signal_keys: set[tuple[str, pd.Timestamp]] = set()

    def position_label() -> str:
        if position is None:
            return ""
        return f"{position.code}:{position.shares}"

    def add_signal_event(
        signal_date: pd.Timestamp,
        code: str,
        row: pd.Series,
        action: str,
        reason: str,
        cash_before: float,
        position_before: str,
        shares: int = 0,
        price: float | None = None,
    ) -> None:
        signal_events.append(
            {
                "signal_date": signal_date,
                "next_date": row.get("next_date"),
                "code": code,
                "name": names.get(code, ""),
                "signal_close": round(float(row["收盘"]), 3),
                "next_open": round(float(row["next_open"]), 3) if pd.notna(row.get("next_open")) else None,
                "open_gap_pct": round(float(row["next_open_gap"]), 2) if pd.notna(row.get("next_open_gap")) else None,
                "action": action,
                "reason": reason,
                "cash_before": round(cash_before, 2),
                "position_before": position_before,
                "shares": shares,
                "price": round(float(price), 3) if price is not None else None,
            }
        )
        explained_signal_keys.add((code, signal_date))


    for current_date in all_dates:
        for code, df_by_date in rows.items():
            if current_date not in df_by_date.index:
                continue
            row = df_by_date.loc[current_date]
            if bool(row.get("buy_signal_rule", False)) and pd.isna(row.get("next_open")):
                add_signal_event(
                    current_date,
                    code,
                    row,
                    "SKIPPED_LAST_DAY",
                    "信号日没有下一交易日开盘价，无法次日买入",
                    cash,
                    position_label(),
                )
            elif bool(row.get("buy_signal_before_gap", False)) and not bool(row.get("buy_signal", False)):
                add_signal_event(
                    current_date,
                    code,
                    row,
                    "SKIPPED_HIGH_GAP",
                    "次日开盘涨跌幅超过阈值，放弃买入",
                    cash,
                    position_label(),
                )

        if (
            pause_after_consecutive_losses
            and position is None
            and consecutive_losses >= MAX_CONSECUTIVE_LOSSES
        ):
            for code, df_by_date in rows.items():
                if current_date in df_by_date.index and bool(df_by_date.loc[current_date].get("buy_signal", False)):
                    row = df_by_date.loc[current_date]
                    if (code, current_date) not in explained_signal_keys:
                        add_signal_event(
                            current_date,
                            code,
                            row,
                            "SKIPPED_OTHER",
                            "连续亏损次数达到上限，暂停买入",
                            cash,
                            "",
                        )

        # 先处理持仓的卖出信号，卖出统一在次日开盘价执行。
        if position and current_date in rows[position.code].index:
            row = rows[position.code].loc[current_date]
            position.hold_days += 1
            position.max_close = max(position.max_close, float(row["收盘"]))
            pnl_pct = float(row["收盘"]) / position.buy_price * 100 - 100
            drawdown_from_high = float(row["收盘"]) / position.max_close * 100 - 100

            reason = None
            if pnl_pct <= rules.stop_loss_pct:
                reason = f"亏损达到 {abs(rules.stop_loss_pct):g}% 止损"
            elif float(row["收盘"]) < float(row[f"ma{MA_EXIT_WINDOW}"]):
                reason = "收盘跌破 10 日均线"
            elif (
                position.hold_days >= rules.max_hold_days_without_profit
                and pnl_pct <= rules.profit_target_pct
            ):
                reason = (
                    f"{rules.max_hold_days_without_profit} 个交易日"
                    f"未盈利超过 {rules.profit_target_pct:g}%"
                )
            elif (
                pnl_pct >= rules.trailing_activate_pct
                and drawdown_from_high <= -rules.trailing_drawdown_pct
            ):
                reason = (
                    f"盈利超过 {rules.trailing_activate_pct:g}% 后"
                    f"回撤 {rules.trailing_drawdown_pct:g}%"
                )

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
                diagnostics[position.code]["sell_count"] += 1
                diagnostics[position.code]["total_pnl"] += pnl
                diagnostics[position.code]["final_cash_after_stock"] = round(cash, 2)
                position = None

        if position is not None:
            for code, df_by_date in rows.items():
                has_signal = current_date in df_by_date.index and bool(
                    df_by_date.loc[current_date].get("buy_signal", False)
                )
                if has_signal:
                    diagnostics[code]["skipped_position_count"] += 1
                    row = df_by_date.loc[current_date]
                    if (code, current_date) not in explained_signal_keys:
                        add_signal_event(
                            current_date,
                            code,
                            row,
                            "SKIPPED_POSITION",
                            "已有持仓，单持仓规则跳过该买入信号",
                            cash,
                            position_label(),
                        )

        # 空仓且未触发连续亏损暂停时，按当日信号选择成交额最高标的，次日开盘买入。
        can_open_position = (
            position is None
            and (
                not pause_after_consecutive_losses
                or consecutive_losses < MAX_CONSECUTIVE_LOSSES
            )
        )
        if can_open_position:
            candidates = []
            for code, df_by_date in rows.items():
                if current_date not in df_by_date.index:
                    continue
                row = df_by_date.loc[current_date]
                if bool(row.get("buy_signal", False)):
                    candidates.append((float(row["成交额"]), code, row))
            if candidates:
                _, code, row = max(candidates, key=lambda x: x[0])
                selected_code = code
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
                    add_signal_event(
                        current_date,
                        code,
                        row,
                        "EXECUTED_BUY",
                        "强势突破，次日开盘买入",
                        cash + buy_value + buy_commission,
                        "",
                        shares,
                        exec_price,
                    )
                    diagnostics[code]["executed_buy_count"] += 1
                    diagnostics[code]["final_cash_after_stock"] = round(cash, 2)
                else:
                    add_signal_event(
                        current_date,
                        code,
                        row,
                        "SKIPPED_CASH",
                        "现金不足以买入一手或覆盖买入成本",
                        cash,
                        "",
                        shares,
                        exec_price,
                    )
                    diagnostics[code]["skipped_cash_count"] += 1

                for _, other_code, other_row in candidates:
                    if other_code != selected_code and (other_code, current_date) not in explained_signal_keys:
                        add_signal_event(
                            current_date,
                            other_code,
                            other_row,
                            "SKIPPED_OTHER",
                            "同日存在多个买入信号，仅选择成交额最高标的",
                            cash,
                            position_label(),
                        )

        market_value = 0.0
        if position and current_date in rows[position.code].index:
            market_value = float(rows[position.code].loc[current_date]["收盘"]) * position.shares
        equity_rows.append(
            {
                "date": current_date,
                "cash": round(cash, 2),
                "market_value": round(market_value, 2),
                "equity": round(cash + market_value, 2),
            }
        )

    for code, df_by_date in rows.items():
        for signal_date, row in df_by_date[df_by_date.get("buy_signal_rule", False)].iterrows():
            if (code, signal_date) not in explained_signal_keys:
                add_signal_event(
                    signal_date,
                    code,
                    row,
                    "SKIPPED_OTHER",
                    "买入信号未被其他分支处理",
                    cash,
                    position_label(),
                )

    for item in diagnostics.values():
        explained_count = sum(1 for event in signal_events if event["code"] == item["code"])
        item["unexplained_signal_count"] = max(
            0, int(item["total_signal_events"]) - explained_count
        )

    diagnostics_df = pd.DataFrame(diagnostics.values(), columns=DIAGNOSTIC_COLUMNS)
    if not diagnostics_df.empty:
        diagnostics_df["total_pnl"] = diagnostics_df["total_pnl"].round(2)
    signal_events_df = pd.DataFrame(signal_events, columns=SIGNAL_EVENT_COLUMNS)
    return (
        pd.DataFrame(trades, columns=TRADE_COLUMNS),
        pd.DataFrame(equity_rows),
        diagnostics_df,
        signal_events_df,
    )
