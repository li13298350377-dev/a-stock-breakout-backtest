"""绩效指标计算。"""
from __future__ import annotations

import pandas as pd


def summarize_performance(equity: pd.DataFrame, trades: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """输出总收益、最大回撤、胜率等绩效汇总。"""
    if equity.empty:
        return pd.DataFrame([{"指标": "无数据", "数值": 0}])

    curve = equity["equity"].astype(float)
    total_return = curve.iloc[-1] / initial_cash - 1
    running_max = curve.cummax()
    max_drawdown = (curve / running_max - 1).min()

    sell_trades = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    wins = int((sell_trades.get("pnl", pd.Series(dtype=float)) > 0).sum())
    total_sells = len(sell_trades)
    win_rate = wins / total_sells if total_sells else 0

    return pd.DataFrame(
        [
            {"指标": "初始资金", "数值": initial_cash},
            {"指标": "期末权益", "数值": round(curve.iloc[-1], 2)},
            {"指标": "总收益率", "数值": round(total_return, 4)},
            {"指标": "最大回撤", "数值": round(max_drawdown, 4)},
            {"指标": "交易次数", "数值": total_sells},
            {"指标": "胜率", "数值": round(win_rate, 4)},
        ]
    )
