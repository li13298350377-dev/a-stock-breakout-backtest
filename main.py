"""一键运行 A 股小盘强势突破策略 v1 回测。"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from backtest import run_backtest
from config import FALLBACK_SYMBOLS, INITIAL_CASH, RESULTS_DIR, START_DATE, END_DATE, TOP_N_BY_AMOUNT
from data_loader import build_candidate_universe, ensure_dirs, load_realtime_quotes, load_stock_history
from metrics import summarize_performance
from signals import passes_history_filters, prepare_history


def main() -> None:
    """完整流程：拉行情、筛股票池、拉历史、回测、输出结果。"""
    ensure_dirs()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("[INFO] 拉取/读取 A 股实时行情...")
    try:
        quotes = load_realtime_quotes()
        universe = build_candidate_universe(quotes, TOP_N_BY_AMOUNT)
    except Exception as exc:
        print(f"[WARN] 实时行情接口失败：{exc}")
        universe = pd.DataFrame()

    if universe.empty:
        print("[WARN] 实时股票池失败，使用固定测试股票列表继续回测。")
        universe = pd.DataFrame({"代码": FALLBACK_SYMBOLS, "名称": FALLBACK_SYMBOLS})

    stock_data: dict[str, pd.DataFrame] = {}
    names: dict[str, str] = {}
    end_date = END_DATE or datetime.now().strftime("%Y%m%d")

    print(f"[INFO] 默认测试成交额最高的前 {TOP_N_BY_AMOUNT} 只候选股票...")
    for _, item in universe.iterrows():
        code = str(item["代码"]).zfill(6)
        name = str(item.get("名称", ""))
        hist = load_stock_history(code, START_DATE, end_date)
        hist = prepare_history(hist)
        if not passes_history_filters(hist):
            print(f"[INFO] {code} {name} 未通过历史数据过滤，跳过。")
            continue
        stock_data[code] = hist
        names[code] = name
        print(f"[INFO] 加入回测：{code} {name}")

    if not stock_data:
        print("[WARN] 没有可回测股票，程序结束。")
        return

    print("[INFO] 开始回测...")
    trades, equity = run_backtest(stock_data, names)
    summary = summarize_performance(equity, trades, INITIAL_CASH)

    trades_path = RESULTS_DIR / "trades.csv"
    equity_path = RESULTS_DIR / "daily_equity.csv"
    summary_path = RESULTS_DIR / "performance_summary.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 交易记录：{trades_path}")
    print(f"[INFO] 每日权益：{equity_path}")
    print(f"[INFO] 绩效汇总：{summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
