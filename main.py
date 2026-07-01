"""一键运行 A 股小盘强势突破策略 v1 回测。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import run_backtest
from config import (
    FALLBACK_SYMBOLS,
    INITIAL_CASH,
    RESULTS_DIR,
    RESULTS_NO_PAUSE_DIR,
    START_DATE,
    END_DATE,
    TOP_N_BY_AMOUNT,
)
from data_loader import build_candidate_universe, ensure_dirs, load_realtime_quotes, load_stock_history
from metrics import summarize_performance
from signals import passes_history_filters, prepare_history


def save_backtest_outputs(
    output_dir: Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    signal_events: pd.DataFrame,
) -> dict[str, Path]:
    """保存一组回测输出文件。"""
    output_dir.mkdir(exist_ok=True)
    paths = {
        "trades": output_dir / "trades.csv",
        "equity": output_dir / "daily_equity.csv",
        "summary": output_dir / "performance_summary.csv",
        "diagnostics": output_dir / "diagnostics.csv",
        "signal_events": output_dir / "signal_events.csv",
    }
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    equity.to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    diagnostics.to_csv(paths["diagnostics"], index=False, encoding="utf-8-sig")
    signal_events.to_csv(paths["signal_events"], index=False, encoding="utf-8-sig")
    return paths


def print_summary_compare(label: str, summary: pd.DataFrame) -> None:
    """打印核心绩效指标。"""
    values = summary.set_index("指标")["数值"].to_dict() if not summary.empty else {}
    print(
        f"[INFO] {label}："
        f"期末权益={values.get('期末权益', 0)}，"
        f"总收益率={values.get('总收益率', 0)}，"
        f"最大回撤={values.get('最大回撤', 0)}，"
        f"交易次数={values.get('交易次数', 0)}，"
        f"胜率={values.get('胜率', 0)}"
    )


def main() -> None:
    """完整流程：拉行情、筛股票池、拉历史、回测、输出结果。"""
    ensure_dirs()
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_NO_PAUSE_DIR.mkdir(exist_ok=True)

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

    print("[INFO] 开始实盘风控模式回测...")
    trades, equity, diagnostics, signal_events = run_backtest(stock_data, names)
    summary = summarize_performance(equity, trades, INITIAL_CASH)
    paths = save_backtest_outputs(RESULTS_DIR, trades, equity, summary, diagnostics, signal_events)

    print("[INFO] 开始 no-pause 研究诊断模式回测...")
    no_pause_trades, no_pause_equity, no_pause_diagnostics, no_pause_signal_events = run_backtest(
        stock_data,
        names,
        pause_after_consecutive_losses=False,
    )
    no_pause_summary = summarize_performance(no_pause_equity, no_pause_trades, INITIAL_CASH)
    no_pause_paths = save_backtest_outputs(
        RESULTS_NO_PAUSE_DIR,
        no_pause_trades,
        no_pause_equity,
        no_pause_summary,
        no_pause_diagnostics,
        no_pause_signal_events,
    )

    total_buy_signals = int(diagnostics["buy_signal_count"].sum()) if not diagnostics.empty else 0
    executed_buys = int(diagnostics["executed_buy_count"].sum()) if not diagnostics.empty else 0
    skipped_buys = 0
    completed_sells = int(diagnostics["sell_count"].sum()) if not diagnostics.empty else 0
    total_signal_events = len(signal_events)
    action_counts = signal_events["action"].value_counts().to_dict() if not signal_events.empty else {}
    if not diagnostics.empty:
        skipped_buys = int(
            diagnostics[
                ["skipped_high_gap_count", "skipped_cash_count", "skipped_position_count"]
            ].sum().sum()
        )

    print(f"[INFO] 成功读取历史数据的股票数量：{len(stock_data)}")
    print(f"[INFO] 总买入信号数量：{total_buy_signals}")
    print(f"[INFO] 总信号明细数量：{total_signal_events}")
    print(f"[INFO] 实际买入次数：{executed_buys}")
    print(f"[INFO] 放弃买入次数：{skipped_buys}")
    print(f"[INFO] 完成卖出次数：{completed_sells}")
    print(f"[INFO] 实盘风控诊断输出：{paths['diagnostics']}")
    print(f"[INFO] 实盘风控信号明细：{paths['signal_events']}")
    print(f"[INFO] 实盘风控信号明细 action 统计：{action_counts}")
    print(f"[INFO] 实盘风控交易记录：{paths['trades']}")
    print(f"[INFO] 实盘风控每日权益：{paths['equity']}")
    print(f"[INFO] 实盘风控绩效汇总：{paths['summary']}")
    print(f"[INFO] no-pause 诊断交易记录：{no_pause_paths['trades']}")
    print(f"[INFO] no-pause 诊断每日权益：{no_pause_paths['equity']}")
    print(f"[INFO] no-pause 诊断绩效汇总：{no_pause_paths['summary']}")
    print_summary_compare("实盘风控模式", summary)
    print_summary_compare("no-pause 诊断模式", no_pause_summary)
    print(summary)


if __name__ == "__main__":
    main()
