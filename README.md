# A 股小盘强势突破策略 v1 回测项目

这是一个普通 Python 项目，不使用 Jupyter Notebook。项目仅用于历史回测和模拟分析，不连接任何券商，也不会实盘下单。

## 功能

- 使用 AKShare 拉取 A 股实时行情和个股历史日线。
- 所有数据缓存到 `data_cache/`，每只股票历史数据单独保存为 CSV。
- 拉取数据失败会自动重试，最终失败后跳过，不让程序崩溃。
- 实盘风控模式回测结果输出到 `results/`：
  - `trades.csv`：交易记录；
  - `daily_equity.csv`：每日权益；
  - `performance_summary.csv`：绩效汇总；
  - `diagnostics.csv`：按股票汇总的诊断计数；
  - `signal_events.csv`：逐信号处理明细。
- 研究诊断用 no-pause 模式回测结果输出到 `results_no_pause/`，文件名与 `results/` 一致。该模式只忽略“连续亏 3 次暂停新开仓”规则，其他买入、卖出、费用、滑点和单持仓规则不变。
- v1.1 风控优化实验版结果输出到 `results_v1_1/`，文件名与 `results/` 一致。该版本仅用于和 v1 对比研究，不代表可实盘。
- v1.1 no-pause 研究诊断模式结果输出到 `results_v1_1_no_pause/`，文件名与 `results/` 一致。该模式使用 v1.1 全部买卖规则，但忽略“连续亏 3 次暂停新开仓”规则，仅用于研究 v1.1 规则本身的长期表现，不能用于实盘。

### `trades.csv` 字段说明

`trades.csv` 保留逐笔交易输出，并将日期、价格、费用和收益字段拆分得更清楚：

- `signal_date`：触发买入或卖出信号的日期。
- `exec_date`：实际成交日期，即信号日后的下一个交易日开盘。
- `code` / `name`：股票代码和名称。
- `side`：交易方向，`BUY` 表示买入，`SELL` 表示卖出。
- `buy_price` / `sell_price`：买入或卖出的实际成交价格，已包含滑点。
- `shares`：成交股数。
- `buy_value` / `sell_value`：买入或卖出的成交金额，不含佣金和税费。
- `buy_commission` / `sell_commission`：买入或卖出佣金。
- `stamp_tax`：卖出印花税，仅卖出记录有值。
- `pnl`：单笔卖出后的净收益，已扣除买入佣金、卖出佣金和印花税。
- `pnl_pct`：单笔净收益率百分比，按净收益除以买入总成本计算。
- `cash_after_trade`：该笔交易完成后的剩余现金。
- `reason`：触发该笔交易的原因。

默认只测试成交额最高的前 20 只股票，确认跑通后可在 `config.py` 调整 `TOP_N_BY_AMOUNT`。

### 输出目录说明

- `results/`：实盘风控模式结果，保留连续亏 3 次暂停新开仓规则。
- `results_no_pause/`：研究诊断模式结果，忽略连续亏 3 次暂停新开仓规则，用于观察暂停后信号质量。
- `results_v1_1/`：v1.1 风控优化实验版结果，用于和 v1 实盘风控模式、v1 no-pause 诊断模式对比。
- `results_v1_1_no_pause/`：v1.1 no-pause 研究诊断模式结果，使用 v1.1 全部买卖规则，仅忽略连续亏 3 次暂停新开仓规则，用于研究 v1.1 规则本身的长期表现。
- `results_no_pause/`、`results_v1_1/` 和 `results_v1_1_no_pause/` 不能用于实盘，只能用于研究诊断和信号质量观察。


## 历史月度动态股票池原型

新增 `monthly_universe.py` 作为独立研究入口，用于生成历史月度动态股票池原型。当前第一阶段只生成 2023 年 1 月股票池，自动识别 2023 年 1 月第一个实际交易日作为 `screen_date`，并将下一个实际交易日作为 `effective_date`。筛选与 Popularity Score v1 只使用 `screen_date` 收盘及以前的数据，不运行任何交易策略，不做 A1 回测。

月度股票池已改为“免费批量 daily + 预筛后 BaoStock 补充”的数据架构：`market_snapshot_provider.py` 对普通历史交易日只调用 Tushare `pro.daily(trade_date=...)`，按交易日获取全市场历史日线并缓存到 `data_cache/market_daily/YYYYMMDD.csv`。程序先用 daily 数据完成主板代码、`screen_date` 收盘价、上市历史长度和 20 日平均成交额预筛，记录 `prefilter_candidate_count`；随后仅对预筛候选股调用 BaoStock 补充 `screen_date` 当日 `isST` 和当时最新已公开的 `totalShare`，缓存到 `data_cache/baostock_screen_enrichment/YYYYMMDD.csv`，支持断点续跑，已成功缓存的 code 不重复请求。

历史总市值使用 `screen_date close × latest published totalShare` 估算，并输出 `share_pub_date`、`share_stat_date` 与 `market_cap_method=CLOSE_X_LATEST_PUBLISHED_QUARTER_TOTAL_SHARE`。该方法使用的是 `screen_date` 当时最近已公开季度的总股本，不是绝对精确的逐日股本快照；因此不得将其描述为逐日总股本，也不得用未来披露数据或当前总股本替代。历史 ST 判断只使用 BaoStock `screen_date` 的 `isST`，`name` 仅用于展示，名称缺失时使用 code fallback 且不会据此判定为非 ST。该入口不再在完整股票池流程中调用 `daily_basic` 或 `stk_premarket`，也不回退到 AKShare 逐股票历史行情。

默认批量数据源为 Tushare Pro，需要环境变量 `TUSHARE_TOKEN`：

```bash
export TUSHARE_TOKEN=你的TushareToken
```

请勿把 token 写入代码或日志。首次运行完整构建前建议先执行快速探测：

```bash
python monthly_universe.py --probe
```

探测成功后再运行完整构建：

```bash
python monthly_universe.py
```

输出目录为 `monthly_universe_results/2023_01/`，包含 `base_universe.csv`、`popularity_ranking.csv`、`monthly_pool.csv` 和 `data_diagnostics.csv`。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键运行

```bash
python main.py
```

## 策略规则摘要

### 股票池

- 总市值 20 亿到 100 亿；
- 股价 5 元到 40 元；
- 近 20 日平均成交额大于 5000 万；
- 剔除 ST、\*ST、退市风险股；
- 剔除上市不足 120 天的新股；
- 剔除停牌或数据缺失股票；
- 只保留沪深主板：`600`、`601`、`603`、`605`、`000`、`001`、`002`、`003` 开头。

### 买入

- 收盘价创近 60 日新高；
- 当日成交额大于过去 20 日平均成交额的 2 倍；
- 当日涨幅在 5% 到 10% 之间；
- 过去 5 日涨幅大于 10%；
- 过去 20 日涨幅小于 60%；
- 收盘价 > 5 日均线 > 10 日均线 > 20 日均线；
- 第二天开盘买入；
- 如果第二天开盘高开超过 5%，放弃买入。

### 卖出

- 买入后亏损达到 8%，第二天开盘卖出；
- 收盘价跌破 10 日均线，第二天开盘卖出；
- 买入后 5 个交易日还没有盈利超过 8%，第二天开盘卖出；
- 盈利超过 20% 后，如果从最高收盘价回撤 10%，第二天开盘卖出。

### v1.1 风控优化实验版

`results_v1_1/` 只调整以下风控阈值，不改股票池、数据拉取、费用、滑点、印花税和单持仓规则：

- 第二天开盘高开超过 3% 放弃买入，低开超过 3% 也放弃买入；
- 买入后亏损达到 5%，第二天开盘卖出；
- 买入后 3 个交易日还没有盈利超过 5%，第二天开盘卖出；
- 盈利超过 10% 后，如果从最高收盘价回撤 5%，第二天开盘卖出。

该版本是风控优化实验版，仅用于历史回测对比，不代表可实盘。`results_v1_1_no_pause/` 仅用于研究 v1.1 规则本身在忽略连续亏损暂停后的长期表现，不能用于实盘。

### 资金

- 初始资金 4000 元；
- 每次只持有一只股票；
- 买入按 100 股整数倍；
- 不补仓、不融资；
- 佣金万 3，每笔最低佣金 5 元；
- 卖出印花税 0.05%；
- 滑点 0.1%；
- 连续亏 3 次暂停新开仓；
- 亏到 3000 以下应暂停实盘，本项目仍只做模拟回测。

## 目录结构

```text
.
├── backtest.py
├── config.py
├── data_loader.py
├── main.py
├── metrics.py
├── README.md
├── requirements.txt
└── signals.py
```
