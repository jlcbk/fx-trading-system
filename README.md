# FX Portfolio System

一个面向常见外汇货币对的本地研究、组合回测、策略筛选和模拟交易系统。系统默认交易
`EURUSD / GBPUSD / USDJPY / USDCHF / AUDUSD / NZDUSD / USDCAD`，强调低目标/止损比、
周内持仓和跨货币组合风控。

> 这是研究与模拟交易软件，不承诺盈利，也不构成投资建议。默认配置不连接真实账户；
> OANDA 适配器只允许 `fxPractice` 域名，代码层禁止生产域名。

## 已实现

- 六类候选策略：regime-aware 均值回归、趋势回踩、伦敦时段突破、假突破反转、
  全货币图强弱反转、滚动 Engle–Granger 协整价差。
- 60+ 个价格、波动、结构、日历、横截面和多货币图因子；三重障碍标签、训练内概率校准、
  purged walk-forward、成对 block bootstrap、FDR 校正、动态成本期望 R 和一次性 holdout。
- 信号只用已收盘 K 线，最早在下一根 K 线开盘执行，避免 look-ahead。
- 每个策略目标/止损比低于 0.85；全局硬上限 0.85；最长持仓硬上限 168 小时。
- 共享币种敞口：同时持有 EURUSD 与 GBPUSD 时，美元风险会合并，而不是当作独立资产。
- 组合风险：单笔风险、总风险、总杠杆、单币种敞口、相关簇、单日亏损和回撤熔断。
- 成本模型：逐品种点差、滑点、双边佣金、可配置 long/short swap，周三三倍 swap。
- 保守回测：同一根 K 线同时触发止损与止盈时按止损成交。
- 策略横向筛选、滚动 walk-forward 样本外选择、可复现 manifest 和数据 SHA-256。
- 公共 Yahoo 行情适配器、CSV 数据适配器、相关多货币合成数据适配器。
- 无副作用的模拟订单计划，以及必须显式确认的 OANDA `fxPractice` 下单适配器。

## 快速开始

需要 Python 3.11+ 和 [uv](https://github.com/astral-sh/uv)：

```bash
uv sync --all-extras
uv run fxtrade validate-config -c configs/demo.yaml
uv run fxtrade backtest -c configs/demo.yaml
uv run fxtrade screen -c configs/demo.yaml
uv run fxtrade walk-forward -c configs/demo.yaml --train-bars 1200 --test-bars 400
uv run fxtrade factor-download -c configs/factors_daily.yaml
uv run fxtrade factor-mine -c configs/factors_daily.yaml
# 只在旧 holdout 之前运行第二轮严格 FDR 开发研究
uv run fxtrade factor-mine -c configs/factors_daily_round2_dev.yaml
```

示例回测使用合成数据，只验证系统行为，不能验证策略收益。产物写入 `outputs/`：

```text
metrics.json           组合和交易指标
trades.csv             每笔交易、成本、R multiple、持仓时间、退出原因
equity.csv             权益、浮盈亏、杠杆和回撤时间序列
signals.csv            实际进入引擎的收盘信号
rejected_signals.json  被哪条风控规则拒绝
run_manifest.json      完整配置、代码版本、数据范围和数据哈希
report.md              人类可读报告
```

## 使用公共历史行情

Yahoo 只适合研究，不是 broker 可成交 bid/ask：

```bash
# 默认拉取 1h 后重采样为 4h；Yahoo 通常只提供最近约 730 天的小时数据
uv run fxtrade download -c configs/default.yaml
uv run fxtrade backtest -c configs/default.yaml
uv run fxtrade screen -c configs/default.yaml
```

生产前应换成目标 broker 的历史 bid/ask，并把真实 spread 与 swap 填入配置。CSV 格式：

```csv
timestamp,open,high,low,close,volume
2025-01-02T00:00:00Z,1.1030,1.1042,1.1022,1.1038,1000
```

每个文件以货币对命名，例如 `data/EURUSD.csv`。

## 模拟交易边界

先生成无副作用计划：

```bash
uv run fxtrade paper-plan -c configs/default.yaml
```

策略必须先在配置中显式设为 `paper_enabled: true` 才会进入计划。当前实证筛选没有候选通过，
所以默认计划为空；仅为验证执行软件时可用 `--include-unapproved`，输出仍不会自动提交。

只有 OANDA 模拟账户可提交，且必须同时具备环境变量和显式确认：

```bash
export OANDA_PRACTICE_ACCOUNT_ID='...'
export OANDA_PRACTICE_TOKEN='...'
uv run fxtrade oanda-practice-submit --plan outputs/paper_plan.json --confirm-practice
```

适配器拒绝 `api-fxtrade.oanda.com`。订单计划与提交刻意拆开，便于人工复核。

## 文档

- [架构与数据流](docs/ARCHITECTURE_ZH.md)
- [策略筛选目录](docs/STRATEGY_CATALOG_ZH.md)
- [开源项目选型](docs/OPEN_SOURCE_SELECTION_ZH.md)
- [验证和上线门槛](docs/VALIDATION_AND_LAUNCH_ZH.md)
- [2026-07-15 实证筛选记录](docs/EMPIRICAL_SCREENING_2026-07-15_ZH.md)
- [多因子挖掘系统与最终实证](docs/MULTIFACTOR_RESEARCH_ZH.md)
- [第二轮成对因子挖掘记录](docs/FACTOR_ROUND2_2026-07-15_ZH.md)
