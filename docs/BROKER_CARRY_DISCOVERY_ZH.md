# Broker 报价、Carry 与受约束因子挖掘

## 已完成的开发范围

这一阶段把系统从“midpoint 技术因子验证器”扩展为以下完整研究链路：

```text
OANDA fxPractice bid/ask 或双边 CSV
        + 历史 spread / swap
        + point-in-time 利率和 forward points
                         │
                         ▼
          Carry 因子 + 原有价格因子
                         │
                         ▼
         有预算、可追踪谱系的因子 DSL
                         │
                         ▼
  paired bootstrap + FDR + purged walk-forward
                         │
                         ▼
       1.0x / 1.5x / 2.0x 成本压力测试
                         │
                         ▼
       rejected / 新 holdout / paper 候选
```

软件链路已经可以运行，但仓库不包含用户的 OANDA token、商业 forward 数据或历史融资
记录。合成数据端到端运行仅用于验证软件，不是盈利证据。

## Bid/ask 数据

OANDA 下载器只允许 `https://api-fxpractice.oanda.com`，请求 `price=BA`，分页拉取完整 candle，
拒绝真实交易域名。使用方式：

```bash
export OANDA_PRACTICE_TOKEN='...'
uv run fxtrade factor-download \
  -c configs/oanda_bid_ask_download.yaml \
  -o data/oanda_bid_ask
```

CSV 每行至少包含：

```text
timestamp
bid_open,bid_high,bid_low,bid_close
ask_open,ask_high,ask_low,ask_close
volume
```

系统从双边价格派生 mid OHLC，并检查：双边各自 OHLC 合法、ask 不低于 bid、时间 UTC、
排序、重复和不完整 candle。回测中：

- 多头按 ask 入场、bid 平仓；空头按 bid 入场、ask 平仓。
- 多头止盈止损检查 bid high/low；空头检查 ask high/low。
- spread 压力测试围绕 mid 放大实际双边报价，不再用固定 spread 替代。
- `spread_atr` 与 `spread_z_20` 把当时可见的执行拥挤状态作为非方向因子。

## 历史 swap

OANDA candle API 不返回历史融资费率，因此每个品种需要独立的 point-in-time 文件：

```csv
available_time,swap_long_pips,swap_short_pips
2025-01-02T22:00:00Z,-0.72,0.31
```

文件放在 `data/oanda_swaps/EURUSD.csv` 等路径。系统只向后 as-of join，不会把未来费率
回填到过去；超过配置的 staleness 后视为缺失。示例见
`examples/swaps/EURUSD.example.csv`。

## Point-in-time 利率与远期数据

利率文件 `currency_rates.csv`：

```csv
observation_time,available_time,currency,policy_rate,ois_1m,ois_3m
2025-01-29T00:00:00Z,2025-01-29T19:00:00Z,USD,4.50,4.32,4.25
```

远期文件 `forward_points.csv`：

```csv
observation_time,available_time,symbol,forward_points_1m,forward_points_3m,spot_reference
2025-01-31T16:00:00Z,2025-01-31T16:00:01Z,EURUSD,0.00165,0.00492,1.0390
```

`observation_time` 表示数据对应时期，`available_time` 表示市场真正可知的时间。修订值可以
具有相同 observation time，但必须有更晚的 available time。因子合并只允许
`available_time <= feature_time`。forward points 使用货币对价格单位，不是 pip 数；
`spot_reference` 是同一 forward 快照对应的即期价，防止用未来或错位 spot 年化。

实现的 carry 因子：

- `rate_differential`：base 减 quote 的政策利率差。
- `curve_slope_differential`：两种货币 OIS 3M–1M slope 之差。
- `forward_discount_1m`：按 30 天年化的 1M forward discount。
- `carry_to_vol_20`：利率差除以 20-bar 年化波动率。

CSV 示例见 `examples/point_in_time/`。

## 受约束因子 DSL

DSL 不执行任意 Python 表达式，只允许配置中的原语和算子：

- 时序：`delta`、`ts_zscore`、`ts_mean`、`ts_std`；
- 横截面：`cs_rank`；
- 条件交互：一个方向因子乘一个非方向状态因子。

每个生成因子记录表达式、父因子、算子、窗口、复杂度和生成顺序。搜索预算由
`max_generated_factors` 硬限制，全部生成候选都计入同一折 FDR 的假设总数，不能只报告最后
入选的公式。

## 运行开发研究

准备 broker CSV、swap 和 PIT 文件后：

```bash
uv run fxtrade factor-data-audit -c configs/factors_broker_carry_dev.yaml
uv run fxtrade factor-mine -c configs/factors_broker_carry_dev.yaml
```

`factor-data-audit` 必须先显示 `broker_ready=true`：所有品种都有 bid/ask、最短历史不少于
8 年、swap 与 carry 覆盖率均不少于 80%。合成数据和 midpoint 数据永远不能通过这一门槛。

该配置结束于 2025-09-15，不重用已经查看过的 Yahoo holdout。主要新增产物：

- `factor_catalog.csv`：固定与 DSL 因子的完整谱系；
- `oos_factor_statistics_by_fold.csv`：paired OOS IC、block-bootstrap p 和 FDR q；
- `cost_stress_by_fold.csv`：1x、1.5x、2x 成本结果；
- `factor_manifest.json`：市场数据哈希、PIT 数据哈希、搜索预算和最终 verdict。
- `frozen_model_status.json`：拒绝时说明原因；只有开发门槛通过才生成带 SHA-256 的
  `frozen_factor_model.json`。

没有真实外部数据时可运行软件验收：

```bash
uv run fxtrade factor-mine -c configs/factors_carry_synthetic.yaml
```

当前确定性合成验收测试了 89 个假设（含 20 个 DSL 表达式），基准成本复合收益 -5.38%，
1.5x 成本 -5.45%，最终为 `rejected_for_trading`。这证明拒绝路径、成本压力和产物工作正常，
不评价 carry 在真实市场中的有效性。

## 晋级逻辑

默认开发晋级至少要求：

- 每折存在通过配置筛选的因子；
- 总交易数不少于 100；
- 至少 75% 开发折盈利；
- 每折 PF 不低于 1.10；
- 1.5x 成本下复合收益仍为正。

开发门槛通过但没有新 holdout 时，只能得到 `research_candidate_requires_new_holdout`；
holdout 也通过时仍只能得到 `research_candidate_requires_paper`。系统不会自动批准真实交易。

## 冻结模型与前向期

只有上述 candidate verdict 才能冻结。模型 JSON 固化：入选因子及谱系、缺失值统计、标准化
参数、线性系数、Platt 校准器、收益假设、研究截止时间、DSL/因子配置和 contract SHA-256。
任何字段被修改、因子配置改变或品种集合变化都会拒绝前向运行。

把严格晚于冻结时间的新 bid/ask、swap 和 PIT 数据追加到原目录后执行：

```bash
uv run fxtrade factor-forward-evaluate \
  --model outputs/factors_broker_carry_dev/frozen_factor_model.json \
  -c configs/factors_broker_carry_forward.yaml \
  -o outputs/factor_forward
```

该命令不重新筛选、不重新拟合，也不会下单；只生成 forward predictions、signals、trades、
equity 和 manifest。未达到配置的 90 天时状态为 `collecting`，达到后也只是
`duration_complete_requires_review`，仍需人工检查并继续累计到计划的 3–6 个月。
