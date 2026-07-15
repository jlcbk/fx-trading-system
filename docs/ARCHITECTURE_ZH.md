# 架构与数据流

## 数据流

```text
CSV / Yahoo / Synthetic / OANDA fxPractice BA
          │
          ▼
  UTC mid 或 bid/ask OHLC 校验与对齐
          │
          ├── point-in-time 利率 / forward / swap
          │               │
          │               ▼
          │       Carry + 受约束因子 DSL
          ├── 单品种策略 ──────┐
          └── 多货币策略 ──────┤
                               ▼
                        加权冲突消解
                               │（收盘信号）
                               ▼
                       下一根 K 线开盘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       组合风险与币种敞口                 成交成本与 FX 换算
              └────────────────┬────────────────┘
                               ▼
                  事件回测 / 模拟订单计划
                               │
                               ▼
               Trades + Equity + Manifest + Report
```

## 关键不变量

1. 策略在时间 `t` 的 close 上生成信号，执行调度使用 `searchsorted(..., side="right")`，因此
   只能在该品种严格晚于 `t` 的 bar 开盘成交。
2. `target_atr / stop_atr <= risk.max_reward_risk <= 1.0`，即使策略参数写得更高，执行引擎
   也会再次裁剪目标距离，风险层还会拒绝越界信号。
3. `max_holding_hours <= 168` 同时由配置验证和执行引擎限制。
4. 所有 PnL、风险、名义本金和货币敞口通过当时可用的 FX 图换算为 USD；非 USD quote
   不使用固定点值。
5. 协整双腿共享 `group_id`；组内任一腿未通过风险检查时，已预开的腿回滚。
6. 回测结束强制平掉全部头寸，不把未实现收益伪装成已实现结果。
7. 同 bar 止盈止损顺序未知时按止损，防止乐观偏差。
8. `enabled` 只表示进入研究；只有额外标记 `paper_enabled` 的策略才能生成 practice 订单计划。
9. 实时数据源删除尚未收盘的最后一根 K 线；practice 计划带到期时间和稳定 client ID，
   重复提交由 broker 幂等键拒绝。
10. 双边数据按可成交侧执行：long 为 ask 入/bid 出，short 为 bid 入/ask 出；止盈止损也检查
    对应可成交侧的 high/low。
11. 宏观、利率、远期和 swap 只按 `available_time` 向后合并；修订值不会回填到其发布时间前。
12. DSL 不执行任意代码；候选数、复杂度、原语和算子均由配置限制，所有候选计入 FDR。

## 当前边界

- Yahoo 仍是 midpoint 公共数据；OANDA candle API 有 bid/ask，但历史 swap 和 point-in-time
  forward/OIS 必须由独立授权数据补充。
- bar 回测无法知道 bar 内路径；保守 stop-first 只能降低而不能消除这一不确定性。
- Carry 数据契约和因子已启用；仓库不分发商业 OIS/forward 历史。宏观 surprise、订单簿和
  新闻数据仍需新的 point-in-time 适配器。
- OANDA 只实现 practice 市价单 + attached stop/target；没有真实资金执行路径。
