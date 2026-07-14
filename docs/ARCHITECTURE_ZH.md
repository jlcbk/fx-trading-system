# 架构与数据流

## 数据流

```text
CSV / Yahoo / Synthetic
          │
          ▼
  UTC OHLC 校验与对齐
          │
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

## 当前边界

- Yahoo 是 midpoint 公共数据，没有历史 bid/ask、可靠 volume 或 broker-specific swap。
- bar 回测无法知道 bar 内路径；保守 stop-first 只能降低而不能消除这一不确定性。
- 宏观 surprise、真实利差 carry、订单簿和新闻策略未启用，因为默认数据契约不包含这些字段。
- OANDA 只实现 practice 市价单 + attached stop/target；没有真实资金执行路径。
