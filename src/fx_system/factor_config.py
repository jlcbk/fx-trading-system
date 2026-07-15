from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import CostConfig, DataConfig, RiskConfig


class FactorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_atr: float = Field(0.70, gt=0)
    stop_atr: float = Field(1.10, gt=0)
    max_holding_hours: int = Field(120, ge=1, le=168)
    train_bars: int = Field(1500, ge=250)
    test_bars: int = Field(400, ge=50)
    step_bars: int = Field(400, ge=50)
    embargo_bars: int = Field(5, ge=0, le=100)
    holdout_bars: int = Field(0, ge=0)
    minimum_train_samples: int = Field(5000, ge=500)
    minimum_expected_net_r: float = Field(0.03, ge=0, le=0.50)
    cost_buffer_r: float = Field(0.02, ge=0, le=0.50)
    minimum_direction_gap: float = Field(0.03, ge=0, le=0.50)
    max_signals_per_timestamp: int = Field(3, ge=1, le=20)
    max_features: int = Field(32, ge=5, le=100)
    maximum_feature_correlation: float = Field(0.90, ge=0.5, le=0.999)
    minimum_feature_coverage: float = Field(0.80, ge=0.5, le=1)
    bootstrap_samples: int = Field(500, ge=100, le=10_000)
    bootstrap_block_bars: int = Field(20, ge=2, le=250)
    factor_fdr_level: float = Field(0.10, gt=0, le=0.50)
    require_fdr_significance: bool = False
    model_c: float = Field(0.10, gt=0, le=100)
    model_l1_ratio: float = Field(0.15, ge=0, le=1)
    calibration_fraction: float = Field(0.20, ge=0.10, le=0.40)
    minimum_calibration_samples: int = Field(1000, ge=100)
    random_state: int = 42
    output_directory: Path = Path("outputs/factors")

    @model_validator(mode="after")
    def non_overlapping_test_windows(self) -> FactorSettings:
        if self.step_bars < self.test_bars:
            raise ValueError("factor.step_bars must be >= test_bars to avoid overlapping OOS folds")
        if 0 < self.holdout_bars < 50:
            raise ValueError("factor.holdout_bars must be zero or at least 50")
        return self

    @property
    def reward_risk(self) -> float:
        return self.target_atr / self.stop_atr


class FactorMiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    factor: FactorSettings = Field(default_factory=FactorSettings)

    @model_validator(mode="after")
    def enforce_execution_limits(self) -> FactorMiningConfig:
        if self.factor.reward_risk > self.risk.max_reward_risk:
            raise ValueError("factor target/stop ratio exceeds risk.max_reward_risk")
        if self.factor.max_holding_hours > self.risk.max_holding_hours:
            raise ValueError("factor holding period exceeds risk.max_holding_hours")
        if self.risk.close_before_weekend:
            raise ValueError(
                "factor mining requires risk.close_before_weekend=false "
                "so labels and execution match"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> FactorMiningConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
