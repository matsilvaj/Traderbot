from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class PathsConfig:
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"


@dataclass
class HyperliquidConfig:
    enabled: bool = True
    wallet_address: Optional[str] = None
    private_key: Optional[str] = None
    network: str = "mainnet"
    symbol: str = "BTC"
    timeframe: str = "1h"
    history_bars: int = 10_000
    live_window_bars: int = 2_000


@dataclass
class DataConfig:
    source: str = "csv"
    csv_path: Optional[str] = "data/binance_btcusdt_h1.csv"
    timestamp_column: str = "timestamp"


@dataclass
class FeatureConfig:
    rsi_period: int = 14
    ma_short_period: int = 9
    ma_long_period: int = 21
    momentum_period: int = 3
    normalize: bool = True


@dataclass
class EnvironmentConfig:
    initial_balance: float = 250.0
    simulation_initial_balance: Optional[float] = None
    max_risk_per_trade: float = 0.01
    min_risk_per_trade: float = 0.001
    action_hold_threshold: float = 0.65
    regime_min_abs_dist_ema_240: float = 0.025
    regime_min_vol_regime_z: float = 0.3
    taker_fee_pct: float = 0.00045
    slippage_pct: float = 0.00010
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    force_exit_only_by_tp_sl: bool = True
    reward_clip_limit: float = 5.0
    pain_decay_factor: float = 0.0
    overtrade_penalty: float = 0.0
    blocked_trade_penalty: float = 0.0
    close_profit_bonus: float = 0.0
    close_loss_penalty: float = 0.0
    broker_min_notional_usd: float = 10.0
    use_broker_constraints: bool = True
    broker_volume_min: float = 0.0001
    broker_volume_step: float = 0.0001
    broker_volume_max: float = 22.0
    broker_contract_size: float = 1.0
    broker_point: float = 1.0
    block_trade_on_excess_risk: bool = False
    max_episode_steps: Optional[int] = None
    trade_cooldown_steps: int = 1
    margin_call_drawdown_pct: float = 0.8
    margin_call_penalty: float = 10.0


@dataclass
class AblationConfig:
    enabled: bool = False
    pain_decay_factors: list[float] = field(default_factory=lambda: [0.0, 0.001, 0.002])


@dataclass
class TrainingConfig:
    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    policy: str = "MlpPolicy"
    device: str = "auto"
    num_envs: int = 1
    episode_steps: int = 10_000
    model_name: str = "ppo_btc_h1"
    train_split: float = 0.7
    seed: int = 42
    log_interval_steps: int = 5_000
    log_trade_events: bool = False
    acceptance_min_total_profit: float = 0.0
    max_drawdown_limit: float = 0.40
    acceptance_min_profit_per_trade: float = 0.0
    acceptance_min_pass_rate: float = 0.6


@dataclass
class ExecutionConfig:
    execution_mode: str = "paper_local"
    decision_mode: str = "ensemble_majority_vote"
    order_slippage: float = 0.05
    defensive_drawdown_pct: float = 0.02
    defensive_risk_multiplier: float = 0.5
    validation_balance: float = 250.0
    ensemble_enabled: bool = True
    backtest_ensemble_only: bool = False
    selected_model_names: list[str] = field(default_factory=list)
    ensemble_model_names: list[str] = field(
        default_factory=lambda: ["ppo_btc_h1_s123", "ppo_btc_h1_s512", "ppo_btc_h1_s2004"]
    )
    allow_live_trading: bool = False
    min_seconds_between_orders: int = 10
    api_retry_attempts: int = 2
    api_retry_delay_seconds: int = 3
    pause_on_error_seconds: int = 60


@dataclass
class RetrainingConfig:
    enabled: bool = False
    interval_minutes: int = 60


@dataclass
class AppConfig:
    app_name: str = "traderbot-rl"
    paths: PathsConfig = field(default_factory=PathsConfig)
    hyperliquid: HyperliquidConfig = field(default_factory=HyperliquidConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)


TIMEFRAME_MINUTES_MAP: dict[str, int] = {
    "1M": 1,
    "2M": 2,
    "3M": 3,
    "4M": 4,
    "10M": 10,
    "15M": 15,
    "30M": 30,
    "1H": 60,
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _build_config(raw: dict[str, Any]) -> AppConfig:
    execution_raw = dict(raw.get("execution", {}))
    if "execution_mode" not in execution_raw and "mode" in execution_raw:
        legacy_mode = str(execution_raw.get("mode", "")).strip().lower()
        if legacy_mode == "paper":
            execution_raw["execution_mode"] = "paper_local"
        elif legacy_mode in {"live", "exchange"}:
            execution_raw["execution_mode"] = "exchange"
        else:
            execution_raw["execution_mode"] = legacy_mode
    execution_raw.pop("mode", None)

    return AppConfig(
        app_name=raw.get("app_name", "traderbot-rl"),
        paths=PathsConfig(**raw.get("paths", {})),
        hyperliquid=HyperliquidConfig(**raw.get("hyperliquid", {})),
        data=DataConfig(**raw.get("data", {})),
        features=FeatureConfig(**raw.get("features", {})),
        environment=EnvironmentConfig(**raw.get("environment", {})),
        ablation=AblationConfig(**raw.get("ablation", {})),
        training=TrainingConfig(**raw.get("training", {})),
        execution=ExecutionConfig(**execution_raw),
        retraining=RetrainingConfig(**raw.get("retraining", {})),
    )


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Carrega configuração do YAML + variáveis de ambiente."""
    config_path = Path(config_path)
    env_path = config_path.resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)

    base = asdict(AppConfig())
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        base = _deep_update(base, user_cfg)

    cfg = _build_config(base)

    # Permite segredos via .env
    def _clean_env(name: str) -> Optional[str]:
        # Fallback para arquivos .env com BOM (primeira chave vira "\ufeffNOME")
        value = os.getenv(name)
        if value is None:
            value = os.getenv(f"\ufeff{name}")
        if value is None:
            return None
        value = value.strip().strip("\"'").strip()
        return value or None

    wallet_env = _clean_env("HL_WALLET_ADDRESS")
    if wallet_env is not None:
        cfg.hyperliquid.wallet_address = wallet_env

    pk_env = _clean_env("HL_PRIVATE_KEY")
    if pk_env is not None:
        cfg.hyperliquid.private_key = pk_env

    return cfg


def ensure_directories(cfg: AppConfig) -> None:
    """Garante criação dos diretórios de trabalho."""
    for folder in [cfg.paths.models_dir, cfg.paths.results_dir, cfg.paths.logs_dir]:
        Path(folder).mkdir(parents=True, exist_ok=True)

