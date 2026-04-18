from __future__ import annotations

import argparse
import gc
import json
import signal
import socket
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from traderbot.config import AppConfig, TIMEFRAME_MINUTES_MAP, ensure_directories, load_config
from traderbot.data.hl_loader import HLDataLoader
from traderbot.data.csv_loader import CSVDataLoader
from traderbot.data.database import create_tables, database_unavailable_reason
from traderbot.env.trading_env import TradingEnv
from traderbot.execution.hl_executor import ExecutionDecision, HyperliquidExecutor
from traderbot.features.engineering import FeatureEngineer, default_feature_columns
from traderbot.rl.backtest import (
    print_metrics,
    run_backtest,
    run_backtest_ensemble,
    save_backtest_result,
)
from traderbot.rl.model_manager import RLModelManager
from traderbot.utils.logger import setup_logger
from traderbot.utils.telegram_notifier import TelegramNotifier

DEFAULT_MULTI_SEEDS = [1, 7, 21, 42, 84, 123, 256, 512, 999, 2004]
_bot_lock: socket.socket | None = None


def _acquire_bot_lock() -> None:
    global _bot_lock  # Mantem a referencia viva na memoria
    _bot_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _bot_lock.bind(("127.0.0.1", 54322))
    except socket.error:
        print("\n[ERRO CRITICO] O Bot ja esta em execucao em outro terminal ou pelo Launcher!")
        print("Feche a outra instancia antes de iniciar uma nova para evitar ordens duplicadas.\n")
        sys.exit(1)


def vecnormalize_stats_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".vecnormalize.pkl")


def build_metric_snapshot(metrics: dict) -> dict[str, float]:
    keys = [
        "total_profit",
        "max_drawdown",
        "profit_per_trade",
        "num_trades",
        "blocked_trades",
        "blocked_by_regime_filter",
        "trades_allowed_by_regime_filter",
        "pct_bars_with_valid_regime",
        "pct_bars_filtered_by_regime",
        "win_rate",
        "blocked_trade_rate",
        "final_equity",
        "exit_by_take_profit",
        "exit_by_stop_loss",
        "exit_by_agent_close",
        "exit_by_margin_call",
        "attempted_reversals",
        "prevented_same_candle_reversals",
        "average_trade_duration_bars",
        "average_win",
        "average_loss",
        "payoff_ratio",
        "profit_factor",
        "trades_closed_by_tp_pct",
        "trades_closed_by_sl_pct",
    ]
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def log_seed_metrics(logger, seed: int, pain_decay_factor: float, metrics: dict) -> None:
    snap = build_metric_snapshot(metrics)
    logger.info(
        "seed=%s | pain_decay_factor=%.4f | total_profit=%.2f | max_drawdown=%.4f | "
        "profit_per_trade=%.4f | num_trades=%s | win_rate=%.2f%% | blocked_trade_rate=%.2f%% | final_equity=%.2f",
        seed,
        pain_decay_factor,
        snap["total_profit"],
        snap["max_drawdown"],
        snap["profit_per_trade"],
        int(snap["num_trades"]),
        snap["win_rate"] * 100.0,
        snap["blocked_trade_rate"] * 100.0,
        snap["final_equity"],
    )


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def build_multi_seed_summary(runs: list[dict], seeds: list[int], acceptance_min_pass_rate: float) -> dict[str, object]:
    accepted_count = sum(1 for row in runs if row.get("accepted"))
    pass_rate = (accepted_count / len(runs)) if runs else 0.0
    total_profits = [float(row.get("total_profit", 0.0)) for row in runs]
    profit_factors = [float(row.get("profit_factor", 0.0)) for row in runs]
    max_drawdowns = [float(row.get("max_drawdown", 0.0)) for row in runs]

    return {
        "num_runs": len(runs),
        "seeds_used": [int(seed) for seed in seeds],
        "accepted_runs": accepted_count,
        "pass_rate": pass_rate,
        "strategy_accepted": pass_rate >= float(acceptance_min_pass_rate),
        "avg_total_profit_per_seed": _safe_mean(total_profits),
        "avg_profit_factor": _safe_mean(profit_factors),
        "avg_max_drawdown": _safe_mean(max_drawdowns),
        "std_total_profit": _safe_std(total_profits),
        "std_profit_factor": _safe_std(profit_factors),
    }


def _latest_multi_seed_summary_path(results_dir: str) -> Path | None:
    candidates = sorted(
        Path(results_dir).glob("multi_train_summary_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def auto_select_model_names(cfg: AppConfig, logger) -> list[str]:
    if cfg.execution.selected_model_names:
        selected = [str(name) for name in cfg.execution.selected_model_names]
        logger.info("Modelos selecionados manualmente via config: %s", selected)
        return selected

    summary_path = _latest_multi_seed_summary_path(cfg.paths.results_dir)
    if summary_path is None or not summary_path.exists():
        logger.info("Nenhum multi_train_summary encontrado; usando ensemble_model_names do config.")
        return [str(name) for name in cfg.execution.ensemble_model_names]

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f) or {}

    selected: list[str] = []
    for run in payload.get("runs", []):
        model_name = Path(str(run.get("model_path", ""))).stem
        profit_factor = float(run.get("profit_factor", 0.0))
        max_drawdown = abs(float(run.get("max_drawdown", 0.0)))
        if model_name and profit_factor > 1.10 and max_drawdown < 0.20:
            selected.append(model_name)

    if selected:
        logger.info(
            "Modelos selecionados automaticamente a partir de %s: %s",
            summary_path.name,
            selected,
        )
        return selected

    logger.info(
        "Nenhum modelo passou no filtro automático (PF > 1.10 e |DD| < 0.20); usando ensemble_model_names do config."
    )
    return [str(name) for name in cfg.execution.ensemble_model_names]


def maybe_normalize_obs(obs: np.ndarray, obs_normalizer) -> np.ndarray:
    if obs_normalizer is None:
        return obs
    return obs_normalizer.normalize_obs(obs.reshape(1, -1))[0]


def split_train_test(df, train_split: float):
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if len(train_df) < 200 or len(test_df) < 200:
        raise ValueError("Dados insuficientes após split. Aumente history_bars.")
    return train_df, test_df


def _infer_bar_minutes(df) -> float | None:
    if len(df.index) < 2:
        return None
    diffs = df.index.to_series().sort_values().diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.dt.total_seconds().median() / 60.0)


def _resample_ohlcv_to_timeframe(raw, timeframe: str, logger):
    target_minutes = TIMEFRAME_MINUTES_MAP.get(str(timeframe).upper())
    if target_minutes is None or target_minutes <= 0:
        return raw

    source_minutes = _infer_bar_minutes(raw)
    if source_minutes is None:
        return raw

    if target_minutes <= round(source_minutes):
        return raw

    rule = f"{int(target_minutes)}min"
    logger.info(
        "Reamostrando histórico para %s | origem~%.0fmin | destino=%smin",
        timeframe,
        source_minutes,
        target_minutes,
    )
    resampled = raw.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "mean",
            "real_volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close", "volume"])
    if resampled.empty:
        raise ValueError(f"Reamostragem para {timeframe} gerou dataset vazio.")
    return resampled


def prepare_datasets(cfg: AppConfig, logger):
    source = str(cfg.data.source).lower().strip()
    if source == "csv":
        logger.info(
            "Carregando CSV (%s) APENAS para inicializar normalizador do modelo PPO...",
            cfg.data.csv_path,
        )
        raw = CSVDataLoader(cfg.data).load()
    else:
        logger.info("Conectando à Hyperliquid para buscar histórico...")
        loader = HLDataLoader(cfg.hyperliquid)
        loader.connect()
        try:
            raw = loader.fetch_historical()
        finally:
            loader.disconnect()

    raw = _resample_ohlcv_to_timeframe(raw, cfg.hyperliquid.timeframe, logger)
    logger.info("Histórico carregado: %s linhas", len(raw))

    fe = FeatureEngineer(cfg.features)
    feat_df = fe.build(raw)
    feat_df["regime_dist_ema_240_raw"] = feat_df["dist_ema_240"].astype(float)
    feat_df["regime_vol_regime_z_raw"] = feat_df["vol_regime_z"].astype(float)
    feat_df["regime_breakout_up_10_flag"] = (feat_df["breakout_up_10"] > 0.0).astype(float)
    feat_df["regime_breakout_down_10_flag"] = (feat_df["breakout_down_10"] > 0.0).astype(float)
    cols = default_feature_columns()

    train_df, test_df = split_train_test(feat_df, cfg.training.train_split)
    logger.info(
        "Split temporal aplicado sem embaralhamento | treino=%s linhas | validação=%s linhas | train_end=%s | val_start=%s",
        len(train_df),
        len(test_df),
        train_df.index[-1] if len(train_df) else None,
        test_df.index[0] if len(test_df) else None,
    )

    if cfg.features.normalize:
        fe.fit_normalizer(train_df, cols)
        train_df = fe.transform(train_df, cols)
        test_df = fe.transform(test_df, cols)

    return train_df, test_df, fe, cols


def evaluate_acceptance(metrics: dict, cfg: AppConfig) -> tuple[bool, str]:
    min_profit = float(cfg.training.acceptance_min_total_profit)
    max_dd_abs = float(cfg.training.max_drawdown_limit)
    min_ppt = float(cfg.training.acceptance_min_profit_per_trade)

    total_profit = float(metrics.get("total_profit", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    profit_per_trade = float(metrics.get("profit_per_trade", 0.0))

    reasons = []
    if total_profit <= min_profit:
        reasons.append(f"total_profit={total_profit:.4f} <= {min_profit:.4f}")
    if abs(max_drawdown) > max_dd_abs:
        reasons.append(f"|drawdown|={abs(max_drawdown):.4f} > {max_dd_abs:.4f}")
    if profit_per_trade <= min_ppt:
        reasons.append(f"profit_per_trade={profit_per_trade:.6f} <= {min_ppt:.6f}")

    accepted = len(reasons) == 0
    reason = "ok" if accepted else "; ".join(reasons)
    return accepted, reason


def prepare_runtime_environment_cfg(cfg: AppConfig, logger):
    env_cfg = deepcopy(cfg.environment)

    if env_cfg.simulation_initial_balance is not None:
        env_cfg.initial_balance = float(env_cfg.simulation_initial_balance)
    elif cfg.execution.validation_balance is not None:
        env_cfg.initial_balance = float(cfg.execution.validation_balance)

    if env_cfg.use_broker_constraints:
        simulation_label = (
            "Configuração Live/Exchange ativada"
            if bool(cfg.execution.allow_live_trading)
            else "Simulação Hyperliquid ativada"
        )
        logger.info(
            "%s | balance=%.2f | volume_min=%.4f | step=%.4f | contract_size=%.4f | max_risk=%.2f%% | hold_threshold=%.2f",
            simulation_label,
            env_cfg.initial_balance,
            env_cfg.broker_volume_min,
            env_cfg.broker_volume_step,
            env_cfg.broker_contract_size,
            env_cfg.max_risk_per_trade * 100.0,
            env_cfg.action_hold_threshold,
        )

    return env_cfg


def train_pipeline(cfg: AppConfig, logger):
    env_cfg = prepare_runtime_environment_cfg(cfg, logger)
    train_env_cfg = deepcopy(env_cfg)
    train_env_cfg.max_episode_steps = int(cfg.training.episode_steps) if cfg.training.episode_steps else None
    train_df, test_df, _, cols = prepare_datasets(cfg, logger)

    logger.info("Inicializando ambiente de treino...")
    manager = RLModelManager(cfg.training)

    def train_env_fn():
        return TradingEnv(df=train_df, feature_cols=cols, cfg=train_env_cfg)

    vec_env = manager.wrap_env(train_env_fn)
    model = manager.build(vec_env)

    logger.info("Iniciando treino PPO por %s timesteps...", cfg.training.total_timesteps)
    manager.train(
        model,
        logger=logger,
        initial_balance=train_env_cfg.initial_balance,
        log_trade_events=cfg.training.log_trade_events,
    )
    train_summary = manager.last_training_summary or {}
    logger.info(
        "Métricas treino | episódios=%s | trades_totais=%s | trades_médios_por_episódio=%.2f",
        int(train_summary.get("episodes", 0.0)),
        int(train_summary.get("total_trades", 0.0)),
        float(train_summary.get("avg_trades_per_episode", 0.0)),
    )
    model_path = manager.save(model, cfg.paths.models_dir)
    stats_path = vecnormalize_stats_path(model_path)
    logger.info("Modelo salvo em: %s", model_path)
    if stats_path.exists():
        logger.info("Estatísticas VecNormalize salvas em: %s", stats_path)

    logger.info("Rodando backtest em dados fora do treino...")
    result = run_backtest(
        model,
        test_df,
        cols,
        env_cfg,
        vecnormalize_path=stats_path,
        logger=logger,
    )
    result.metrics["train_episodes"] = float(train_summary.get("episodes", 0.0))
    result.metrics["train_total_trades"] = float(train_summary.get("total_trades", 0.0))
    result.metrics["train_avg_trades_per_episode"] = float(
        train_summary.get("avg_trades_per_episode", 0.0)
    )
    accepted, reason = evaluate_acceptance(result.metrics, cfg)
    result.metrics["accepted"] = bool(accepted)
    result.metrics["acceptance_reason"] = reason
    logger.info("Aceitação do modelo: %s | %s", accepted, reason)
    logger.info(print_metrics(result.metrics))
    log_seed_metrics(
        logger,
        int(cfg.training.seed) if cfg.training.seed is not None else -1,
        float(cfg.environment.pain_decay_factor),
        result.metrics,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths = save_backtest_result(result, cfg.paths.results_dir, f"backtest_{stamp}")
    logger.info("Resultados salvos: %s", paths)

    return model, model_path, result


def train_multi_pipeline(cfg: AppConfig, logger, seeds: list[int]):
    base_name = cfg.training.model_name
    summary: list[dict] = []

    logger.info("Iniciando treino múltiplo com seeds: %s", seeds)
    for idx, seed in enumerate(seeds, start=1):
        run_cfg = deepcopy(cfg)
        run_cfg.training.seed = int(seed)
        run_cfg.training.model_name = f"{base_name}_s{seed}"

        logger.info(
            "Ciclo %s/%s | seed=%s | model_name=%s",
            idx,
            len(seeds),
            seed,
            run_cfg.training.model_name,
        )

        model = None
        result = None
        try:
            model, model_path, result = train_pipeline(run_cfg, logger)
            metrics = result.metrics
            summary.append(
                {
                    "seed": int(seed),
                    "pain_decay_factor": float(run_cfg.environment.pain_decay_factor),
                    "model_path": str(model_path),
                    "total_profit": float(metrics.get("total_profit", 0.0)),
                    "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                    "num_trades": float(metrics.get("num_trades", 0.0)),
                    "blocked_trades": float(metrics.get("blocked_trades", 0.0)),
                    "blocked_by_regime_filter": float(metrics.get("blocked_by_regime_filter", 0.0)),
                    "trades_allowed_by_regime_filter": float(metrics.get("trades_allowed_by_regime_filter", 0.0)),
                    "pct_bars_with_valid_regime": float(metrics.get("pct_bars_with_valid_regime", 0.0)),
                    "pct_bars_filtered_by_regime": float(metrics.get("pct_bars_filtered_by_regime", 0.0)),
                    "blocked_trade_rate": float(metrics.get("blocked_trade_rate", 0.0)),
                    "win_rate": float(metrics.get("win_rate", 0.0)),
                    "profit_per_trade": float(metrics.get("profit_per_trade", 0.0)),
                    "final_equity": float(metrics.get("final_equity", 0.0)),
                    "exit_by_take_profit": float(metrics.get("exit_by_take_profit", 0.0)),
                    "exit_by_stop_loss": float(metrics.get("exit_by_stop_loss", 0.0)),
                    "exit_by_agent_close": float(metrics.get("exit_by_agent_close", 0.0)),
                    "exit_by_margin_call": float(metrics.get("exit_by_margin_call", 0.0)),
                    "attempted_reversals": float(metrics.get("attempted_reversals", 0.0)),
                    "prevented_same_candle_reversals": float(
                        metrics.get("prevented_same_candle_reversals", 0.0)
                    ),
                    "average_trade_duration_bars": float(metrics.get("average_trade_duration_bars", 0.0)),
                    "average_win": float(metrics.get("average_win", 0.0)),
                    "average_loss": float(metrics.get("average_loss", 0.0)),
                    "payoff_ratio": float(metrics.get("payoff_ratio", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "trades_closed_by_tp_pct": float(metrics.get("trades_closed_by_tp_pct", 0.0)),
                    "trades_closed_by_sl_pct": float(metrics.get("trades_closed_by_sl_pct", 0.0)),
                    "train_total_trades": float(metrics.get("train_total_trades", 0.0)),
                    "accepted": bool(metrics.get("accepted", False)),
                    "acceptance_reason": str(metrics.get("acceptance_reason", "")),
                }
            )
            log_seed_metrics(logger, int(seed), float(run_cfg.environment.pain_decay_factor), metrics)
        finally:
            if model is not None:
                try:
                    env = model.get_env()
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                del model
            if result is not None:
                del result
            gc.collect()

    consolidated = build_multi_seed_summary(summary, seeds, float(cfg.training.acceptance_min_pass_rate))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg.paths.results_dir) / f"multi_train_summary_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": consolidated,
                "runs": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Resumo multi-seed salvo em: %s", out_path)
    logger.info(
        "Pass rate multi-seed: %s/%s (%.1f%%) | estratégia aceita=%s",
        int(consolidated["accepted_runs"]),
        len(summary),
        float(consolidated["pass_rate"]) * 100.0,
        bool(consolidated["strategy_accepted"]),
    )
    logger.info(
        "Resumo consolidado | seeds=%s | lucro_médio=%.2f | PF_médio=%.4f | DD_médio=%.4f | desvio_lucro=%.2f | desvio_PF=%.4f",
        ",".join(str(seed) for seed in consolidated["seeds_used"]),
        float(consolidated["avg_total_profit_per_seed"]),
        float(consolidated["avg_profit_factor"]),
        float(consolidated["avg_max_drawdown"]),
        float(consolidated["std_total_profit"]),
        float(consolidated["std_profit_factor"]),
    )
    for row in summary:
        logger.info(
            "seed=%s | pain_decay_factor=%.4f | lucro=%.2f | dd=%.4f | ppt=%.4f | trades_teste=%s | blocked_rate=%.2f%% | win_rate=%.2f%% | final_equity=%.2f | trades_treino=%s",
            row["seed"],
            row["pain_decay_factor"],
            row["total_profit"],
            row["max_drawdown"],
            row["profit_per_trade"],
            int(row["num_trades"]),
            row["blocked_trade_rate"] * 100.0,
            row["win_rate"] * 100.0,
            row["final_equity"],
            int(row["train_total_trades"]),
        )


def train_ablation_pipeline(cfg: AppConfig, logger, seeds: list[int] | None = None):
    factors = [float(x) for x in cfg.ablation.pain_decay_factors]
    if not factors:
        raise ValueError("A ablação foi habilitada, mas pain_decay_factors está vazio.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ablation_runs: list[dict] = []
    base_model_name = cfg.training.model_name

    for factor in factors:
        run_cfg = deepcopy(cfg)
        run_cfg.environment.pain_decay_factor = factor
        factor_tag = str(factor).replace("-", "m").replace(".", "p")
        run_cfg.training.model_name = f"{base_model_name}_pd{factor_tag}"
        logger.info("Iniciando ablação | pain_decay_factor=%.4f | model_name=%s", factor, run_cfg.training.model_name)

        if seeds:
            train_multi_pipeline(run_cfg, logger, seeds)
            ablation_runs.append({"pain_decay_factor": factor, "mode": "multi_seed"})
        else:
            _, model_path, result = train_pipeline(run_cfg, logger)
            ablation_runs.append(
                {
                    "pain_decay_factor": factor,
                    "mode": "single_seed",
                    "model_path": str(model_path),
                    **build_metric_snapshot(result.metrics),
                    "accepted": bool(result.metrics.get("accepted", False)),
                }
            )

    out_path = Path(cfg.paths.results_dir) / f"ablation_pain_decay_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": ablation_runs}, f, ensure_ascii=False, indent=2)
    logger.info("Resumo da ablação salvo em: %s", out_path)


def load_named_execution_models(cfg: AppConfig, logger) -> list[tuple[str, object]]:
    loaded: list[tuple[str, object]] = []
    selected_model_names = auto_select_model_names(cfg, logger)
    for name in selected_model_names:
        path = Path(cfg.paths.models_dir) / f"{name}.zip"
        if path.exists():
            loaded.append((name, (RLModelManager.load(path), vecnormalize_stats_path(path))))
        else:
            logger.info("Modelo do ensemble não encontrado e será ignorado: %s", path)
    return loaded


def backtest_pipeline(cfg: AppConfig, logger, model_path: str | None):
    env_cfg = prepare_runtime_environment_cfg(cfg, logger)
    _, test_df, _, cols = prepare_datasets(cfg, logger)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {path}")

        model = RLModelManager.load(path)
        result = run_backtest(
            model,
            test_df,
            cols,
            env_cfg,
            vecnormalize_path=vecnormalize_stats_path(path),
            logger=logger,
        )
        logger.info(print_metrics(result.metrics))
        paths = save_backtest_result(result, cfg.paths.results_dir, f"backtest_{stamp}")
        logger.info("Resultados salvos: %s", paths)
        return

    if cfg.execution.ensemble_enabled:
        named_models = load_named_execution_models(cfg, logger)
        if not named_models:
            raise FileNotFoundError(
                "Nenhum modelo do ensemble foi encontrado em "
                f"{Path(cfg.paths.models_dir).resolve()}"
            )

        summary_runs: list[dict] = []
        if not cfg.execution.backtest_ensemble_only:
            for name, model in named_models:
                logger.info("Backtest individual | modelo=%s", name)
                model_obj, stats_path = model
                result = run_backtest(
                    model_obj,
                    test_df,
                    cols,
                    env_cfg,
                    vecnormalize_path=stats_path,
                    logger=logger,
                )
                logger.info("%s | %s", name, print_metrics(result.metrics))
                paths = save_backtest_result(result, cfg.paths.results_dir, f"backtest_{stamp}_{name}")
                logger.info("Resultados salvos (%s): %s", name, paths)
                summary_runs.append(
                    {
                        "model_name": name,
                        "decision_mode": str(result.metrics.get("decision_mode", "single")),
                        "total_profit": float(result.metrics.get("total_profit", 0.0)),
                        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                        "num_trades": float(result.metrics.get("num_trades", 0.0)),
                        "blocked_trades": float(result.metrics.get("blocked_trades", 0.0)),
                        "blocked_by_regime_filter": float(result.metrics.get("blocked_by_regime_filter", 0.0)),
                        "trades_allowed_by_regime_filter": float(
                            result.metrics.get("trades_allowed_by_regime_filter", 0.0)
                        ),
                        "pct_bars_with_valid_regime": float(result.metrics.get("pct_bars_with_valid_regime", 0.0)),
                        "pct_bars_filtered_by_regime": float(
                            result.metrics.get("pct_bars_filtered_by_regime", 0.0)
                        ),
                        "win_rate": float(result.metrics.get("win_rate", 0.0)),
                        "profit_per_trade": float(result.metrics.get("profit_per_trade", 0.0)),
                        "final_equity": float(result.metrics.get("final_equity", 0.0)),
                        "exit_by_take_profit": float(result.metrics.get("exit_by_take_profit", 0.0)),
                        "exit_by_stop_loss": float(result.metrics.get("exit_by_stop_loss", 0.0)),
                        "exit_by_agent_close": float(result.metrics.get("exit_by_agent_close", 0.0)),
                        "exit_by_margin_call": float(result.metrics.get("exit_by_margin_call", 0.0)),
                        "attempted_reversals": float(result.metrics.get("attempted_reversals", 0.0)),
                        "prevented_same_candle_reversals": float(
                            result.metrics.get("prevented_same_candle_reversals", 0.0)
                        ),
                        "average_trade_duration_bars": float(
                            result.metrics.get("average_trade_duration_bars", 0.0)
                        ),
                        "average_win": float(result.metrics.get("average_win", 0.0)),
                        "average_loss": float(result.metrics.get("average_loss", 0.0)),
                        "payoff_ratio": float(result.metrics.get("payoff_ratio", 0.0)),
                        "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
                        "trades_closed_by_tp_pct": float(result.metrics.get("trades_closed_by_tp_pct", 0.0)),
                        "trades_closed_by_sl_pct": float(result.metrics.get("trades_closed_by_sl_pct", 0.0)),
                    }
                )

        ensemble_result = run_backtest_ensemble(
            [model for _, model in named_models],
            test_df,
            cols,
            env_cfg,
            logger=logger,
        )
        logger.info("Ensemble votação | %s", print_metrics(ensemble_result.metrics))
        ensemble_paths = save_backtest_result(
            ensemble_result,
            cfg.paths.results_dir,
            f"backtest_{stamp}_ensemble_vote",
        )
        logger.info("Resultados salvos (ensemble): %s", ensemble_paths)
        summary_runs.append(
            {
                "model_name": "ensemble_vote",
                "decision_mode": str(ensemble_result.metrics.get("decision_mode", "")),
                "total_profit": float(ensemble_result.metrics.get("total_profit", 0.0)),
                "max_drawdown": float(ensemble_result.metrics.get("max_drawdown", 0.0)),
                "num_trades": float(ensemble_result.metrics.get("num_trades", 0.0)),
                "blocked_trades": float(ensemble_result.metrics.get("blocked_trades", 0.0)),
                "blocked_by_regime_filter": float(
                    ensemble_result.metrics.get("blocked_by_regime_filter", 0.0)
                ),
                "trades_allowed_by_regime_filter": float(
                    ensemble_result.metrics.get("trades_allowed_by_regime_filter", 0.0)
                ),
                "pct_bars_with_valid_regime": float(
                    ensemble_result.metrics.get("pct_bars_with_valid_regime", 0.0)
                ),
                "pct_bars_filtered_by_regime": float(
                    ensemble_result.metrics.get("pct_bars_filtered_by_regime", 0.0)
                ),
                "win_rate": float(ensemble_result.metrics.get("win_rate", 0.0)),
                "profit_per_trade": float(ensemble_result.metrics.get("profit_per_trade", 0.0)),
                "final_equity": float(ensemble_result.metrics.get("final_equity", 0.0)),
                "exit_by_take_profit": float(ensemble_result.metrics.get("exit_by_take_profit", 0.0)),
                "exit_by_stop_loss": float(ensemble_result.metrics.get("exit_by_stop_loss", 0.0)),
                "exit_by_agent_close": float(ensemble_result.metrics.get("exit_by_agent_close", 0.0)),
                "exit_by_margin_call": float(ensemble_result.metrics.get("exit_by_margin_call", 0.0)),
                "attempted_reversals": float(ensemble_result.metrics.get("attempted_reversals", 0.0)),
                "prevented_same_candle_reversals": float(
                    ensemble_result.metrics.get("prevented_same_candle_reversals", 0.0)
                ),
                "average_trade_duration_bars": float(
                    ensemble_result.metrics.get("average_trade_duration_bars", 0.0)
                ),
                "average_win": float(ensemble_result.metrics.get("average_win", 0.0)),
                "average_loss": float(ensemble_result.metrics.get("average_loss", 0.0)),
                "payoff_ratio": float(ensemble_result.metrics.get("payoff_ratio", 0.0)),
                "profit_factor": float(ensemble_result.metrics.get("profit_factor", 0.0)),
                "trades_closed_by_tp_pct": float(ensemble_result.metrics.get("trades_closed_by_tp_pct", 0.0)),
                "trades_closed_by_sl_pct": float(ensemble_result.metrics.get("trades_closed_by_sl_pct", 0.0)),
                "ensemble_size": float(ensemble_result.metrics.get("ensemble_size", 0.0)),
                "vote_hold_total": float(ensemble_result.metrics.get("vote_hold_total", 0.0)),
                "vote_buy_total": float(ensemble_result.metrics.get("vote_buy_total", 0.0)),
                "vote_sell_total": float(ensemble_result.metrics.get("vote_sell_total", 0.0)),
                "tie_steps": float(ensemble_result.metrics.get("tie_steps", 0.0)),
            }
        )

        summary_path = Path(cfg.paths.results_dir) / f"backtest_{stamp}_comparison.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "initial_balance": float(env_cfg.initial_balance),
                        "decision_mode": "individual_and_ensemble_vote",
                        "models_tested": [name for name, _ in named_models],
                    },
                    "runs": summary_runs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Resumo comparativo salvo em: %s", summary_path)
        return

    path = Path(cfg.paths.models_dir) / f"{cfg.training.model_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}")

    model = RLModelManager.load(path)
    result = run_backtest(
        model,
        test_df,
        cols,
        env_cfg,
        vecnormalize_path=vecnormalize_stats_path(path),
        logger=logger,
    )
    logger.info(print_metrics(result.metrics))
    paths = save_backtest_result(result, cfg.paths.results_dir, f"backtest_{stamp}")
    logger.info("Resultados salvos: %s", paths)


def build_live_observation(
    cfg: AppConfig,
    executor: HyperliquidExecutor,
    feature_cols: list[str],
    fe: FeatureEngineer,
) -> tuple[np.ndarray, dict[str, object]]:
    df = executor.get_live_market_window()
    feat = fe.build(df)
    feat_raw = feat.copy()
    if cfg.features.normalize:
        feat = fe.transform(feat, feature_cols)

    latest = feat.iloc[-1]
    latest_raw = feat_raw.iloc[-1]
    current_price = float(latest_raw["close"])
    obs = latest[feature_cols].to_numpy(dtype="float32")
    position_snapshot = executor.get_position_snapshot()
    side = float(position_snapshot["side"])
    unrealized = float(position_snapshot["unrealized_pnl"])
    entry_price = float(position_snapshot.get("avg_entry_price", 0.0))
    time_in_position = float(position_snapshot.get("time_in_position", 0.0))
    if side != 0.0 and entry_price > 0:
        stop_move = (
            (current_price - (entry_price * (1.0 - cfg.environment.stop_loss_pct * side))) / entry_price
        ) * side
        take_move = (
            ((entry_price * (1.0 + cfg.environment.take_profit_pct * side)) - current_price) / entry_price
        ) * side
        entry_distance_from_ema_240 = float(position_snapshot.get("entry_distance_from_ema_240", 0.0))
    else:
        stop_move = 0.0
        take_move = 0.0
        entry_distance_from_ema_240 = 0.0
    regime_dist_ema_240 = float(latest_raw.get("regime_dist_ema_240_raw", latest_raw.get("dist_ema_240", 0.0)))
    regime_vol_regime_z = float(latest_raw.get("regime_vol_regime_z_raw", latest_raw.get("vol_regime_z", 0.0)))
    regime_valid = (
        abs(regime_dist_ema_240) >= float(cfg.environment.regime_min_abs_dist_ema_240)
        and regime_vol_regime_z > float(cfg.environment.regime_min_vol_regime_z)
    )
    extras = np.array(
        [
            side,
            unrealized,
            time_in_position,
            float(stop_move),
            float(take_move),
            float(entry_distance_from_ema_240),
        ],
        dtype="float32",
    )
    live_context = {
        "entry_distance_from_ema_240": float(latest_raw.get("dist_ema_240", 0.0)),
        "reference_price": current_price,
        "bar_timestamp": latest_raw.name.isoformat() if hasattr(latest_raw.name, "isoformat") else str(latest_raw.name),
        "regime_valid_for_entry": bool(regime_valid),
        "regime_dist_ema_240": regime_dist_ema_240,
        "regime_vol_regime_z": regime_vol_regime_z,
        "candle_open": float(latest_raw.get("open", current_price)),
        "candle_high": float(latest_raw.get("high", current_price)),
        "candle_low": float(latest_raw.get("low", current_price)),
        "candle_close": current_price,
        "candle_volume": float(latest_raw.get("volume", 0.0)),
        "rsi_14": float(latest_raw.get("rsi_14", 0.0)),
        "dist_ema_240": float(latest_raw.get("dist_ema_240", 0.0)),
        "dist_ema_960": float(latest_raw.get("dist_ema_960", 0.0)),
        "atr_pct": float(latest_raw.get("atr_pct", 0.0)),
        "volatility_24": float(latest_raw.get("volatility_24", 0.0)),
        "vol_regime_z": float(latest_raw.get("vol_regime_z", 0.0)),
        "range_pct": float(latest_raw.get("range_pct", 0.0)),
        "range_compression_10": float(latest_raw.get("range_compression_10", 0.0)),
        "range_expansion_3": float(latest_raw.get("range_expansion_3", 0.0)),
        "breakout_up_10": float(latest_raw.get("breakout_up_10", 0.0)),
        "breakout_down_10": float(latest_raw.get("breakout_down_10", 0.0)),
        "volume_zscore_20": float(latest_raw.get("volume_zscore_20", 0.0)),
        "volume_ratio_20": float(latest_raw.get("volume_ratio_20", 0.0)),
    }
    return np.concatenate([obs, extras]).astype(np.float32, copy=False), live_context


def infer_latest_action(model, obs: np.ndarray):
    action, _ = model.predict(obs, deterministic=True)
    return float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))


def action_bucket(raw_action: float, hold_threshold: float) -> int:
    if raw_action > hold_threshold:
        return 1
    if raw_action < -hold_threshold:
        return -1
    return 0


def action_bucket_label(bucket: int) -> str:
    return {1: "buy", -1: "sell", 0: "hold"}[bucket]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _human_runtime_block_reason(reason: Any) -> str:
    reason_text = str(reason or "").strip().lower()
    mapping = {
        "regime_filter": "entrada bloqueada pelo filtro de regime",
        "cooldown": "entrada bloqueada pelo cooldown",
        "order_cooldown": "entrada bloqueada por cooldown de ordem",
        "duplicate_cycle": "entrada bloqueada para evitar ordem duplicada na mesma vela",
        "open_position_locked": "sinal ignorado porque existe posicao travada por TP/SL ou posicao em manutencao",
        "prevented_same_candle_reversal": "reversao impedida na mesma vela",
        "min_notional_risk": "entrada pulada porque o notional minimo violaria o risco maximo",
        "min_notional": "entrada bloqueada por notional minimo da exchange",
        "volume_min": "entrada bloqueada por volume minimo da exchange",
    }
    return mapping.get(reason_text, reason_text or "sem bloqueio")


def _runtime_execution_action(result: dict[str, Any], position_side: int) -> str:
    if result.get("error"):
        return "error"
    if result.get("opened_position"):
        return "opened_position"
    if result.get("closed_position"):
        return "closed_position"
    if result.get("blocked_reason"):
        return "blocked_entry"
    if position_side != 0:
        return "position_held"
    return "no_entry"


def _runtime_event_code(result: dict[str, Any], position_side: int) -> str:
    if result.get("error"):
        return "runtime.cycle_error"
    if result.get("opened_position"):
        return "execution.trade_open"
    if result.get("closed_position"):
        return "execution.trade_close"
    blocked_reason = str(result.get("blocked_reason") or "").strip().lower()
    if blocked_reason:
        return f"blocked.{blocked_reason}"
    if position_side != 0:
        return "execution.position_held"
    return "runtime.no_entry"


def _close_event_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("close_event") or result.get("stop_take_event") or {}


def _safe_notifier_call(
    logger,
    notifier: TelegramNotifier | None,
    method_name: str,
    *args,
    **kwargs,
) -> bool:
    if notifier is None:
        return False

    try:
        method = getattr(notifier, method_name)
    except AttributeError:
        logger.error("TelegramNotifier nao possui o metodo %s.", method_name)
        return False

    try:
        return bool(method(*args, **kwargs))
    except Exception:
        logger.exception("Falha ao executar notificacao Telegram (%s).", method_name)
        return False


def _notify_position_closed_from_result(
    cfg: AppConfig,
    result: dict[str, Any],
    logger,
    notifier: TelegramNotifier | None,
) -> None:
    if notifier is None or not bool(result.get("closed_position", False)):
        return

    close_event = _close_event_from_result(result)
    raw_side = int(close_event.get("side", 0) or 0)
    position_side = "long" if raw_side > 0 else ("short" if raw_side < 0 else None)

    pnl_value_raw = close_event.get("pnl")
    if pnl_value_raw in (None, ""):
        pnl_value_raw = close_event.get("gross_pnl")
    pnl_value = None if pnl_value_raw in (None, "") else _as_float(pnl_value_raw)

    exit_price_raw = close_event.get("exit_price")
    exit_price = None if exit_price_raw in (None, "") else _as_float(exit_price_raw)

    _safe_notifier_call(
        logger,
        notifier,
        "notify_position_closed",
        asset=str(cfg.hyperliquid.symbol).strip().upper(),
        pnl=pnl_value,
        side=position_side,
        reason=str(close_event.get("trigger") or result.get("trigger") or ""),
        exit_price=exit_price,
    )


def _close_open_position_on_shutdown(
    cfg: AppConfig,
    executor: HyperliquidExecutor,
    logger,
    notifier: TelegramNotifier | None,
    trigger: str,
) -> bool:
    try:
        snapshot = executor.get_position_snapshot()
    except Exception as exc:
        logger.exception("Falha ao consultar posicao no encerramento do runtime.")
        _safe_notifier_call(
            logger,
            notifier,
            "notify_critical_error",
            "Falha ao verificar posicao aberta no encerramento do Traderbot",
            f"{type(exc).__name__}: {exc}",
            status="Offline",
        )
        return True

    if int(snapshot.get("side", 0) or 0) == 0:
        return False

    logger.warning(
        "Runtime encerrando com posicao aberta; tentando fechar | trigger=%s | snapshot=%s",
        trigger,
        json.dumps(snapshot, ensure_ascii=False),
    )

    try:
        close_result = executor.close_open_position(trigger=trigger)
    except Exception as exc:
        logger.exception("Falha ao fechar posicao no encerramento do runtime.")
        _safe_notifier_call(
            logger,
            notifier,
            "notify_critical_error",
            "Falha ao fechar posicao na parada do Traderbot",
            f"{type(exc).__name__}: {exc}",
            status="Offline",
        )
        return True

    logger.info("Fechamento de seguranca do runtime | %s", json.dumps(close_result, ensure_ascii=False))

    if bool(close_result.get("closed_position", False)):
        _notify_position_closed_from_result(cfg, close_result, logger, notifier)
        return False

    _safe_notifier_call(
        logger,
        notifier,
        "notify_critical_error",
        "Falha ao fechar posicao na parada do Traderbot",
        str(close_result.get("error") or close_result.get("message") or "Posicao permaneceu aberta."),
        status="Offline",
    )
    return True


def _install_runtime_signal_handlers(logger) -> dict[signal.Signals, Any]:
    previous_handlers: dict[signal.Signals, Any] = {}

    def _handle_shutdown_signal(signum, _frame) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        logger.warning("Sinal %s recebido; iniciando encerramento do runtime.", signal_name)
        raise KeyboardInterrupt(f"shutdown signal {signal_name}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        current_signal = getattr(signal, signal_name, None)
        if current_signal is None:
            continue
        previous_handlers[current_signal] = signal.getsignal(current_signal)
        signal.signal(current_signal, _handle_shutdown_signal)

    return previous_handlers


def _restore_runtime_signal_handlers(previous_handlers: dict[signal.Signals, Any]) -> None:
    for current_signal, previous_handler in previous_handlers.items():
        signal.signal(current_signal, previous_handler)


def _handle_runtime_side_effects(
    cfg: AppConfig,
    result: dict[str, Any],
    logger,
    notifier: TelegramNotifier | None = None,
) -> None:
    symbol = str(cfg.hyperliquid.symbol).strip().upper()
    position_snapshot = result.get("position_snapshot") or {}

    if bool(result.get("opened_position", False)):
        open_side = str(result.get("trade_direction", "")).strip().lower()
        open_price = _as_float(result.get("open_price") or position_snapshot.get("avg_entry_price"))
        open_amount = _as_float(result.get("position_size", result.get("volume", result.get("adjusted_volume", 0.0))))
        sizing = result.get("sizing") or {}
        open_notional = _as_float(sizing.get("adjusted_notional"))
        if open_notional <= 0 and open_price > 0 and open_amount > 0:
            open_notional = open_price * open_amount
        if notifier is not None and open_side in {"buy", "sell"} and open_price > 0 and open_amount > 0:
            _safe_notifier_call(
                logger,
                notifier,
                "notify_order_executed",
                asset=symbol,
                side=open_side,
                price=open_price,
                quantity=open_notional if open_notional > 0 else open_amount,
                tp=result.get("take_profit_price"),
                sl=result.get("stop_loss_price"),
            )

    if bool(result.get("closed_position", False)):
        _notify_position_closed_from_result(cfg, result, logger, notifier)


def _build_decision_reason(vote_info: dict[str, Any], final_action: float, hold_threshold: float) -> str:
    bucket = str(vote_info.get("bucket", "hold")).upper()
    if vote_info.get("mode") == "ensemble":
        votes = vote_info.get("votes") or {}
        return (
            f"Ensemble decidiu {bucket} com votos "
            f"buy={int(votes.get('buy', 0))}, hold={int(votes.get('hold', 0))}, sell={int(votes.get('sell', 0))}; "
            f"acao final={final_action:.4f}; threshold_hold={hold_threshold:.2f}"
        )
    raw_action = _as_float(vote_info.get("raw_action", final_action))
    return (
        f"PPO individual decidiu {bucket} com acao bruta={raw_action:.4f}; "
        f"acao final={final_action:.4f}; threshold_hold={hold_threshold:.2f}"
    )


def _build_runtime_cycle_log(
    cfg: AppConfig,
    live_context: dict[str, Any],
    vote_info: dict[str, Any],
    action: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    position_snapshot = result.get("position_snapshot") or {}
    position_side = int(position_snapshot.get("side", 0) or 0)
    blocked_reason = result.get("blocked_reason")
    execution_action = _runtime_execution_action(result, position_side)
    event_code = _runtime_event_code(result, position_side)
    hold_threshold = float(cfg.environment.action_hold_threshold)
    final_action = float(action)
    decision_reason = _build_decision_reason(vote_info, final_action, hold_threshold)
    regime_valid = bool(live_context["regime_valid_for_entry"])
    trade_direction = str(result.get("trade_direction", "") or "").upper()

    feature_snapshot = {
        "rsi_14": _as_float(live_context.get("rsi_14")),
        "dist_ema_240": _as_float(live_context.get("dist_ema_240")),
        "dist_ema_960": _as_float(live_context.get("dist_ema_960")),
        "atr_pct": _as_float(live_context.get("atr_pct")),
        "volatility_24": _as_float(live_context.get("volatility_24")),
        "vol_regime_z": _as_float(live_context.get("vol_regime_z")),
        "range_pct": _as_float(live_context.get("range_pct")),
        "range_compression_10": _as_float(live_context.get("range_compression_10")),
        "range_expansion_3": _as_float(live_context.get("range_expansion_3")),
        "breakout_up_10": _as_float(live_context.get("breakout_up_10")),
        "breakout_down_10": _as_float(live_context.get("breakout_down_10")),
        "volume_zscore_20": _as_float(live_context.get("volume_zscore_20")),
        "volume_ratio_20": _as_float(live_context.get("volume_ratio_20")),
    }

    market_snapshot = {
        "bar_timestamp": str(live_context["bar_timestamp"]),
        "open": _as_float(live_context.get("candle_open")),
        "high": _as_float(live_context.get("candle_high")),
        "low": _as_float(live_context.get("candle_low")),
        "close": _as_float(live_context.get("candle_close")),
        "volume": _as_float(live_context.get("candle_volume")),
        "reference_price": _as_float(live_context.get("reference_price")),
    }

    decision = {
        "decision_mode": str(cfg.execution.decision_mode),
        "trade_direction": trade_direction,
        "action_hold_threshold": hold_threshold,
        "final_action": final_action,
        "signal_bucket": action_bucket_label(action_bucket(final_action, hold_threshold)),
        "vote_bucket": vote_info.get("bucket"),
        "votes": vote_info.get("votes"),
        "model_votes": vote_info.get("model_votes"),
        "raw_actions": vote_info.get("raw_actions"),
        "raw_action": vote_info.get("raw_action"),
        "tie_hold": bool(vote_info.get("tie_hold", False)),
        "confidence_pct": abs(final_action) * 100.0,
        "reason": decision_reason,
    }

    filters = {
        "regime_valid_for_entry": regime_valid,
        "regime_dist_ema_240": _as_float(live_context.get("regime_dist_ema_240")),
        "regime_vol_regime_z": _as_float(live_context.get("regime_vol_regime_z")),
        "regime_thresholds": {
            "min_abs_dist_ema_240": float(cfg.environment.regime_min_abs_dist_ema_240),
            "min_vol_regime_z": float(cfg.environment.regime_min_vol_regime_z),
        },
        "blocked_reason": blocked_reason,
        "blocked_reason_human": _human_runtime_block_reason(blocked_reason) if blocked_reason else None,
        "blocked_by_min_notional": bool(result.get("blocked_by_min_notional", False)),
        "bumped_to_min_notional": bool(result.get("bumped_to_min_notional", False)),
        "skipped_due_to_min_notional_risk": bool(result.get("skipped_due_to_min_notional_risk", False)),
        "ignored_signal": bool(result.get("ignored_signal", False)),
        "attempted_reversal": bool(result.get("attempted_reversal", False)),
        "prevented_same_candle_reversal": bool(result.get("prevented_same_candle_reversal", False)),
        "force_exit_only_by_tp_sl": bool(result.get("force_exit_only_by_tp_sl", False)),
    }

    execution = {
        "action_taken": execution_action,
        "opened_trade": bool(result.get("opened_position", False)),
        "closed_trade": bool(result.get("closed_position", False)),
        "position_side": position_side,
        "position_label": "LONG" if position_side > 0 else ("SHORT" if position_side < 0 else "FLAT"),
        "position_is_open": position_side != 0,
        "position_size": _as_float(result.get("position_size", result.get("volume", 0.0))),
        "position_avg_entry_price": _as_float(position_snapshot.get("avg_entry_price")),
        "position_unrealized_pnl": _as_float(position_snapshot.get("unrealized_pnl")),
        "position_time_in_bars": _as_float(position_snapshot.get("time_in_position")),
        "entry_distance_from_ema_240": _as_float(position_snapshot.get("entry_distance_from_ema_240")),
        "stop_loss_price": _as_float(result.get("stop_loss_price")),
        "take_profit_price": _as_float(result.get("take_profit_price")),
        "exit_reason": result.get("exit_reason") or (result.get("stop_take_event", {}) or {}).get("trigger"),
        "error": result.get("error"),
        "response_ok": bool(result.get("ok", False)),
    }

    sizing = {
        "available_to_trade": result.get("available_to_trade"),
        "available_to_trade_source_field": result.get("available_to_trade_source_field"),
        "sizing_balance_used": result.get("sizing_balance_used"),
        "sizing_balance_source": result.get("sizing_balance_source"),
        "risk_pct": result.get("risk_pct"),
        "risk_amount": result.get("risk_amount"),
        "stop_loss_pct": result.get("stop_loss_pct"),
        "stop_distance_price": result.get("stop_distance_price"),
        "target_notional": result.get("target_notional"),
        "raw_volume": result.get("raw_volume"),
        "adjusted_volume": result.get("adjusted_volume"),
        "adjusted_notional": result.get("adjusted_notional"),
        "notional_value": result.get("notional_value"),
        "min_notional_usd": result.get("min_notional_usd"),
        "effective_risk_amount": result.get("effective_risk_amount"),
        "exceeds_target_risk": result.get("exceeds_target_risk"),
        "leverage_configured": result.get("leverage_configured"),
        "leverage_applied": result.get("leverage_applied"),
        "leverage_used": result.get("leverage_used"),
        "margin_estimated_consumed": result.get("margin_estimated_consumed"),
        "has_explicit_leverage_logic": result.get("has_explicit_leverage_logic"),
    }

    system_state = {
        "network": result.get("network", cfg.hyperliquid.network),
        "symbol": cfg.hyperliquid.symbol,
        "timeframe": cfg.hyperliquid.timeframe,
        "execution_mode": result.get("execution_mode", cfg.execution.execution_mode),
        "operational_mode": result.get("mode"),
        "decision_mode": str(cfg.execution.decision_mode),
        "order_slippage": float(getattr(cfg.execution, "order_slippage", 0.0)),
        "model_slippage_pct": float(cfg.environment.slippage_pct),
    }

    return {
        "event": "runtime_cycle",
        "event_code": event_code,
        "module": "runtime",
        "stage": "decision_cycle",
        "timestamp": str(result.get("timestamp", "")),
        "bar_timestamp": str(live_context["bar_timestamp"]),
        "symbol": cfg.hyperliquid.symbol,
        "timeframe": cfg.hyperliquid.timeframe,
        "network": result.get("network", cfg.hyperliquid.network),
        "decision_reason": decision_reason,
        "market_snapshot": market_snapshot,
        "feature_snapshot": feature_snapshot,
        "decision": decision,
        "filters": filters,
        "execution": execution,
        "sizing": sizing,
        "system_state": system_state,
        # flat compatibility keys for launcher/runtime consumers
        "decision_mode": decision["decision_mode"],
        "regime_valid": filters["regime_valid_for_entry"],
        "regime_dist_ema_240": filters["regime_dist_ema_240"],
        "regime_vol_regime_z": filters["regime_vol_regime_z"],
        "votes": decision["votes"],
        "model_votes": decision["model_votes"],
        "vote_bucket": decision["vote_bucket"],
        "tie_hold": decision["tie_hold"],
        "final_action": decision["final_action"],
        "reference_price": market_snapshot["reference_price"],
        "position_side": execution["position_side"],
        "position_label": execution["position_label"],
        "position_is_open": execution["position_is_open"],
        "opened_trade": execution["opened_trade"],
        "closed_trade": execution["closed_trade"],
        "position_size": execution["position_size"],
        "position_avg_entry_price": execution["position_avg_entry_price"],
        "position_unrealized_pnl": execution["position_unrealized_pnl"],
        "position_time_in_bars": execution["position_time_in_bars"],
        "entry_distance_from_ema_240": execution["entry_distance_from_ema_240"],
        "available_to_trade": sizing["available_to_trade"],
        "available_to_trade_source_field": sizing["available_to_trade_source_field"],
        "sizing_balance_used": sizing["sizing_balance_used"],
        "sizing_balance_source": sizing["sizing_balance_source"],
        "order_slippage": system_state["order_slippage"],
        "model_slippage_pct": system_state["model_slippage_pct"],
        "risk_pct": sizing["risk_pct"],
        "risk_amount": sizing["risk_amount"],
        "stop_loss_pct": sizing["stop_loss_pct"],
        "stop_distance_price": sizing["stop_distance_price"],
        "target_notional": sizing["target_notional"],
        "raw_volume": sizing["raw_volume"],
        "adjusted_volume": sizing["adjusted_volume"],
        "adjusted_notional": sizing["adjusted_notional"],
        "notional_value": sizing["notional_value"],
        "min_notional_usd": sizing["min_notional_usd"],
        "blocked_by_min_notional": filters["blocked_by_min_notional"],
        "bumped_to_min_notional": filters["bumped_to_min_notional"],
        "effective_risk_amount": sizing["effective_risk_amount"],
        "exceeds_target_risk": sizing["exceeds_target_risk"],
        "skipped_due_to_min_notional_risk": filters["skipped_due_to_min_notional_risk"],
        "leverage_configured": sizing["leverage_configured"],
        "leverage_applied": sizing["leverage_applied"],
        "leverage_used": sizing["leverage_used"],
        "margin_estimated_consumed": sizing["margin_estimated_consumed"],
        "has_explicit_leverage_logic": sizing["has_explicit_leverage_logic"],
        "blocked_reason": filters["blocked_reason"],
        "blocked_reason_human": filters["blocked_reason_human"],
        "error": execution["error"],
        "stop_loss_price": execution["stop_loss_price"],
        "take_profit_price": execution["take_profit_price"],
        "exit_reason": execution["exit_reason"],
        "force_exit_only_by_tp_sl": filters["force_exit_only_by_tp_sl"],
        "ignored_signal": filters["ignored_signal"],
        "attempted_reversal": filters["attempted_reversal"],
        "prevented_same_candle_reversal": filters["prevented_same_candle_reversal"],
        "execution_action": execution_action,
    }


def load_execution_models(cfg: AppConfig, logger):
    if cfg.execution.ensemble_enabled:
        named_models = load_named_execution_models(cfg, logger)
        models = [model for _, model in named_models]
        loaded_names = [name for name, _ in named_models]
        if models:
            logger.info("Ensemble carregado com modelos: %s", loaded_names)
            return models
        logger.info("Nenhum modelo do ensemble encontrado, voltando para modelo único.")

    path = Path(cfg.paths.models_dir) / f"{cfg.training.model_name}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}")
    return RLModelManager.load(path), vecnormalize_stats_path(path)


def execution_models_available(cfg: AppConfig, logger) -> bool:
    if cfg.execution.ensemble_enabled:
        selected_model_names = auto_select_model_names(cfg, logger)
        return any((Path(cfg.paths.models_dir) / f"{name}.zip").exists() for name in selected_model_names)
    return (Path(cfg.paths.models_dir) / f"{cfg.training.model_name}.zip").exists()


def majority_vote_bucket(raw_actions: list[float], hold_threshold: float) -> tuple[int, dict[str, int]]:
    buckets = [action_bucket(raw_action, hold_threshold) for raw_action in raw_actions]
    counts = {
        "buy": buckets.count(1),
        "hold": buckets.count(0),
        "sell": buckets.count(-1),
    }
    max_votes = max(counts.values()) if counts else 0
    winners = [label for label, votes in counts.items() if votes == max_votes]
    if len(winners) != 1:
        return 0, counts
    return {"buy": 1, "hold": 0, "sell": -1}[winners[0]], counts


def infer_latest_action_ensemble(models, obs: np.ndarray, hold_threshold: float):
    if not isinstance(models, list):
        model, obs_normalizer = models
        model_obs = maybe_normalize_obs(obs, obs_normalizer)
        raw_action = infer_latest_action(model, model_obs)
        bucket = action_bucket(raw_action, hold_threshold)
        return raw_action, {
            "mode": "single",
            "raw_action": raw_action,
            "bucket": action_bucket_label(bucket),
        }

    raw_actions = []
    model_votes: dict[str, float] = {}
    for index, entry in enumerate(models, start=1):
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            model_name, model, obs_normalizer = entry
        else:
            model_name = f"model_{index}"
            model, obs_normalizer = entry
        model_obs = maybe_normalize_obs(obs, obs_normalizer)
        raw_action = infer_latest_action(model, model_obs)
        raw_actions.append(raw_action)
        label = str(model_name or f"model_{index}")
        if label in model_votes:
            label = f"{label}_{index}"
        model_votes[label] = float(raw_action)

    winner_bucket, counts = majority_vote_bucket(raw_actions, hold_threshold)
    if winner_bucket == 1:
        winner_actions = [abs(raw_action) for raw_action in raw_actions if action_bucket(raw_action, hold_threshold) == 1]
        action_value = float(np.mean(winner_actions)) if winner_actions else float(hold_threshold)
        final_action = float(np.clip(max(action_value, hold_threshold), hold_threshold, 1.0))
    elif winner_bucket == -1:
        winner_actions = [abs(raw_action) for raw_action in raw_actions if action_bucket(raw_action, hold_threshold) == -1]
        action_value = float(np.mean(winner_actions)) if winner_actions else float(hold_threshold)
        final_action = float(np.clip(-max(action_value, hold_threshold), -1.0, -hold_threshold))
    else:
        final_action = 0.0

    return final_action, {
        "mode": "ensemble",
        "raw_actions": [float(x) for x in raw_actions],
        "model_votes": model_votes,
        "final_action": final_action,
        "bucket": action_bucket_label(winner_bucket),
        "votes": counts,
        "tie_hold": counts["buy"] == counts["sell"] or list(counts.values()).count(max(counts.values())) > 1,
    }


def prepare_live_inference_context(cfg: AppConfig, logger):
    train_df, _, fe, cols = prepare_datasets(cfg, logger)
    return fe, cols, train_df


def timeframe_sleep_seconds(timeframe: str) -> int:
    minutes = TIMEFRAME_MINUTES_MAP.get(str(timeframe).upper())
    if minutes is None:
        return 60
    return max(60, int(minutes) * 60)


def sleep_until_next_cycle(timeframe: str) -> None:
    cycle_seconds = timeframe_sleep_seconds(timeframe)
    now = time.time()
    next_boundary = ((int(now) // cycle_seconds) + 1) * cycle_seconds
    sleep_seconds = max(1.0, next_boundary - now + 1.0)
    time.sleep(sleep_seconds)


def run_execution_pipeline(
    cfg: AppConfig,
    logger,
    model_path: str | None,
    notifier: TelegramNotifier | None = None,
):
    env_cfg = prepare_runtime_environment_cfg(cfg, logger)
    fe, cols, train_df = prepare_live_inference_context(cfg, logger)

    def build_obs_normalizer(stats_path: Path | None):
        if stats_path is None or not stats_path.exists():
            return None
        sample_df = train_df.tail(min(len(train_df), 512)).copy()
        dummy_env = DummyVecEnv([lambda: TradingEnv(df=sample_df, feature_cols=cols, cfg=env_cfg)])
        vec_env = VecNormalize.load(str(stats_path), dummy_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env

    def build_model_entries(model_override: str | None = None):
        if model_override:
            path = Path(model_override)
            if not path.exists():
                raise FileNotFoundError(f"Modelo não encontrado em {path}")
            return (RLModelManager.load(path), build_obs_normalizer(vecnormalize_stats_path(path)))

        if cfg.execution.ensemble_enabled:
            named_models = load_named_execution_models(cfg, logger)
            if named_models:
                return [
                    (name, model, build_obs_normalizer(stats_path))
                    for name, (model, stats_path) in named_models
                ]

        models = load_execution_models(cfg, logger)
        if isinstance(models, list):
            return [(model, build_obs_normalizer(stats_path)) for model, stats_path in models]

        model, stats_path = models
        return (model, build_obs_normalizer(stats_path))

    if not model_path and not execution_models_available(cfg, logger):
        logger.info("Modelo não encontrado, treinando novo modelo...")
        _, _, _ = train_pipeline(cfg, logger)
    model_entries = build_model_entries(model_path)

    executor = HyperliquidExecutor(
        cfg.hyperliquid,
        cfg.execution,
        env_cfg,
        stop_loss_pct=cfg.environment.stop_loss_pct,
        take_profit_pct=cfg.environment.take_profit_pct,
        slippage_pct=cfg.environment.slippage_pct,
    )
    executor.connect()
    runtime_balance_context = executor.get_sizing_balance_context(cfg.execution.execution_mode)
    logger.info(
        "Executor conectado | network=%s | execution_mode=%s | base_url=%s | decision_mode=%s | modelos=%s | available_to_trade=%.2f | sizing_balance_used=%.2f | saldo_fonte=%s | order_slippage=%.4f | model_slippage=%.5f",
        cfg.hyperliquid.network,
        cfg.execution.execution_mode,
        executor._base_url(),
        cfg.execution.decision_mode,
        cfg.execution.selected_model_names if cfg.execution.selected_model_names else "auto",
        float(runtime_balance_context.get("available_to_trade", 0.0)),
        float(runtime_balance_context.get("sizing_balance_used", 0.0)),
        str(runtime_balance_context.get("sizing_balance_source")),
        float(cfg.execution.order_slippage),
        float(cfg.environment.slippage_pct),
    )
    last_retrain = time.time()
    last_runtime_error_signature = ""
    last_runtime_error_notified_at = 0.0
    shutdown_trigger = "runtime_shutdown"
    previous_signal_handlers = _install_runtime_signal_handlers(logger)
    shutdown_had_error = False

    try:
        while True:
            try:
                obs, live_context = build_live_observation(cfg, executor, cols, fe)
                action, vote_info = infer_latest_action_ensemble(
                    model_entries,
                    obs,
                    float(env_cfg.action_hold_threshold),
                )
                result = executor.execute(
                    ExecutionDecision(
                        action=action,
                        reason=f"ação inferida pelo PPO ({json.dumps(vote_info, ensure_ascii=False)})",
                        entry_distance_from_ema_240=float(live_context["entry_distance_from_ema_240"]),
                        regime_valid_for_entry=bool(live_context["regime_valid_for_entry"]),
                        regime_dist_ema_240=float(live_context["regime_dist_ema_240"]),
                        regime_vol_regime_z=float(live_context["regime_vol_regime_z"]),
                        reference_price=float(live_context["reference_price"]),
                        bar_timestamp=str(live_context["bar_timestamp"]),
                    )
                )
                cycle_log = _build_runtime_cycle_log(cfg, live_context, vote_info, action, result)
                logger.info("Ciclo runtime HL | %s", json.dumps(cycle_log, ensure_ascii=False))
                _handle_runtime_side_effects(cfg, result, logger, notifier=notifier)
            except Exception as exc:
                pause_seconds = max(1.0, float(cfg.execution.pause_on_error_seconds))
                logger.exception(
                    "Erro no ciclo do runtime Hyperliquid; pausando %.1fs antes de tentar novamente.",
                    pause_seconds,
                )
                now = time.time()
                error_signature = f"{type(exc).__name__}:{exc}"
                if (
                    error_signature != last_runtime_error_signature
                    or (now - last_runtime_error_notified_at) >= max(300.0, pause_seconds * 5.0)
                ):
                    _safe_notifier_call(
                        logger,
                        notifier,
                        "notify_critical_error",
                        "Erro no ciclo do runtime Hyperliquid",
                        error_signature,
                        status="Online",
                    )
                    last_runtime_error_signature = error_signature
                    last_runtime_error_notified_at = now
                time.sleep(pause_seconds)
                continue

            if cfg.retraining.enabled:
                interval_s = cfg.retraining.interval_minutes * 60
                if time.time() - last_retrain >= interval_s:
                    logger.info("Executando re-treinamento automático.")
                    _, _, _ = train_pipeline(cfg, logger)
                    fe, cols, train_df = prepare_live_inference_context(cfg, logger)
                    model_entries = build_model_entries()
                    executor.disconnect()
                    executor.connect()
                    last_retrain = time.time()

            sleep_until_next_cycle(cfg.hyperliquid.timeframe)
    except KeyboardInterrupt as exc:
        shutdown_trigger = "shutdown_signal"
        logger.info("Encerramento do runtime solicitado por interrupcao/sinal: %s", exc)
        raise
    except SystemExit:
        shutdown_trigger = "shutdown_signal"
        logger.info("Encerramento do runtime solicitado por SystemExit.")
        raise
    except Exception:
        shutdown_trigger = "shutdown_after_error"
        raise
    finally:
        _restore_runtime_signal_handlers(previous_signal_handlers)
        shutdown_had_error = _close_open_position_on_shutdown(cfg, executor, logger, notifier, shutdown_trigger)
        if notifier is not None and shutdown_trigger != "shutdown_after_error" and not shutdown_had_error:
            _safe_notifier_call(logger, notifier, "notify_stopped")
        executor.disconnect()


def check_hyperliquid_pipeline(cfg: AppConfig, logger):
    logger.info(
        "Diagnóstico Hyperliquid | wallet=%s network=%s symbol=%s timeframe=%s",
        cfg.hyperliquid.wallet_address,
        cfg.hyperliquid.network,
        cfg.hyperliquid.symbol,
        cfg.hyperliquid.timeframe,
    )
    executor = HyperliquidExecutor(
        cfg.hyperliquid,
        cfg.execution,
        cfg.environment,
        stop_loss_pct=cfg.environment.stop_loss_pct,
        take_profit_pct=cfg.environment.take_profit_pct,
        slippage_pct=cfg.environment.slippage_pct,
    )
    executor.connect()
    try:
        status = executor.check_connection()
        status.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        spec = executor.get_symbol_trading_spec()
        risk_state = executor.get_account_risk_state()
        sizing_context = executor.get_sizing_balance_context(cfg.execution.execution_mode)
        sizing_balance = float(sizing_context.get("sizing_balance_used", 0.0))
        validation_sizing = executor.calculate_order_volume(
            balance=sizing_balance,
            dynamic_risk_pct=float(cfg.environment.max_risk_per_trade),
            defensive=False,
            available_to_trade=float(sizing_context.get("available_to_trade", 0.0)),
            available_to_trade_source_field=str(sizing_context.get("available_to_trade_source_field")),
            sizing_balance_source=str(sizing_context.get("sizing_balance_source")),
        )
        status["trading_spec"] = {
            "volume_min": spec.volume_min,
            "volume_step": spec.volume_step,
            "volume_max": spec.volume_max,
            "contract_size": spec.contract_size,
            "point": spec.point,
        }
        status["execution_mode"] = cfg.execution.execution_mode
        status["available_to_trade_source_field"] = sizing_context.get("available_to_trade_source_field")
        status["sizing_balance_used"] = sizing_balance
        status["sizing_balance_source"] = sizing_context.get("sizing_balance_source")
        status["validation_sizing"] = validation_sizing
        status["account_risk_state"] = risk_state
        status["event"] = "healthcheck"
        status["event_code"] = "healthcheck.completed"
        status["module"] = "connectivity"
        status["stage"] = "post_check"
        status["symbol"] = cfg.hyperliquid.symbol
        status["timeframe"] = cfg.hyperliquid.timeframe
        status["health_summary"] = {
            "connected": bool(status.get("connected")),
            "can_trade": bool(status.get("can_trade")),
            "wallet_loaded": bool(status.get("wallet_loaded")),
            "symbol_found": bool(status.get("symbol_found")),
            "history_bars": int(status.get("history_bars", 0) or 0),
            "healthy": bool(status.get("connected")) and bool(status.get("can_trade")),
        }
        status["account_summary"] = {
            "available_to_trade": float(sizing_context.get("available_to_trade", 0.0) or 0.0),
            "sizing_balance_used": sizing_balance,
            "sizing_balance_source": sizing_context.get("sizing_balance_source"),
            "equity": float(risk_state.get("equity", 0.0) or 0.0),
            "drawdown_pct": float(risk_state.get("drawdown_pct", 0.0) or 0.0),
            "defensive": bool(risk_state.get("defensive", False)),
        }
        status["risk_limits"] = {
            "max_risk_per_trade": float(cfg.environment.max_risk_per_trade),
            "stop_loss_pct": float(cfg.environment.stop_loss_pct),
            "take_profit_pct": float(cfg.environment.take_profit_pct),
            "action_hold_threshold": float(cfg.environment.action_hold_threshold),
        }
    finally:
        executor.disconnect()
    logger.info("Status Hyperliquid: %s", json.dumps(status, ensure_ascii=False))


def smoke_hyperliquid_pipeline(
    cfg: AppConfig,
    logger,
    side: str = "buy",
    wait_seconds: float = 3.0,
    order_slippage: float | None = None,
    network_override: str | None = None,
    allow_mainnet: bool = False,
):
    smoke_cfg = deepcopy(cfg)
    if network_override:
        smoke_cfg.hyperliquid.network = str(network_override).lower().strip()
    smoke_cfg.execution.execution_mode = "exchange"
    smoke_cfg.execution.allow_live_trading = True

    network = str(smoke_cfg.hyperliquid.network).lower().strip()
    if network == "mainnet" and not allow_mainnet:
        raise RuntimeError(
            "Smoke test em mainnet bloqueado por segurança. Use --allow-mainnet para confirmar."
        )

    env_cfg = prepare_runtime_environment_cfg(smoke_cfg, logger)
    effective_smoke_slippage = float(
        cfg.execution.order_slippage if order_slippage is None else order_slippage
    )
    logger.info(
        "Smoke test Hyperliquid | network=%s | execution_mode=%s | side=%s | wait_seconds=%.1f | smoke_slippage=%.4f",
        smoke_cfg.hyperliquid.network,
        smoke_cfg.execution.execution_mode,
        str(side).lower().strip(),
        float(wait_seconds),
        effective_smoke_slippage,
    )
    executor = HyperliquidExecutor(
        smoke_cfg.hyperliquid,
        smoke_cfg.execution,
        env_cfg,
        stop_loss_pct=smoke_cfg.environment.stop_loss_pct,
        take_profit_pct=smoke_cfg.environment.take_profit_pct,
        slippage_pct=smoke_cfg.environment.slippage_pct,
    )
    executor.connect()
    try:
        payload = executor.run_smoke_test(
            side=side,
            wait_seconds=wait_seconds,
            order_slippage=effective_smoke_slippage,
        )
    finally:
        executor.disconnect()

    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    payload["event"] = "smoke_test"
    payload["event_code"] = "system.smoke_test"
    payload["module"] = "execution"
    payload["stage"] = "smoke_test_complete"
    payload["symbol"] = smoke_cfg.hyperliquid.symbol
    payload["timeframe"] = smoke_cfg.hyperliquid.timeframe
    payload["decision_summary"] = {
        "side_requested": str(side).lower().strip(),
        "opened_position": bool(payload.get("opened_position")),
        "closed_position": bool(payload.get("closed_position")),
        "ok": bool(payload.get("ok")),
    }
    payload["risk_limits"] = {
        "max_risk_per_trade": float(smoke_cfg.environment.max_risk_per_trade),
        "stop_loss_pct": float(smoke_cfg.environment.stop_loss_pct),
        "take_profit_pct": float(smoke_cfg.environment.take_profit_pct),
        "order_slippage": effective_smoke_slippage,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(smoke_cfg.paths.results_dir) / f"smoke_hyperliquid_{stamp}_{network}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Resultado smoke salvo em: %s", output_path)
    logger.info("Smoke test Hyperliquid | %s", json.dumps(payload, ensure_ascii=False))
    return payload


def close_hyperliquid_position_pipeline(
    cfg: AppConfig,
    logger,
    notifier: TelegramNotifier | None = None,
):
    logger.info(
        "Fechamento manual Hyperliquid | network=%s execution_mode=%s",
        cfg.hyperliquid.network,
        cfg.execution.execution_mode,
    )
    executor = HyperliquidExecutor(
        cfg.hyperliquid,
        cfg.execution,
        cfg.environment,
        stop_loss_pct=cfg.environment.stop_loss_pct,
        take_profit_pct=cfg.environment.take_profit_pct,
        slippage_pct=cfg.environment.slippage_pct,
    )
    executor.connect()
    try:
        payload = executor.close_open_position(trigger="manual_close_from_cli")
    finally:
        executor.disconnect()
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    payload["event"] = "manual_close"
    payload["event_code"] = "risk.manual_close"
    payload["module"] = "execution"
    payload["stage"] = "manual_close_complete"
    payload["symbol"] = cfg.hyperliquid.symbol
    payload["timeframe"] = cfg.hyperliquid.timeframe
    payload["decision_summary"] = {
        "ok": bool(payload.get("ok")),
        "trigger": payload.get("trigger", "manual_close_from_cli"),
        "closed_position": bool(payload.get("closed_position", False)),
        "error": payload.get("error"),
    }
    logger.info("Fechamento manual Hyperliquid | %s", json.dumps(payload, ensure_ascii=False))
    _notify_position_closed_from_result(cfg, payload, logger, notifier)
    return payload


def apply_cli_runtime_overrides(cfg: AppConfig, args, logger) -> AppConfig:
    overrides = []
    if getattr(args, "network_override", None):
        cfg.hyperliquid.network = str(args.network_override).lower().strip()
        overrides.append(f"network={cfg.hyperliquid.network}")
    if getattr(args, "execution_mode_override", None):
        cfg.execution.execution_mode = str(args.execution_mode_override).lower().strip()
        overrides.append(f"execution_mode={cfg.execution.execution_mode}")
    if getattr(args, "allow_live_trading", False):
        cfg.execution.allow_live_trading = True
        overrides.append("allow_live_trading=true")
    if getattr(args, "order_slippage_override", None) is not None:
        cfg.execution.order_slippage = float(args.order_slippage_override)
        overrides.append(f"order_slippage={cfg.execution.order_slippage}")
    if overrides:
        logger.info("Overrides de runtime aplicados via CLI: %s", ", ".join(overrides))
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Traderbot RL + Hyperliquid")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo de configuração")
    parser.add_argument(
        "--network-override",
        choices=["mainnet", "testnet"],
        default=None,
        help="Override da rede Hyperliquid sem editar o config.yaml",
    )
    parser.add_argument(
        "--execution-mode-override",
        choices=["paper_local", "exchange"],
        default=None,
        help="Override do execution_mode sem editar o config.yaml",
    )
    parser.add_argument(
        "--allow-live-trading",
        action="store_true",
        help="Ativa allow_live_trading só para esta execução",
    )
    parser.add_argument(
        "--order-slippage-override",
        type=float,
        default=None,
        help="Override do order_slippage só para esta execução",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Treina PPO e já executa backtest")
    p_train_multi = sub.add_parser("train-multi", help="Roda múltiplos ciclos de treino/backtest por seed")
    p_train_multi.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_MULTI_SEEDS),
        help="Lista de 10 seeds separadas por vírgula. Ex: 1,7,21,42,84,123,256,512,999,2004",
    )

    p_backtest = sub.add_parser("backtest", help="Avalia modelo salvo")
    p_backtest.add_argument("--model-path", type=str, default=None, help="Caminho do modelo .zip")

    p_run = sub.add_parser("run", help="Roda loop de inferência e execução (paper/live)")
    p_run.add_argument("--model-path", type=str, default=None, help="Caminho do modelo .zip")
    sub.add_parser("check-hyperliquid", help="Valida conexão Hyperliquid (wallet/rede/símbolo)")
    sub.add_parser("close-hyperliquid-position", help="Fecha a posição aberta atual na Hyperliquid")
    p_smoke = sub.add_parser(
        "smoke-hyperliquid",
        help="Abre e fecha uma ordem mínima na Hyperliquid para validar execução operacional",
    )
    p_smoke.add_argument("--side", choices=["buy", "sell"], default="buy", help="Lado da ordem do smoke test")
    p_smoke.add_argument(
        "--wait-seconds",
        type=float,
        default=3.0,
        help="Tempo de espera entre abertura e fechamento manual do smoke test",
    )
    p_smoke.add_argument(
        "--smoke-slippage",
        type=float,
        default=None,
        help="Slippage operacional do smoke test para cruzar o livro com ordem IOC",
    )
    p_smoke.add_argument(
        "--network",
        choices=["mainnet", "testnet"],
        default=None,
        help="Override opcional da rede só para o smoke test",
    )
    p_smoke.add_argument(
        "--allow-mainnet",
        action="store_true",
        help="Confirma explicitamente o smoke test na mainnet",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_directories(cfg)

    logger = setup_logger(cfg.app_name, cfg.paths.logs_dir)
    cfg = apply_cli_runtime_overrides(cfg, args, logger)
    notifier = TelegramNotifier(logger=logger)

    try:
        if args.command == "backtest":
            try:
                create_tables()
                logger.info("Tabelas PostgreSQL verificadas/inicializadas com sucesso.")
            except Exception:
                logger.info(
                    "PostgreSQL indisponivel. Persistencia em banco desativada nesta execucao: %s",
                    database_unavailable_reason() or "erro de conexao",
                )

        if args.command == "run":
            _acquire_bot_lock()
            _safe_notifier_call(logger, notifier, "notify_startup")

        logger.info("Iniciando pipeline com comando: %s", args.command)

        if args.command == "train":
            if cfg.ablation.enabled:
                train_ablation_pipeline(cfg, logger)
            else:
                train_pipeline(cfg, logger)
        elif args.command == "train-multi":
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            if not seeds:
                raise ValueError("Nenhuma seed válida informada em --seeds")
            if cfg.ablation.enabled:
                train_ablation_pipeline(cfg, logger, seeds=seeds)
            else:
                train_multi_pipeline(cfg, logger, seeds)
        elif args.command == "backtest":
            backtest_pipeline(cfg, logger, args.model_path)
        elif args.command == "run":
            run_execution_pipeline(cfg, logger, args.model_path, notifier=notifier)
        elif args.command == "check-hyperliquid":
            check_hyperliquid_pipeline(cfg, logger)
        elif args.command == "close-hyperliquid-position":
            close_hyperliquid_position_pipeline(cfg, logger, notifier=notifier)
        elif args.command == "smoke-hyperliquid":
            smoke_hyperliquid_pipeline(
                cfg,
                logger,
                side=args.side,
                wait_seconds=args.wait_seconds,
                order_slippage=args.smoke_slippage,
                network_override=args.network,
                allow_mainnet=bool(args.allow_mainnet),
            )
        else:
            raise ValueError("Comando inválido")
    except Exception as exc:
        _safe_notifier_call(
            logger,
            notifier,
            "notify_critical_error",
            "Falha critica no arranque/execucao do Traderbot",
            f"{type(exc).__name__}: {exc}",
            status="Offline",
        )
        raise
    except KeyboardInterrupt:
        logger.info("Traderbot encerrado por interrupcao.")


if __name__ == "__main__":
    main()
