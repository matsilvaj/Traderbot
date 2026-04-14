from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from traderbot.config import AppConfig, MT5_TIMEFRAME_MAP, ensure_directories, load_config
from traderbot.data.csv_loader import CSVDataLoader
from traderbot.env.trading_env import TradingEnv
from traderbot.execution.mt5_executor import ExecutionDecision, MT5TradeExecutor
from traderbot.features.engineering import FeatureEngineer, default_feature_columns
from traderbot.rl.backtest import (
    print_metrics,
    run_backtest,
    run_backtest_ensemble,
    save_backtest_result,
)
from traderbot.rl.model_manager import RLModelManager
from traderbot.utils.logger import setup_logger


def vecnormalize_stats_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".vecnormalize.pkl")


def build_metric_snapshot(metrics: dict) -> dict[str, float]:
    keys = [
        "total_profit",
        "max_drawdown",
        "profit_per_trade",
        "num_trades",
        "win_rate",
        "blocked_trade_rate",
        "final_equity",
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


def prepare_datasets(cfg: AppConfig, logger):
    source = str(cfg.data.source).lower().strip()
    if source == "csv":
        logger.info("Carregando histórico a partir do CSV: %s", cfg.data.csv_path)
        raw = CSVDataLoader(cfg.data).load()
    else:
        logger.info("Conectando ao MT5 para buscar histórico...")
        executor = MT5TradeExecutor(
            cfg.mt5,
            cfg.execution,
            stop_loss_pct=cfg.environment.stop_loss_pct,
            take_profit_pct=cfg.environment.take_profit_pct,
        )
        executor.connect()
        try:
            raw = executor.fetch_historical()
        finally:
            executor.disconnect()

    logger.info("Histórico carregado: %s linhas", len(raw))

    fe = FeatureEngineer(cfg.features)
    feat_df = fe.build(raw)
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
    max_dd_abs = float(cfg.training.acceptance_max_drawdown_abs)
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

    env_cfg.use_fixed_trade_volume = not bool(cfg.execution.use_dynamic_position_sizing)
    env_cfg.fixed_trade_volume = float(cfg.execution.lot)
    if env_cfg.use_fixed_trade_volume:
        env_cfg.block_trade_on_excess_risk = False
    else:
        env_cfg.risk_per_trade = float(cfg.execution.risk_per_trade)

    if env_cfg.use_broker_constraints:
        executor = MT5TradeExecutor(
            cfg.mt5,
            cfg.execution,
            stop_loss_pct=env_cfg.stop_loss_pct,
            take_profit_pct=env_cfg.take_profit_pct,
        )
        executor.connect()
        try:
            spec = executor.get_symbol_trading_spec()
        finally:
            executor.disconnect()

        env_cfg.broker_volume_min = spec.volume_min
        env_cfg.broker_volume_step = spec.volume_step
        env_cfg.broker_volume_max = spec.volume_max
        env_cfg.broker_contract_size = spec.contract_size
        env_cfg.broker_point = spec.point
        logger.info(
            "Simulação realista ativada | balance=%.2f | volume_min=%.4f | step=%.4f | contract_size=%.4f | fixed_volume=%s",
            env_cfg.initial_balance,
            env_cfg.broker_volume_min,
            env_cfg.broker_volume_step,
            env_cfg.broker_contract_size,
            env_cfg.fixed_trade_volume if env_cfg.use_fixed_trade_volume else "dinâmico",
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

        _, model_path, result = train_pipeline(run_cfg, logger)
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
                "blocked_trade_rate": float(metrics.get("blocked_trade_rate", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "profit_per_trade": float(metrics.get("profit_per_trade", 0.0)),
                "final_equity": float(metrics.get("final_equity", 0.0)),
                "train_total_trades": float(metrics.get("train_total_trades", 0.0)),
                "accepted": bool(metrics.get("accepted", False)),
                "acceptance_reason": str(metrics.get("acceptance_reason", "")),
            }
        )
        log_seed_metrics(logger, int(seed), float(run_cfg.environment.pain_decay_factor), metrics)

    accepted_count = sum(1 for row in summary if row["accepted"])
    pass_rate = (accepted_count / len(summary)) if summary else 0.0
    strategy_accepted = pass_rate >= float(cfg.training.acceptance_min_pass_rate)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg.paths.results_dir) / f"multi_train_summary_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "num_runs": len(summary),
                    "accepted_runs": accepted_count,
                    "pass_rate": pass_rate,
                    "strategy_accepted": strategy_accepted,
                },
                "runs": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Resumo multi-seed salvo em: %s", out_path)
    logger.info(
        "Pass rate multi-seed: %s/%s (%.1f%%) | estratégia aceita=%s",
        accepted_count,
        len(summary),
        pass_rate * 100.0,
        strategy_accepted,
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
    for name in cfg.execution.ensemble_model_names:
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
                    "win_rate": float(result.metrics.get("win_rate", 0.0)),
                    "profit_per_trade": float(result.metrics.get("profit_per_trade", 0.0)),
                    "final_equity": float(result.metrics.get("final_equity", 0.0)),
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
                "win_rate": float(ensemble_result.metrics.get("win_rate", 0.0)),
                "profit_per_trade": float(ensemble_result.metrics.get("profit_per_trade", 0.0)),
                "final_equity": float(ensemble_result.metrics.get("final_equity", 0.0)),
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
    executor: MT5TradeExecutor,
    feature_cols: list[str],
    fe: FeatureEngineer,
    obs_normalizer=None,
) -> np.ndarray:
    df = executor.fetch_historical()
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
    if side != 0.0 and entry_price > 0:
        stop_move = ((current_price - (entry_price * (1.0 - cfg.environment.stop_loss_pct * side))) / entry_price) * side
        take_move = (((entry_price * (1.0 + cfg.environment.take_profit_pct * side)) - current_price) / entry_price) * side
        entry_distance_from_ema_240 = (
            (entry_price - float(latest_raw.get("ema_240", entry_price))) / (float(latest_raw.get("ema_240", entry_price)) + 1e-12)
        )
    else:
        stop_move = 0.0
        take_move = 0.0
        entry_distance_from_ema_240 = 0.0
    extras = np.array(
        [
            side,
            unrealized,
            0.0,
            float(stop_move),
            float(take_move),
            float(entry_distance_from_ema_240),
        ],
        dtype="float32",
    )
    return maybe_normalize_obs(
        np.concatenate([obs, extras]).astype(np.float32, copy=False),
        obs_normalizer,
    ).astype(np.float32, copy=False)


def infer_latest_action(model, obs: np.ndarray):
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


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


def execution_models_available(cfg: AppConfig) -> bool:
    if cfg.execution.ensemble_enabled:
        return any((Path(cfg.paths.models_dir) / f"{name}.zip").exists() for name in cfg.execution.ensemble_model_names)
    return (Path(cfg.paths.models_dir) / f"{cfg.training.model_name}.zip").exists()


def infer_latest_action_ensemble(models, obs: np.ndarray):
    if not isinstance(models, list):
        model, _stats_path = models
        action = infer_latest_action(model, obs)
        return action, {"mode": "single", "votes": {str(action): 1}}

    votes = [int(model.predict(obs, deterministic=True)[0]) for model, _stats_path in models]
    counts = {action: votes.count(action) for action in (0, 1, 2)}
    max_votes = max(counts.values())
    winners = [action for action, count in counts.items() if count == max_votes]
    action = 0 if len(winners) != 1 else winners[0]
    return action, {"mode": "ensemble", "votes": counts}


def prepare_live_inference_context(cfg: AppConfig, logger):
    train_df, _, fe, cols = prepare_datasets(cfg, logger)
    if cfg.features.normalize:
        fe.fit_normalizer(train_df, cols)
    return fe, cols, train_df


def timeframe_sleep_seconds(timeframe: str) -> int:
    minutes = MT5_TIMEFRAME_MAP.get(str(timeframe).upper())
    if minutes is None:
        return 60
    return max(60, int(minutes) * 60)


def sleep_until_next_cycle(timeframe: str) -> None:
    cycle_seconds = timeframe_sleep_seconds(timeframe)
    now = time.time()
    next_boundary = ((int(now) // cycle_seconds) + 1) * cycle_seconds
    sleep_seconds = max(1.0, next_boundary - now + 1.0)
    time.sleep(sleep_seconds)


def run_execution_pipeline(cfg: AppConfig, logger, model_path: str | None):
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

    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {path}")
        models = (RLModelManager.load(path), vecnormalize_stats_path(path))
    elif not execution_models_available(cfg):
        logger.info("Modelo não encontrado, treinando novo modelo...")
        _, _, _ = train_pipeline(cfg, logger)
        models = load_execution_models(cfg, logger)
    else:
        models = load_execution_models(cfg, logger)

    if isinstance(models, list):
        model_entries = [(model, build_obs_normalizer(stats_path)) for model, stats_path in models]
    else:
        model, stats_path = models
        model_entries = (model, build_obs_normalizer(stats_path))

    executor = MT5TradeExecutor(
        cfg.mt5,
        cfg.execution,
        stop_loss_pct=cfg.environment.stop_loss_pct,
        take_profit_pct=cfg.environment.take_profit_pct,
    )
    executor.connect()
    logger.info("Executor conectado no modo: %s", cfg.execution.mode)
    last_retrain = time.time()

    try:
        while True:
            if isinstance(model_entries, list):
                obs = build_live_observation(cfg, executor, cols, fe)
                normalized_models = [
                    (model, maybe_normalize_obs(obs, obs_normalizer))
                    for model, obs_normalizer in model_entries
                ]
                votes = [int(model.predict(model_obs, deterministic=True)[0]) for model, model_obs in normalized_models]
                counts = {action: votes.count(action) for action in (0, 1, 2)}
                max_votes = max(counts.values())
                winners = [action for action, count in counts.items() if count == max_votes]
                action = 0 if len(winners) != 1 else winners[0]
                vote_info = {"mode": "ensemble", "votes": counts}
            else:
                model, obs_normalizer = model_entries
                obs = build_live_observation(cfg, executor, cols, fe, obs_normalizer=obs_normalizer)
                action, vote_info = infer_latest_action_ensemble((model, None), obs)
            result = executor.execute(
                ExecutionDecision(
                    action=action,
                    reason=f"ação inferida pelo PPO ({json.dumps(vote_info, ensure_ascii=False)})",
                )
            )
            logger.info(
                "Decisão=%s | Votos=%s | Resultado=%s",
                action,
                json.dumps(vote_info, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )

            if cfg.retraining.enabled:
                interval_s = cfg.retraining.interval_minutes * 60
                if time.time() - last_retrain >= interval_s:
                    logger.info("Executando re-treinamento automático.")
                    _, _, _ = train_pipeline(cfg, logger)
                    models = load_execution_models(cfg, logger)
                    fe, cols, train_df = prepare_live_inference_context(cfg, logger)
                    if isinstance(models, list):
                        model_entries = [(model, build_obs_normalizer(stats_path)) for model, stats_path in models]
                    else:
                        model, stats_path = models
                        model_entries = (model, build_obs_normalizer(stats_path))
                    executor.disconnect()
                    executor.connect()
                    last_retrain = time.time()

            sleep_until_next_cycle(cfg.mt5.timeframe)
    finally:
        executor.disconnect()


def check_mt5_pipeline(cfg: AppConfig, logger):
    masked_pass = "***" if cfg.mt5.password else None
    logger.info(
        "Diagnóstico MT5 | login=%s server=%s path=%s password=%s symbol=%s timeframe=%s",
        cfg.mt5.login,
        cfg.mt5.server,
        cfg.mt5.path,
        masked_pass,
        cfg.mt5.symbol,
        cfg.mt5.timeframe,
    )
    executor = MT5TradeExecutor(
        cfg.mt5,
        cfg.execution,
        stop_loss_pct=cfg.environment.stop_loss_pct,
        take_profit_pct=cfg.environment.take_profit_pct,
    )
    executor.connect()
    try:
        status = executor.check_connection()
        spec = executor.get_symbol_trading_spec()
        risk_state = executor.get_account_risk_state()
        validation_sizing = executor.calculate_order_volume(
            balance=cfg.execution.validation_balance,
            stop_loss_pct=cfg.environment.stop_loss_pct,
            defensive=False,
        )
        status["trading_spec"] = {
            "volume_min": spec.volume_min,
            "volume_step": spec.volume_step,
            "volume_max": spec.volume_max,
            "contract_size": spec.contract_size,
            "point": spec.point,
        }
        status["validation_sizing"] = validation_sizing
        status["account_risk_state"] = risk_state
    finally:
        executor.disconnect()
    logger.info("Status MT5: %s", json.dumps(status, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Traderbot RL + MetaTrader 5")
    parser.add_argument("--config", type=str, default="config.yaml", help="Caminho do arquivo de configuração")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Treina PPO e já executa backtest")
    p_train_multi = sub.add_parser("train-multi", help="Roda múltiplos ciclos de treino/backtest por seed")
    p_train_multi.add_argument(
        "--seeds",
        type=str,
        default="1,7,21,42,84,123,256,512,999,2024",
        help="Lista de seeds separadas por vírgula. Ex: 1,7,21,42,84,123,256,512,999,2024",
    )

    p_backtest = sub.add_parser("backtest", help="Avalia modelo salvo")
    p_backtest.add_argument("--model-path", type=str, default=None, help="Caminho do modelo .zip")

    p_run = sub.add_parser("run", help="Roda loop de inferência e execução (paper/live)")
    p_run.add_argument("--model-path", type=str, default=None, help="Caminho do modelo .zip")
    sub.add_parser("check-mt5", help="Valida conexão MT5 (login/servidor/símbolo)")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    ensure_directories(cfg)

    logger = setup_logger(cfg.app_name, cfg.paths.logs_dir)
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
        run_execution_pipeline(cfg, logger, args.model_path)
    elif args.command == "check-mt5":
        check_mt5_pipeline(cfg, logger)
    else:
        raise ValueError("Comando inválido")


if __name__ == "__main__":
    main()
