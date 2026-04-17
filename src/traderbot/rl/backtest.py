from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from traderbot.config import EnvironmentConfig
from traderbot.data.database import save_session_metrics
from traderbot.env.trading_env import TradingEnv


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def _log_progress(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)
    else:
        print(message, flush=True)


def _maybe_log_backtest_progress(
    *,
    logger,
    start_time: float,
    step_index: int,
    total_steps: int,
    trade_count: int,
    log_interval_steps: int,
    label: str,
) -> None:
    if step_index <= 0 or (step_index % log_interval_steps) != 0:
        return

    elapsed = max(1e-6, time.time() - start_time)
    speed = step_index / elapsed
    pct = 100.0 * step_index / max(1, total_steps)
    remaining_steps = max(0, total_steps - step_index)
    eta_seconds = remaining_steps / max(speed, 1e-6)
    _log_progress(
        logger,
        f"Progresso do {label}: {step_index}/{total_steps} ({pct:.1f}%) | "
        f"~{speed:.1f} steps/s | ETA ~{eta_seconds / 60.0:.1f} min | trades={trade_count}",
    )


def _build_backtest_result(env: TradingEnv, metrics: dict[str, float]) -> BacktestResult:
    trades = pd.DataFrame(
        [
            {
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "duration_bars": t.duration_bars,
            }
            for t in env.trades
        ]
    )
    equity_curve = pd.DataFrame({"equity": env.equity_curve})
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def _load_obs_normalizer(
    df: pd.DataFrame,
    feature_cols: list[str],
    env_cfg: EnvironmentConfig,
    vecnormalize_path: str | Path | None = None,
):
    if vecnormalize_path is None or not Path(vecnormalize_path).exists():
        return None
    base_env = DummyVecEnv([lambda: TradingEnv(df=df, feature_cols=feature_cols, cfg=env_cfg)])
    vec_env = VecNormalize.load(str(vecnormalize_path), base_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _normalize_obs(obs, obs_normalizer):
    if obs_normalizer is None:
        return obs
    return obs_normalizer.normalize_obs(obs.reshape(1, -1))[0]


def _coerce_action_array(action) -> Any:
    return np.asarray(action, dtype=np.float32).reshape(1)


def _action_bucket(action, hold_threshold: float) -> int:
    raw_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    if raw_action > hold_threshold:
        return 1
    if raw_action < -hold_threshold:
        return -1
    return 0


def _majority_vote_bucket(raw_actions: list[float], hold_threshold: float) -> tuple[int, dict[int, int]]:
    buckets = [_action_bucket(action, hold_threshold) for action in raw_actions]
    counts = {bucket: buckets.count(bucket) for bucket in (-1, 0, 1)}
    max_votes = max(counts.values()) if counts else 0
    winners = [bucket for bucket, votes in counts.items() if votes == max_votes]
    if len(winners) != 1:
        return 0, counts
    return winners[0], counts


def run_backtest(
    model: PPO,
    df: pd.DataFrame,
    feature_cols: list[str],
    env_cfg: EnvironmentConfig,
    vecnormalize_path: str | Path | None = None,
    deterministic: bool = True,
    logger=None,
    log_interval_steps: int = 10_000,
) -> BacktestResult:
    """Executa avaliação fora do treino em dados holdout."""
    env = TradingEnv(df=df, feature_cols=feature_cols, cfg=env_cfg)
    obs_normalizer = _load_obs_normalizer(df, feature_cols, env_cfg, vecnormalize_path=vecnormalize_path)
    try:
        obs, _ = env.reset()
        total_steps = max(1, len(df) - 1)
        start_time = time.time()

        done = False
        while not done:
            action, _ = model.predict(_normalize_obs(obs, obs_normalizer), deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(_coerce_action_array(action))
            done = terminated or truncated
            _maybe_log_backtest_progress(
                logger=logger,
                start_time=start_time,
                step_index=env.current_step,
                total_steps=total_steps,
                trade_count=env.trade_count,
                log_interval_steps=log_interval_steps,
                label="backtest",
            )

        metrics = env.get_metrics()
        metrics["decision_mode"] = "single"
        metrics["backtest_elapsed_seconds"] = float(time.time() - start_time)
        return _build_backtest_result(env, metrics)
    finally:
        if obs_normalizer is not None:
            obs_normalizer.close()


def run_backtest_ensemble(
    models: Sequence[PPO | tuple[PPO, str | Path | None]],
    df: pd.DataFrame,
    feature_cols: list[str],
    env_cfg: EnvironmentConfig,
    vecnormalize_path: str | Path | None = None,
    deterministic: bool = True,
    logger=None,
    log_interval_steps: int = 10_000,
) -> BacktestResult:
    """Executa backtest com votação majoritária entre múltiplos modelos."""
    if not models:
        raise ValueError("Ensemble vazio para backtest.")

    env = TradingEnv(df=df, feature_cols=feature_cols, cfg=env_cfg)
    default_normalizer = _load_obs_normalizer(df, feature_cols, env_cfg, vecnormalize_path=vecnormalize_path)
    model_entries: list[tuple[PPO, Any]] = []
    for item in models:
        if isinstance(item, tuple):
            model_obj, stats_path = item
            obs_normalizer = _load_obs_normalizer(df, feature_cols, env_cfg, vecnormalize_path=stats_path)
            model_entries.append((model_obj, obs_normalizer))
        else:
            model_entries.append((item, default_normalizer))
    try:
        obs, _ = env.reset()
        total_steps = max(1, len(df) - 1)
        start_time = time.time()
        hold_threshold = float(env_cfg.action_hold_threshold)

        vote_totals = {-1: 0, 0: 0, 1: 0}
        tie_steps = 0
        done = False
        while not done:
            raw_actions = [
                float(
                    np.asarray(
                        model.predict(_normalize_obs(obs, obs_normalizer), deterministic=deterministic)[0],
                        dtype=np.float32,
                    ).reshape(-1)[0]
                )
                for model, obs_normalizer in model_entries
            ]
            winner_bucket, counts = _majority_vote_bucket(raw_actions, hold_threshold)
            for bucket, count in counts.items():
                vote_totals[bucket] += count

            if winner_bucket == 1:
                winner_actions = [abs(raw_action) for raw_action in raw_actions if _action_bucket(raw_action, hold_threshold) == 1]
                action_value = float(np.mean(winner_actions)) if winner_actions else float(hold_threshold)
                action = np.array([float(np.clip(max(action_value, hold_threshold), hold_threshold, 1.0))], dtype=np.float32)
            elif winner_bucket == -1:
                winner_actions = [abs(raw_action) for raw_action in raw_actions if _action_bucket(raw_action, hold_threshold) == -1]
                action_value = float(np.mean(winner_actions)) if winner_actions else float(hold_threshold)
                action = np.array([float(np.clip(-max(action_value, hold_threshold), -1.0, -hold_threshold))], dtype=np.float32)
            else:
                action = np.array([0.0], dtype=np.float32)

            if list(counts.values()).count(max(counts.values())) > 1:
                tie_steps += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            _maybe_log_backtest_progress(
                logger=logger,
                start_time=start_time,
                step_index=env.current_step,
                total_steps=total_steps,
                trade_count=env.trade_count,
                log_interval_steps=log_interval_steps,
                label="backtest ensemble",
            )

        metrics = env.get_metrics()
        metrics["decision_mode"] = "ensemble_majority_vote"
        metrics["ensemble_size"] = float(len(models))
        metrics["vote_hold_total"] = float(vote_totals[0])
        metrics["vote_buy_total"] = float(vote_totals[1])
        metrics["vote_sell_total"] = float(vote_totals[-1])
        metrics["tie_steps"] = float(tie_steps)
        metrics["backtest_elapsed_seconds"] = float(time.time() - start_time)
        return _build_backtest_result(env, metrics)
    finally:
        closed_ids: set[int] = set()
        if default_normalizer is not None:
            default_normalizer.close()
            closed_ids.add(id(default_normalizer))
        for _, obs_normalizer in model_entries:
            if obs_normalizer is not None and id(obs_normalizer) not in closed_ids:
                obs_normalizer.close()
                closed_ids.add(id(obs_normalizer))


def save_backtest_result(result: BacktestResult, results_dir: str, prefix: str) -> dict[str, str]:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    trades_path = Path(results_dir) / f"{prefix}_trades.csv"
    equity_path = Path(results_dir) / f"{prefix}_equity.csv"
    save_session_metrics(
        session_id=prefix,
        total_profit=float(result.metrics.get("total_profit", 0.0)),
        win_rate=float(result.metrics.get("win_rate", 0.0)),
        max_drawdown=float(result.metrics.get("max_drawdown", 0.0)),
    )

    result.trades.to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)

    return {
        "metrics_session_id": prefix,
        "trades": str(trades_path),
        "equity": str(equity_path),
    }


def print_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"Lucro total: {metrics['total_profit']:.2f} | "
        f"Drawdown máximo: {metrics['max_drawdown']:.2%} | "
        f"Trades: {int(metrics['num_trades'])} | "
        f"Bloqueados: {int(metrics.get('blocked_trades', 0))} | "
        f"Regime ok: {metrics.get('pct_bars_with_valid_regime', 0.0):.2%} | "
        f"Regime filtrado: {metrics.get('pct_bars_filtered_by_regime', 0.0):.2%} | "
        f"Bloq.regime: {int(metrics.get('blocked_by_regime_filter', 0))} | "
        f"Profit/trade: {metrics.get('profit_per_trade', 0.0):.4f} | "
        f"Taxa de acerto: {metrics['win_rate']:.2%} | "
        f"PF: {metrics.get('profit_factor', 0.0):.2f} | "
        f"TP/SL/Agent: {int(metrics.get('exit_by_take_profit', 0))}/"
        f"{int(metrics.get('exit_by_stop_loss', 0))}/"
        f"{int(metrics.get('exit_by_agent_close', 0))} | "
        f"Rev. evitadas: {int(metrics.get('prevented_same_candle_reversals', 0))}"
    )
