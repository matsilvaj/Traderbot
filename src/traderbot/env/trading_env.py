from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from traderbot.config import EnvironmentConfig


@dataclass
class TradeEvent:
    side: str
    entry_price: float
    exit_price: float
    pnl: float


class TradingEnv(gym.Env):
    """Ambiente de trading para scalping BTC M1.

    Ações:
        0: HOLD
        1: BUY (alvo = posição comprada)
        2: SELL (alvo = posição vendida)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        cfg: EnvironmentConfig,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=False).copy()
        self.feature_cols = feature_cols
        self.cfg = cfg
        self.render_mode = render_mode

        if len(self.df) < 100:
            raise ValueError("Dataset muito pequeno para ambiente de RL.")

        self.price_col = "raw_close" if "raw_close" in self.df.columns else "close"
        self.feature_matrix = self.df.loc[:, self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.price_array = self.df.loc[:, self.price_col].to_numpy(dtype=np.float64, copy=True)

        obs_dim = len(self.feature_cols) + 6  # + posição + unrealized + contexto do trade
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.last_equity = self.cfg.initial_balance
        self.balance = self.cfg.initial_balance
        self.equity = self.cfg.initial_balance

        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.entry_step = 0
        self.entry_distance_from_ema_240 = 0.0
        self.trades: list[TradeEvent] = []
        self.trade_count = 0
        self.win_count = 0
        self.blocked_trade_count = 0
        self.equity_curve: list[float] = []
        self.cooldown_steps_remaining = 0
        self.margin_call_triggered = False
        self.last_closed_trade_cost = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_equity = self.cfg.initial_balance
        self.balance = self.cfg.initial_balance
        self.equity = self.cfg.initial_balance

        self.position = 0
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.entry_step = 0
        self.entry_distance_from_ema_240 = 0.0
        self.trades = []
        self.trade_count = 0
        self.win_count = 0
        self.blocked_trade_count = 0
        self.equity_curve = [self.equity]
        self.cooldown_steps_remaining = 0
        self.margin_call_triggered = False
        self.last_closed_trade_cost = 0.0

        return self._get_obs(), {}

    def _price(self, idx: int) -> float:
        safe_idx = min(max(int(idx), 0), len(self.price_array) - 1)
        return float(self.price_array[safe_idx])

    def _trade_units(self, volume: float | None = None) -> float:
        if self.cfg.use_broker_constraints:
            return float((volume if volume is not None else self.position_volume) * self.cfg.broker_contract_size)
        return 1.0

    def _estimate_cost(self, price: float, units: float = 1.0) -> float:
        bps = self.cfg.spread_bps + self.cfg.slippage_bps
        return price * (bps / 10_000.0) * units

    def _unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        units = self._trade_units()
        return (price - self.entry_price) * self.position * units

    def _round_volume_to_step(self, volume: float) -> float:
        step = float(self.cfg.broker_volume_step)
        volume_min = float(self.cfg.broker_volume_min)
        if step <= 0:
            return max(volume, volume_min)
        rounded = np.floor(volume / step) * step
        return max(float(rounded), volume_min)

    def _calculate_trade_volume(self, price: float) -> tuple[float, bool]:
        if self.cfg.use_fixed_trade_volume:
            volume = float(self.cfg.fixed_trade_volume)
            if self.cfg.use_broker_constraints:
                volume = self._round_volume_to_step(volume)
                if float(self.cfg.broker_volume_max) > 0:
                    volume = min(volume, float(self.cfg.broker_volume_max))
            return float(volume), False

        if not self.cfg.use_broker_constraints:
            return 1.0, False

        balance = float(self.balance)
        risk_amount = balance * float(self.cfg.risk_per_trade)
        stop_distance_price = price * float(self.cfg.stop_loss_pct)
        contract_size = max(float(self.cfg.broker_contract_size), 1e-12)
        loss_per_lot = stop_distance_price * contract_size
        raw_volume = 0.0 if loss_per_lot <= 0 else (risk_amount / loss_per_lot)

        if raw_volume < float(self.cfg.broker_volume_min) and self.cfg.block_trade_on_excess_risk:
            return 0.0, True

        volume = self._round_volume_to_step(raw_volume)
        if float(self.cfg.broker_volume_max) > 0:
            volume = min(volume, float(self.cfg.broker_volume_max))
        return float(volume), False

    def _close_position(self, price: float) -> float:
        if self.position == 0:
            return 0.0

        units = self._trade_units()
        cost = self._estimate_cost(price, units=units)
        pnl = (price - self.entry_price) * self.position * units - cost
        self.balance += pnl
        side = "LONG" if self.position == 1 else "SHORT"
        self.last_closed_trade_cost = cost

        self.trades.append(
            TradeEvent(
                side=side,
                entry_price=self.entry_price,
                exit_price=price,
                pnl=pnl,
            )
        )
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1

        self.position = 0
        self.entry_price = 0.0
        self.position_volume = 0.0
        self.entry_step = 0
        self.entry_distance_from_ema_240 = 0.0
        self.cooldown_steps_remaining = max(0, int(self.cfg.trade_cooldown_steps))
        return pnl

    def _open_position(self, target_position: int, price: float) -> tuple[float, bool]:
        if target_position == 0:
            return 0.0, False

        if self.cooldown_steps_remaining > 0:
            self.blocked_trade_count += 1
            return 0.0, False

        volume, blocked = self._calculate_trade_volume(price)
        if blocked or volume <= 0:
            self.blocked_trade_count += 1
            return 0.0, False

        units = self._trade_units(volume=volume)
        cost = self._estimate_cost(price, units=units)
        self.balance -= cost
        self.position = target_position
        self.entry_price = price
        self.position_volume = volume
        self.entry_step = int(self.current_step)
        if "dist_ema_240" in self.df.columns:
            self.entry_distance_from_ema_240 = float(self.df.loc[self.current_step, "dist_ema_240"])
        else:
            self.entry_distance_from_ema_240 = 0.0
        return -cost, True

    def _apply_action(self, action: int, price: float) -> tuple[float, int]:
        # mapeamento de ação para posição alvo
        target_position = {0: self.position, 1: 1, 2: -1}[action]

        realized_pnl = 0.0
        position_changes = 0

        if target_position != self.position:
            # Fecha posição existente se necessário
            if self.position != 0:
                realized_pnl += self._close_position(price)
                position_changes += 1

            # Abre nova posição
            if target_position != 0:
                open_pnl, opened = self._open_position(target_position, price)
                realized_pnl += open_pnl
                if opened:
                    position_changes += 1

        return realized_pnl, position_changes

    def _check_stop_take(self, price: float) -> tuple[float, int]:
        """Aplica stop-loss/take-profit com base no preço atual."""
        if self.position == 0:
            return 0.0, 0

        move = ((price - self.entry_price) / (self.entry_price + 1e-12)) * self.position
        if move <= -self.cfg.stop_loss_pct:
            return self._close_position(price), 1
        if move >= self.cfg.take_profit_pct:
            return self._close_position(price), 1
        return 0.0, 0

    def _get_obs(self) -> np.ndarray:
        safe_idx = min(max(int(self.current_step), 0), len(self.feature_matrix) - 1)
        row = self.feature_matrix[safe_idx]
        current_price = self._price(safe_idx)
        unrealized = self._unrealized_pnl(current_price)
        time_in_position = float(max(0, safe_idx - self.entry_step)) if self.position != 0 else 0.0
        if self.position != 0 and self.entry_price > 0:
            stop_move = ((current_price - (self.entry_price * (1.0 - self.cfg.stop_loss_pct * self.position))) / self.entry_price) * self.position
            take_move = (((self.entry_price * (1.0 + self.cfg.take_profit_pct * self.position)) - current_price) / self.entry_price) * self.position
            distance_to_stop = float(stop_move)
            distance_to_take = float(take_move)
        else:
            distance_to_stop = 0.0
            distance_to_take = 0.0
        extras = np.array(
            [
                float(self.position),
                float(unrealized),
                float(time_in_position),
                float(distance_to_stop),
                float(distance_to_take),
                float(self.entry_distance_from_ema_240),
            ],
            dtype=np.float32,
        )
        return np.concatenate([row, extras], axis=0)

    def step(self, action: int):
        current_price = self._price(self.current_step)
        starting_cooldown = self.cooldown_steps_remaining
        realized_pnl, position_changes = self._check_stop_take(current_price)
        action_pnl, action_changes = self._apply_action(action, current_price)
        realized_pnl += action_pnl
        position_changes += action_changes

        unrealized = self._unrealized_pnl(current_price)
        self.equity = self.balance + unrealized

        reward = self.equity - self.last_equity
        reward -= float(self.cfg.overtrade_penalty) * position_changes

        pain_penalty = float(
            np.clip(
                min(0.0, unrealized) * float(self.cfg.pain_decay_factor),
                -float(self.cfg.reward_clip_limit),
                0.0,
            )
        )
        reward += pain_penalty

        margin_call = False
        margin_call_penalty_applied = 0.0
        margin_call_equity = self.cfg.initial_balance * (1.0 - float(self.cfg.margin_call_drawdown_pct))
        if self.equity <= margin_call_equity:
            self.margin_call_triggered = True
            margin_call = True
            if self.position != 0:
                forced_close_pnl = self._close_position(current_price)
                realized_pnl += forced_close_pnl
                close_adjustment = self.balance - self.equity
                self.equity = self.balance
                reward += close_adjustment
            margin_call_penalty_applied = float(self.cfg.margin_call_penalty)
            reward -= margin_call_penalty_applied

        reward = float(
            np.clip(
                reward,
                -float(self.cfg.reward_clip_limit),
                float(self.cfg.reward_clip_limit),
            )
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 or margin_call
        if self.position == 0 and self.cooldown_steps_remaining == starting_cooldown and self.cooldown_steps_remaining > 0:
            self.cooldown_steps_remaining -= 1

        if self.cfg.max_episode_steps is not None:
            terminated = terminated or (self.current_step >= self.cfg.max_episode_steps)

        if terminated and not margin_call and self.position != 0:
            final_price = self._price(self.current_step)
            pre_close_equity = self.equity
            terminal_close_pnl = self._close_position(final_price)
            realized_pnl += terminal_close_pnl
            self.equity = self.balance
            reward += self.balance - pre_close_equity
            reward = float(
                np.clip(
                    reward,
                    -float(self.cfg.reward_clip_limit),
                    float(self.cfg.reward_clip_limit),
                )
            )

        self.last_equity = self.equity
        self.equity_curve.append(self.equity)

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = self._get_info_dict(realized_pnl, margin_call_penalty_applied)
        if terminated:
            info["final_metrics"] = self.get_metrics()

        return obs, float(reward), terminated, False, info

    def _get_info_dict(self, realized_pnl_step: float, margin_call_penalty_applied: float = 0.0) -> dict:
        """Função auxiliar para limpar o dict de info"""
        info = {
            "equity": self.equity,
            "balance": self.balance,
            "position": self.position,
            "trade_count": self.trade_count,
            "blocked_trade_count": self.blocked_trade_count,
            "win_count": self.win_count,
            "realized_pnl_step": float(realized_pnl_step),
            "cooldown_steps_remaining": int(self.cooldown_steps_remaining),
            "margin_call_triggered": bool(self.margin_call_triggered),
            "margin_call_penalty_applied": float(margin_call_penalty_applied),
        }
        if self.trades:
            t = self.trades[-1]
            info["last_trade"] = {
                "side": t.side,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "pnl": float(t.pnl),
            }
        return info

    def get_metrics(self) -> dict[str, float]:
        equity = np.array(self.equity_curve, dtype=float)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / (peak + 1e-12)
        max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0
        blocked_trade_rate = (
            float(self.blocked_trade_count / (self.blocked_trade_count + self.trade_count))
            if (self.blocked_trade_count + self.trade_count) > 0
            else 0.0
        )

        total_profit = float(self.equity - self.cfg.initial_balance)
        win_rate = float(self.win_count / self.trade_count) if self.trade_count > 0 else 0.0
        profit_per_trade = float(total_profit / self.trade_count) if self.trade_count > 0 else 0.0

        return {
            "total_profit": total_profit,
            "max_drawdown": max_drawdown,
            "num_trades": float(self.trade_count),
            "blocked_trades": float(self.blocked_trade_count),
            "blocked_trade_rate": blocked_trade_rate,
            "win_rate": win_rate,
            "profit_per_trade": profit_per_trade,
            "final_equity": float(self.equity),
        }

    def render(self):
        if self.render_mode == "human":
            print(
                f"step={self.current_step} equity={self.equity:.2f} "
                f"balance={self.balance:.2f} position={self.position} trades={self.trade_count}"
            )
