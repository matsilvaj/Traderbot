from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class TrainingProgressCallback(BaseCallback):
    """Callback para reportar progresso do treino em intervalos fixos."""

    def __init__(
        self,
        total_timesteps: int,
        log_interval_steps: int,
        initial_balance: float,
        log_trade_events: bool = True,
        logger=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.log_interval_steps = max(1, int(log_interval_steps))
        self.initial_balance = float(initial_balance)
        self.log_trade_events = bool(log_trade_events)
        self.app_logger = logger
        self._start_time: Optional[float] = None
        self._last_log_step = 0
        self._episodes = 0
        self._total_trades = 0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _log(self, msg: str) -> None:
        if self.app_logger is not None:
            self.app_logger.info(msg)
        else:
            print(msg, flush=True)

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._log(f"Treino iniciado: total_timesteps={self.total_timesteps}")

    def _on_step(self) -> bool:
        current = int(self.num_timesteps)
        if (current - self._last_log_step) >= self.log_interval_steps:
            self._last_log_step = current
            elapsed = max(1e-6, time.time() - (self._start_time or time.time()))
            pct = 100.0 * current / self.total_timesteps
            speed = current / elapsed
            remaining_steps = max(0, self.total_timesteps - current)
            eta_seconds = (remaining_steps / speed) if speed > 1e-9 else float("inf")
            eta_text = self._format_duration(eta_seconds) if eta_seconds != float("inf") else "indisponivel"
            finish_text = (
                datetime.fromtimestamp(time.time() + eta_seconds).strftime("%H:%M:%S")
                if eta_seconds != float("inf")
                else "--:--:--"
            )
            infos = self.locals.get("infos")
            trade_count = 0
            if infos and len(infos) > 0:
                trade_count = int((infos[0] or {}).get("trade_count", 0))
            self._log(
                f"Progresso do treino: {current}/{self.total_timesteps} "
                f"({pct:.1f}%) | ~{speed:.1f} steps/s | ETA={eta_text} | fim~{finish_text} | trades={trade_count}"
            )

        infos = self.locals.get("infos")
        if infos and len(infos) > 0:
            info = infos[0] or {}
            final_metrics = info.get("final_metrics")
            if isinstance(final_metrics, dict):
                ep_trades = int(float(final_metrics.get("num_trades", 0.0)))
                self._episodes += 1
                self._total_trades += ep_trades
        return True

    def _on_training_end(self) -> None:
        elapsed = max(1e-6, time.time() - (self._start_time or time.time()))
        self._log(f"Treino finalizado em {elapsed:.1f}s")

    def get_training_summary(self) -> dict[str, float]:
        avg_trades = (self._total_trades / self._episodes) if self._episodes > 0 else 0.0
        return {
            "episodes": float(self._episodes),
            "total_trades": float(self._total_trades),
            "avg_trades_per_episode": float(avg_trades),
        }
