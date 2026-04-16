from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize

from traderbot.config import TrainingConfig
from traderbot.rl.callbacks import TrainingProgressCallback


class RLModelManager:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.last_training_summary: dict[str, float] = {}

    def build(self, env: VecEnv) -> PPO:
        """Cria modelo PPO com parâmetros de configuração."""
        model = PPO(
            policy=self.cfg.policy,
            env=env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            ent_coef=self.cfg.ent_coef,
            clip_range=self.cfg.clip_range,
            seed=self.cfg.seed,
            device=self.cfg.device,
            verbose=0,
        )
        return model

    def wrap_env(self, env_fn):
        """Envolve ambiente no formato esperado pelo SB3."""
        num_envs = max(1, int(self.cfg.num_envs))
        env_fns = [env_fn for _ in range(num_envs)]
        use_subproc = num_envs > 1 and os.name != "nt"
        if use_subproc:
            try:
                base_env = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
            except Exception:
                base_env = VecMonitor(DummyVecEnv(env_fns))
        else:
            base_env = VecMonitor(DummyVecEnv(env_fns))
        return VecNormalize(base_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    def train(
        self,
        model: PPO,
        logger=None,
        initial_balance: float = 10_000.0,
        log_trade_events: bool = True,
    ) -> PPO:
        callback = TrainingProgressCallback(
            total_timesteps=self.cfg.total_timesteps,
            log_interval_steps=self.cfg.log_interval_steps,
            initial_balance=initial_balance,
            log_trade_events=log_trade_events,
            logger=logger,
        )
        model.learn(
            total_timesteps=self.cfg.total_timesteps,
            progress_bar=False,
            callback=callback,
        )
        self.last_training_summary = callback.get_training_summary()
        return model

    def save(self, model: PPO, models_dir: str) -> Path:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(models_dir) / f"{self.cfg.model_name}.zip"
        model.save(str(model_path))
        if isinstance(model.get_env(), VecNormalize):
            stats_path = model_path.with_suffix(".vecnormalize.pkl")
            model.get_env().save(str(stats_path))
        return model_path

    @staticmethod
    def load(model_path: str | Path, env=None) -> PPO:
        return PPO.load(str(model_path), env=env)
