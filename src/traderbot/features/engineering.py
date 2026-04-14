from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from traderbot.config import FeatureConfig


@dataclass
class FeatureEngineer:
    cfg: FeatureConfig
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features sem olhar o futuro (apenas histórico disponível)."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame sem colunas obrigatórias para features: {missing}")

        out = df.loc[:, list(dict.fromkeys(list(df.columns) + required_cols))].copy()
        if not isinstance(out.index, pd.DatetimeIndex) and "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
            out = out.dropna(subset=["time"]).set_index("time")

        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        for col in required_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["raw_open"] = out["open"]
        out["raw_high"] = out["high"]
        out["raw_low"] = out["low"]
        out["raw_close"] = out["close"]
        out["dollar_volume"] = out["close"] * out["volume"]

        out["return_1"] = out["close"].pct_change(1)
        out["return_3"] = out["close"].pct_change(3)
        out["return_5"] = out["close"].pct_change(5)
        out["return_12"] = out["close"].pct_change(12)
        out["return_24"] = out["close"].pct_change(24)
        out["momentum"] = out["close"].pct_change(self.cfg.momentum_period)
        out["momentum_12"] = out["close"] - out["close"].shift(12)
        out["momentum_24"] = out["close"] - out["close"].shift(24)
        out["rsi"] = self._rsi(out["close"], self.cfg.rsi_period)
        out["rsi_14"] = self._rsi(out["close"], 14)
        out["ma_short"] = out["close"].rolling(self.cfg.ma_short_period).mean()
        out["ma_long"] = out["close"].rolling(self.cfg.ma_long_period).mean()
        out["ema_240"] = out["close"].ewm(span=240, adjust=False).mean()
        out["ema_960"] = out["close"].ewm(span=960, adjust=False).mean()
        out["slope_ema_240"] = out["ema_240"].pct_change(1)
        out["slope_ema_960"] = out["ema_960"].pct_change(1)
        out["ma_ratio"] = out["ma_short"] / (out["ma_long"] + 1e-12)
        out["dist_ema_240"] = (out["close"] - out["ema_240"]) / (out["ema_240"] + 1e-12)
        out["dist_ema_960"] = (out["close"] - out["ema_960"]) / (out["ema_960"] + 1e-12)
        out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)
        out["range_pct"] = (out["high"] - out["low"]) / (out["close"].shift(1) + 1e-12)
        out["trend_spread"] = (out["ma_short"] - out["ma_long"]) / (out["close"] + 1e-12)
        out["trend_slope"] = out["ma_long"].pct_change(self.cfg.momentum_period)
        out["price_vs_ma_short"] = (out["close"] - out["ma_short"]) / (out["close"] + 1e-12)
        out["price_vs_ma_long"] = (out["close"] - out["ma_long"]) / (out["close"] + 1e-12)
        out["volatility_10"] = out["return_1"].rolling(10).std()
        out["volatility_12"] = out["return_1"].rolling(12).std()
        out["volatility_24"] = out["return_1"].rolling(24).std()
        out["body_ratio"] = (out["close"] - out["open"]).abs() / (
            (out["high"] - out["low"]).abs() + 1e-12
        )
        candle_range = (out["high"] - out["low"]).replace(0.0, np.nan)
        body_high = out[["open", "close"]].max(axis=1)
        body_low = out[["open", "close"]].min(axis=1)
        out["body_pct"] = (out["close"] - out["open"]).abs() / (candle_range + 1e-12)
        out["upper_wick_pct"] = (out["high"] - body_high) / (candle_range + 1e-12)
        out["lower_wick_pct"] = (body_low - out["low"]) / (candle_range + 1e-12)
        out["close_location"] = ((out["close"] - out["low"]) / (candle_range + 1e-12)).clip(0.0, 1.0)
        out["gap_pct"] = (out["open"] - out["close"].shift(1)) / (out["close"].shift(1) + 1e-12)
        out["higher_high"] = (out["high"] > out["high"].shift(1)).astype(float)
        out["lower_low"] = (out["low"] < out["low"].shift(1)).astype(float)
        out["higher_low"] = (out["low"] > out["low"].shift(1)).astype(float)
        out["lower_high"] = (out["high"] < out["high"].shift(1)).astype(float)
        out["trend_strength_20"] = (
            (out["close"] - out["close"].rolling(20).mean()) / (out["close"] + 1e-12)
        )
        out["range_compression_10"] = out["hl_range"].rolling(10).mean()
        out["range_expansion_3"] = out["hl_range"] / (out["hl_range"].rolling(3).mean() + 1e-12)
        out["breakout_up_10"] = (
            (out["close"] - out["high"].shift(1).rolling(10).max()) / (out["close"] + 1e-12)
        )
        out["breakout_down_10"] = (
            (out["close"] - out["low"].shift(1).rolling(10).min()) / (out["close"] + 1e-12)
        )
        out["up_candle"] = (out["close"] > out["open"]).astype(float)
        out["down_candle"] = (out["close"] < out["open"]).astype(float)
        out["up_streak_3"] = out["up_candle"].rolling(3).sum() / 3.0
        out["down_streak_3"] = out["down_candle"].rolling(3).sum() / 3.0

        # Volatilidade e Regime (ATR aproximado)
        out["prev_close"] = out["close"].shift(1)
        out["tr1"] = out["high"] - out["low"]
        out["tr2"] = (out["high"] - out["prev_close"]).abs()
        out["tr3"] = (out["low"] - out["prev_close"]).abs()
        out["true_range"] = out[["tr1", "tr2", "tr3"]].max(axis=1)
        out["atr_14"] = out["true_range"].rolling(14).mean() / (out["close"] + 1e-12)
        out["atr_pct"] = out["atr_14"]

        bb_mid = out["close"].rolling(20).mean()
        bb_std = out["close"].rolling(20).std()
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        out["bollinger_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-12)
        out["distance_bb_mid"] = (out["close"] - bb_mid) / (bb_mid + 1e-12)
        out["zscore_close_20"] = (out["close"] - bb_mid) / (bb_std + 1e-12)

        rolling_vol_mean = out["volatility_24"].rolling(50).mean()
        rolling_vol_std = out["volatility_24"].rolling(50).std()
        out["vol_regime_z"] = (out["volatility_24"] - rolling_vol_mean) / (rolling_vol_std + 1e-12)

        volume_mean_20 = out["volume"].rolling(20).mean()
        volume_std_20 = out["volume"].rolling(20).std()
        out["volume_zscore_20"] = (out["volume"] - volume_mean_20) / (volume_std_20 + 1e-12)
        out["volume_ratio_20"] = out["volume"] / (volume_mean_20 + 1e-12)

        dollar_volume_mean_20 = out["dollar_volume"].rolling(20).mean()
        dollar_volume_std_20 = out["dollar_volume"].rolling(20).std()
        out["dollar_volume_zscore_20"] = (
            (out["dollar_volume"] - dollar_volume_mean_20) / (dollar_volume_std_20 + 1e-12)
        )

        lowest_low_14 = out["low"].rolling(14).min()
        highest_high_14 = out["high"].rolling(14).max()
        out["stoch_k_14"] = (
            (out["close"] - lowest_low_14) / ((highest_high_14 - lowest_low_14) + 1e-12)
        )

        out = out.drop(columns=["prev_close", "tr1", "tr2", "tr3", "true_range"])

        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        if out.empty:
            raise ValueError("Sem linhas válidas após cálculo das features e dropna.")
        return out

    def fit_normalizer(self, train_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Ajusta normalização apenas no conjunto de treino."""
        self.means_ = {col: float(train_df[col].mean()) for col in feature_cols}
        self.stds_ = {col: float(train_df[col].std() + 1e-8) for col in feature_cols}

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        if not self.cfg.normalize:
            return out

        if not self.means_ or not self.stds_:
            raise RuntimeError("Normalizer não ajustado. Rode fit_normalizer antes de transform.")

        for col in feature_cols:
            out[col] = (out[col] - self.means_[col]) / max(self.stds_[col], 1e-8)

        return out


def default_feature_columns() -> list[str]:
    cols = [
        "return_1",
        "return_3",
        "return_5",
        "return_12",
        "return_24",
        "rsi_14",
        "dist_ema_240",
        "dist_ema_960",
        "atr_pct",
        "volatility_12",
        "volatility_24",
        "vol_regime_z",
        "range_pct",
        "range_compression_10",
        "range_expansion_3",
        "breakout_up_10",
        "breakout_down_10",
        "bollinger_width",
        "volume_zscore_20",
        "volume_ratio_20",
    ]
    return list(dict.fromkeys(cols))
