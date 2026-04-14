from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from traderbot.config import HyperliquidConfig, TIMEFRAME_MINUTES_MAP

try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except ImportError:  # pragma: no cover
    Info = None
    constants = None


@dataclass
class HLDataLoader:
    cfg: HyperliquidConfig
    info: Optional[Info] = field(default=None, init=False)

    def _base_url(self) -> str:
        network = str(self.cfg.network).lower().strip()
        if constants is None:
            return ""
        if network == "testnet":
            return constants.TESTNET_API_URL
        return constants.MAINNET_API_URL

    def _timeframe_minutes(self) -> int:
        minutes = TIMEFRAME_MINUTES_MAP.get(str(self.cfg.timeframe).upper())
        if minutes is None:
            raise ValueError(f"Timeframe Hyperliquid não suportado: {self.cfg.timeframe}")
        return int(minutes)

    def connect(self) -> None:
        if Info is None:
            raise RuntimeError(
                "Pacote hyperliquid-python-sdk não encontrado. Instale as dependências do requirements.txt."
            )
        if self.info is None:
            self.info = Info(self._base_url(), skip_ws=True)

    def disconnect(self) -> None:
        if self.info is not None:
            disconnect = getattr(self.info, "disconnect_websocket", None)
            if callable(disconnect):
                try:
                    disconnect()
                except RuntimeError:
                    pass
        self.info = None

    def _ensure_info(self) -> Info:
        if self.info is None:
            self.connect()
        return self.info

    def check_connection(self) -> dict:
        info = self._ensure_info()
        meta = info.meta()
        mids = info.all_mids()
        symbol = str(self.cfg.symbol)
        universe = {item.get("name") for item in meta.get("universe", [])}

        latest_mid = mids.get(symbol)
        return {
            "connected": True,
            "network": str(self.cfg.network),
            "base_url": info.base_url,
            "symbol_requested": symbol,
            "symbol_found": symbol in universe,
            "timeframe": str(self.cfg.timeframe),
            "history_bars": int(self.cfg.history_bars),
            "latest_mid": float(latest_mid) if latest_mid is not None else None,
        }

    def _candles_to_df(self, candles: list[dict]) -> pd.DataFrame:
        if not candles:
            raise RuntimeError(
                f"Sem candles para símbolo '{self.cfg.symbol}' em {self.cfg.timeframe}."
            )

        df = pd.DataFrame(candles)
        out = df.rename(
            columns={
                "t": "time",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        ).copy()
        out["time"] = pd.to_datetime(out["time"], unit="ms", utc=True, errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["spread"] = 0.0
        out["real_volume"] = out["volume"]

        out = out[
            ["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]
        ]
        out = out.sort_values("time").drop_duplicates(subset=["time"]).dropna()
        if out.empty:
            raise RuntimeError(
                f"Candles inválidos após limpeza para símbolo '{self.cfg.symbol}'."
            )
        return out.set_index("time")

    def fetch_historical(self, end_time: Optional[datetime] = None, bars: Optional[int] = None) -> pd.DataFrame:
        info = self._ensure_info()
        end_time = end_time or datetime.now(timezone.utc)
        end_ms = int(end_time.timestamp() * 1000)
        interval_ms = self._timeframe_minutes() * 60_000
        bars = max(int(self.cfg.history_bars if bars is None else bars), 1)
        start_ms = end_ms - (bars * interval_ms)
        candles = info.candles_snapshot(
            str(self.cfg.symbol),
            str(self.cfg.timeframe),
            start_ms,
            end_ms,
        )
        return self._candles_to_df(candles)

    def fetch_latest_bar(self) -> pd.DataFrame:
        info = self._ensure_info()
        now = datetime.now(timezone.utc)
        end_ms = int(now.timestamp() * 1000)
        interval_ms = self._timeframe_minutes() * 60_000
        start_ms = end_ms - (3 * interval_ms)
        candles = info.candles_snapshot(
            str(self.cfg.symbol),
            str(self.cfg.timeframe),
            start_ms,
            end_ms,
        )
        return self._candles_to_df(candles).tail(1)

    def update_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = self.fetch_latest_bar()
        out = pd.concat([df, latest]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out
