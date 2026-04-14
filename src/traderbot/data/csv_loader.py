from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traderbot.config import DataConfig


@dataclass
class CSVDataLoader:
    cfg: DataConfig

    def load(self) -> pd.DataFrame:
        csv_path = Path(self.cfg.csv_path or "")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de mercado não encontrado em {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"CSV vazio em {csv_path}")

        timestamp_col = self.cfg.timestamp_column
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Coluna de timestamp '{timestamp_col}' não encontrada no CSV {csv_path}"
            )

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

        out = df.rename(columns={timestamp_col: "time"}).copy()
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.dropna(subset=["time"])

        for col in required_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        optional_numeric = [col for col in ["spread", "real_volume"] if col in out.columns]
        for col in optional_numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        if "spread" not in out.columns:
            out["spread"] = 0.0
        if "real_volume" not in out.columns:
            out["real_volume"] = out["volume"]

        out = out[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]
        out = out.sort_values("time").drop_duplicates(subset=["time"]).dropna()
        if out.empty:
            raise ValueError(f"CSV sem linhas válidas após limpeza: {csv_path}")

        return out.set_index("time")
