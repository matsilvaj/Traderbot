from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from traderbot.config import MT5Config

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None


MT5_TIMEFRAME_CONST = {
    "M1": "TIMEFRAME_M1",
    "M2": "TIMEFRAME_M2",
    "M3": "TIMEFRAME_M3",
    "M4": "TIMEFRAME_M4",
    "M5": "TIMEFRAME_M5",
    "M10": "TIMEFRAME_M10",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
}


@dataclass
class MT5DataLoader:
    cfg: MT5Config
    _resolved_symbol: Optional[str] = None

    def _initialize_terminal(self) -> None:
        """Inicializa terminal com fallback (com path e sem path)."""
        attempts: list[tuple[str, tuple[int, str]]] = []
        auth_kwargs = {
            "login": self.cfg.login,
            "password": self.cfg.password,
            "server": self.cfg.server,
            "timeout": 60_000,
        }

        # Tentativa 1: com path + credenciais
        if self.cfg.path:
            ok = mt5.initialize(path=self.cfg.path, **auth_kwargs)
            if ok:
                return
            attempts.append(
                (
                    f"initialize(path={self.cfg.path}, login=***, server={self.cfg.server})",
                    mt5.last_error(),
                )
            )
            mt5.shutdown()

        # Tentativa 2: sem path + credenciais
        ok = mt5.initialize(**auth_kwargs)
        if ok:
            return
        attempts.append(
            (
                f"initialize(login=***, server={self.cfg.server})",
                mt5.last_error(),
            )
        )
        mt5.shutdown()

        # Tentativa 3: com path explícito (se informado), sem credenciais
        if self.cfg.path:
            ok = mt5.initialize(path=self.cfg.path)
            if ok:
                return
            attempts.append((f"initialize(path={self.cfg.path})", mt5.last_error()))
            mt5.shutdown()

        # Tentativa 4: terminal padrão do sistema, sem credenciais
        ok = mt5.initialize()
        if ok:
            return
        attempts.append(("initialize()", mt5.last_error()))
        mt5.shutdown()

        detail = " | ".join([f"{name} -> {err}" for name, err in attempts])
        raise RuntimeError(f"Falha ao inicializar MT5. Tentativas: {detail}")

    def connect(self) -> None:
        """Inicializa conexão com o terminal do MetaTrader 5."""
        if mt5 is None:
            raise RuntimeError(
                "Pacote MetaTrader5 não encontrado. Instale dependências em requirements.txt"
            )

        if not (self.cfg.login and self.cfg.password and self.cfg.server):
            raise RuntimeError(
                "Credenciais MT5 incompletas. Preencha MT5_LOGIN, MT5_PASSWORD e MT5_SERVER no .env."
            )

        self._initialize_terminal()

        # Login explícito para garantir sessão correta, mesmo quando initialize já autenticou.
        logged = mt5.login(login=self.cfg.login, password=self.cfg.password, server=self.cfg.server)
        if not logged and mt5.last_error()[0] not in (0, 1):
            raise RuntimeError(
                "Falha no login MT5: "
                f"{mt5.last_error()}. "
                "Confirme login/senha, servidor exato e se a conta permite conexão via terminal."
            )

    def check_connection(self) -> dict:
        """Valida conexão/login/símbolo e retorna diagnóstico resumido."""
        self.connect()
        try:
            account = mt5.account_info()
            terminal = mt5.terminal_info()
            resolved_symbol = self._resolve_symbol()
            symbol_info = mt5.symbol_info(resolved_symbol)
            symbol_ok = mt5.symbol_select(resolved_symbol, True)

            return {
                "connected": True,
                "account_login": getattr(account, "login", None) if account else None,
                "account_server": getattr(account, "server", None) if account else None,
                "terminal_name": getattr(terminal, "name", None) if terminal else None,
                "terminal_company": getattr(terminal, "company", None) if terminal else None,
                "symbol_requested": self.cfg.symbol,
                "symbol_resolved": resolved_symbol,
                "symbol_found": symbol_info is not None,
                "symbol_selected": bool(symbol_ok),
            }
        finally:
            self.disconnect()

    def disconnect(self) -> None:
        """Finaliza conexão com o MT5."""
        if mt5 is not None:
            mt5.shutdown()

    def _resolve_timeframe(self) -> int:
        key = MT5_TIMEFRAME_CONST.get(self.cfg.timeframe.upper())
        if key is None:
            raise ValueError(f"Timeframe não suportado: {self.cfg.timeframe}")
        return getattr(mt5, key)

    def _resolve_symbol(self) -> str:
        """Resolve símbolo com fallback para sufixos/prefixos do broker."""
        if self._resolved_symbol:
            return self._resolved_symbol

        requested = self.cfg.symbol
        info = mt5.symbol_info(requested)
        if info is not None:
            mt5.symbol_select(requested, True)
            self._resolved_symbol = requested
            return requested

        all_symbols = mt5.symbols_get()
        if not all_symbols:
            raise RuntimeError("Não foi possível listar símbolos no MT5.")

        names = [s.name for s in all_symbols]
        req_norm = requested.lower().replace("/", "").replace("-", "").replace("_", "")

        exact_ci = [n for n in names if n.lower() == requested.lower()]
        starts = [n for n in names if n.lower().startswith(req_norm)]
        contains = [n for n in names if req_norm in n.lower().replace("/", "").replace("-", "").replace("_", "")]
        btc_like = [n for n in names if "btc" in n.lower() and "usd" in n.lower()]

        candidates = exact_ci or starts or contains or btc_like
        if not candidates:
            sample = ", ".join(names[:20])
            raise RuntimeError(
                f"Símbolo '{requested}' não encontrado. "
                f"Exemplos disponíveis: {sample}"
            )

        chosen = candidates[0]
        if not mt5.symbol_select(chosen, True):
            raise RuntimeError(
                f"Símbolo resolvido '{chosen}', mas não foi possível selecioná-lo no Market Watch."
            )

        self._resolved_symbol = chosen
        return chosen

    def fetch_historical(self, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Coleta candles históricos e retorna DataFrame estruturado."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 indisponível no ambiente.")

        timeframe = self._resolve_timeframe()
        symbol = self._resolve_symbol()
        end_time = end_time or datetime.utcnow()

        rates = mt5.copy_rates_from(symbol, timeframe, end_time, self.cfg.history_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Sem dados para símbolo '{symbol}' (solicitado: '{self.cfg.symbol}') "
                f"em {self.cfg.timeframe}. Verifique se o ativo está habilitado no Market Watch."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]
        df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
        return df

    def fetch_latest_bar(self) -> pd.DataFrame:
        """Busca o candle mais recente (1 barra)."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 indisponível no ambiente.")

        timeframe = self._resolve_timeframe()
        symbol = self._resolve_symbol()
        now = datetime.utcnow() + timedelta(seconds=5)
        rates = mt5.copy_rates_from(symbol, timeframe, now, 2)
        if rates is None or len(rates) == 0:
            raise RuntimeError("Não foi possível obter a barra mais recente.")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]
        df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
        return df.tail(1)

    def update_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Atualiza DataFrame histórico com a barra mais recente."""
        latest = self.fetch_latest_bar()
        out = pd.concat([df, latest]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out
