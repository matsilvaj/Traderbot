from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from typing import Optional

import pandas as pd

from traderbot.config import ExecutionConfig, MT5Config

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
class ExecutionDecision:
    action: int
    reason: str = ""


@dataclass
class SymbolTradingSpec:
    symbol: str
    volume_min: float
    volume_step: float
    volume_max: float
    contract_size: float
    point: float


class MT5TradeExecutor:
    """Executor de ordens com proteção para não operar live sem autorização explícita."""

    def __init__(
        self,
        mt5_cfg: MT5Config,
        exec_cfg: ExecutionConfig,
        stop_loss_pct: float = 0.003,
        take_profit_pct: float = 0.006,
    ):
        self.mt5_cfg = mt5_cfg
        self.exec_cfg = exec_cfg
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.last_order_ts = 0.0
        self._resolved_symbol: Optional[str] = None
        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0

    def _initialize_terminal(self) -> None:
        attempts: list[tuple[str, tuple[int, str]]] = []
        auth_kwargs = {
            "login": self.mt5_cfg.login,
            "password": self.mt5_cfg.password,
            "server": self.mt5_cfg.server,
            "timeout": 60_000,
        }

        if self.mt5_cfg.path:
            ok = mt5.initialize(path=self.mt5_cfg.path, **auth_kwargs)
            if ok:
                return
            attempts.append(
                (
                    f"initialize(path={self.mt5_cfg.path}, login=***, server={self.mt5_cfg.server})",
                    mt5.last_error(),
                )
            )
            mt5.shutdown()

        ok = mt5.initialize(**auth_kwargs)
        if ok:
            return
        attempts.append(
            (
                f"initialize(login=***, server={self.mt5_cfg.server})",
                mt5.last_error(),
            )
        )
        mt5.shutdown()

        if self.mt5_cfg.path:
            ok = mt5.initialize(path=self.mt5_cfg.path)
            if ok:
                return
            attempts.append((f"initialize(path={self.mt5_cfg.path})", mt5.last_error()))
            mt5.shutdown()

        ok = mt5.initialize()
        if ok:
            return
        attempts.append(("initialize()", mt5.last_error()))
        mt5.shutdown()

        detail = " | ".join([f"{name} -> {err}" for name, err in attempts])
        raise RuntimeError(f"Falha ao iniciar MT5. Tentativas: {detail}")

    def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError("Pacote MetaTrader5 não instalado.")
        if not (self.mt5_cfg.login and self.mt5_cfg.password and self.mt5_cfg.server):
            raise RuntimeError(
                "Credenciais MT5 incompletas. Preencha MT5_LOGIN, MT5_PASSWORD e MT5_SERVER no .env."
            )

        self._initialize_terminal()

        logged = mt5.login(
            login=self.mt5_cfg.login,
            password=self.mt5_cfg.password,
            server=self.mt5_cfg.server,
        )
        if not logged and mt5.last_error()[0] not in (0, 1):
            raise RuntimeError(f"Falha no login MT5: {mt5.last_error()}")

    def disconnect(self) -> None:
        if mt5 is not None:
            mt5.shutdown()

    def _can_send_now(self) -> bool:
        elapsed = time.time() - self.last_order_ts
        return elapsed >= self.exec_cfg.min_seconds_between_orders

    def _resolve_symbol(self) -> str:
        if self._resolved_symbol:
            return self._resolved_symbol

        requested = self.mt5_cfg.symbol
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
        candidates = [n for n in names if req_norm in n.lower().replace("/", "").replace("-", "").replace("_", "")]
        if not candidates:
            raise RuntimeError(f"Símbolo '{requested}' não encontrado no MT5.")

        chosen = candidates[0]
        if not mt5.symbol_select(chosen, True):
            raise RuntimeError(f"Símbolo '{chosen}' não pôde ser selecionado no Market Watch.")

        self._resolved_symbol = chosen
        return chosen

    def _resolve_timeframe(self) -> int:
        key = MT5_TIMEFRAME_CONST.get(self.mt5_cfg.timeframe.upper())
        if key is None:
            raise ValueError(f"Timeframe não suportado: {self.mt5_cfg.timeframe}")
        return getattr(mt5, key)

    def check_connection(self) -> dict:
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
            "symbol_requested": self.mt5_cfg.symbol,
            "symbol_resolved": resolved_symbol,
            "symbol_found": symbol_info is not None,
            "symbol_selected": bool(symbol_ok),
        }

    def fetch_historical(self, end_time: Optional[datetime] = None) -> pd.DataFrame:
        timeframe = self._resolve_timeframe()
        symbol = self._resolve_symbol()
        end_time = end_time or datetime.utcnow()

        rates = mt5.copy_rates_from(symbol, timeframe, end_time, self.mt5_cfg.history_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Sem dados para símbolo '{symbol}' (solicitado: '{self.mt5_cfg.symbol}') "
                f"em {self.mt5_cfg.timeframe}. Verifique se o ativo está habilitado no Market Watch."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]
        return df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    def _position_side(self) -> int:
        symbol = self._resolve_symbol()
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return 0

        total_volume = 0.0
        direction = 0.0
        for p in positions:
            signed = p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume
            total_volume += abs(signed)
            direction += signed

        if total_volume == 0:
            return 0
        return 1 if direction > 0 else -1

    def _get_tick_price(self, side: int = 0) -> float:
        symbol = self._resolve_symbol()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Não foi possível obter tick para '{symbol}'.")
        if side > 0:
            return float(tick.bid)
        if side < 0:
            return float(tick.ask)
        return float((tick.ask + tick.bid) / 2.0)

    def get_position_snapshot(self) -> dict:
        if self.exec_cfg.mode.lower() == "paper":
            unrealized_pnl = 0.0
            if self.paper_position != 0 and self.paper_entry_price > 0:
                current_price = self._get_tick_price(side=self.paper_position)
                unrealized_pnl = (current_price - self.paper_entry_price) * self.paper_position * self.paper_volume
            return {
                "symbol": self._resolve_symbol(),
                "side": self.paper_position,
                "volume": self.paper_volume,
                "avg_entry_price": self.paper_entry_price,
                "unrealized_pnl": unrealized_pnl,
            }

        symbol = self._resolve_symbol()
        positions = list(mt5.positions_get(symbol=symbol) or [])
        if not positions:
            return {
                "symbol": symbol,
                "side": 0,
                "volume": 0.0,
                "avg_entry_price": 0.0,
                "unrealized_pnl": 0.0,
            }

        signed_volume = 0.0
        weighted_entry_value = 0.0
        unrealized_pnl = 0.0
        for position in positions:
            sign = 1.0 if position.type == mt5.POSITION_TYPE_BUY else -1.0
            signed = sign * float(position.volume)
            signed_volume += signed
            weighted_entry_value += signed * float(position.price_open)
            unrealized_pnl += float(getattr(position, "profit", 0.0))

        if abs(signed_volume) <= 1e-12:
            avg_entry_price = 0.0
            side = 0
        else:
            avg_entry_price = abs(weighted_entry_value / signed_volume)
            side = 1 if signed_volume > 0 else -1

        return {
            "symbol": symbol,
            "side": side,
            "volume": abs(signed_volume),
            "avg_entry_price": avg_entry_price,
            "unrealized_pnl": unrealized_pnl,
        }

    def _paper_close_position(self, price: float) -> dict:
        if self.paper_position == 0 or self.paper_volume <= 0:
            return {"closed": False, "reason": "no_open_position"}

        pnl = (price - self.paper_entry_price) * self.paper_position * self.paper_volume
        snapshot = {
            "closed": True,
            "side": self.paper_position,
            "entry_price": self.paper_entry_price,
            "exit_price": price,
            "volume": self.paper_volume,
            "pnl": pnl,
        }
        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0
        return snapshot

    def _paper_check_stop_take(self) -> dict | None:
        if self.paper_position == 0 or self.paper_entry_price <= 0:
            return None

        current_price = self._get_tick_price(side=self.paper_position)
        move = ((current_price - self.paper_entry_price) / max(self.paper_entry_price, 1e-12)) * self.paper_position
        if move <= -self.stop_loss_pct:
            result = self._paper_close_position(current_price)
            result["trigger"] = "stop_loss"
            return result
        if move >= self.take_profit_pct:
            result = self._paper_close_position(current_price)
            result["trigger"] = "take_profit"
            return result
        return None

    def _get_open_positions(self) -> list:
        symbol = self._resolve_symbol()
        return list(mt5.positions_get(symbol=symbol) or [])

    def get_symbol_trading_spec(self) -> SymbolTradingSpec:
        symbol = self._resolve_symbol()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Não foi possível obter symbol_info para '{symbol}'.")

        return SymbolTradingSpec(
            symbol=symbol,
            volume_min=float(getattr(info, "volume_min", 0.0)),
            volume_step=float(getattr(info, "volume_step", 0.0)),
            volume_max=float(getattr(info, "volume_max", 0.0)),
            contract_size=float(getattr(info, "trade_contract_size", 0.0)),
            point=float(getattr(info, "point", 0.0)),
        )

    def get_account_balance(self) -> float:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("Não foi possível obter account_info do MT5.")
        return float(getattr(account, "balance", 0.0))

    def get_account_risk_state(self) -> dict:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("Não foi possível obter account_info do MT5.")

        balance = float(getattr(account, "balance", 0.0))
        equity = float(getattr(account, "equity", balance))
        drawdown_pct = 0.0 if balance <= 0 else max(0.0, (balance - equity) / balance)
        defensive = drawdown_pct >= float(self.exec_cfg.defensive_drawdown_pct)
        return {
            "balance": balance,
            "equity": equity,
            "drawdown_pct": drawdown_pct,
            "defensive": defensive,
        }

    def _round_volume_to_step(self, volume: float, volume_min: float, volume_step: float) -> float:
        if volume_step <= 0:
            return max(volume, volume_min)
        rounded = floor(volume / volume_step) * volume_step
        return max(rounded, volume_min)

    def calculate_order_volume(self, balance: float, stop_loss_pct: float, defensive: bool = False) -> dict:
        spec = self.get_symbol_trading_spec()
        symbol = spec.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Não foi possível obter tick para '{symbol}'.")

        price = float(tick.ask)
        risk_pct = float(self.exec_cfg.risk_per_trade)
        if defensive:
            risk_pct *= float(self.exec_cfg.defensive_risk_multiplier)

        risk_amount = balance * risk_pct
        stop_distance_price = price * float(stop_loss_pct)
        loss_per_lot = stop_distance_price * max(spec.contract_size, 1.0)
        raw_volume = 0.0 if loss_per_lot <= 0 else (risk_amount / loss_per_lot)

        adjusted_volume = self._round_volume_to_step(raw_volume, spec.volume_min, spec.volume_step)
        adjusted_volume = min(adjusted_volume, spec.volume_max) if spec.volume_max > 0 else adjusted_volume
        effective_risk = adjusted_volume * loss_per_lot
        exceeds_target_risk = effective_risk > risk_amount if risk_amount > 0 else False

        return {
            "symbol": symbol,
            "price": price,
            "balance": balance,
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "stop_loss_pct": stop_loss_pct,
            "stop_distance_price": stop_distance_price,
            "contract_size": spec.contract_size,
            "volume_min": spec.volume_min,
            "volume_step": spec.volume_step,
            "volume_max": spec.volume_max,
            "raw_volume": raw_volume,
            "adjusted_volume": adjusted_volume,
            "effective_risk_amount": effective_risk,
            "exceeds_target_risk": exceeds_target_risk,
        }

    def _send_market_order(self, order_type: int, volume: float) -> dict:
        symbol = self._resolve_symbol()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "error": "Sem tick disponível"}

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        if order_type == mt5.ORDER_TYPE_BUY:
            sl = price * (1.0 - self.stop_loss_pct)
            tp = price * (1.0 + self.take_profit_pct)
        else:
            sl = price * (1.0 + self.stop_loss_pct)
            tp = price * (1.0 - self.take_profit_pct)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self.exec_cfg.deviation,
            "magic": self.exec_cfg.magic,
            "comment": "traderbot_rl",
            "sl": sl,
            "tp": tp,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "error": str(mt5.last_error())}

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return {
            "ok": ok,
            "retcode": result.retcode,
            "comment": result.comment,
            "price": price,
            "sl": sl,
            "tp": tp,
        }

    def _close_position_ticket(self, position, volume: float | None = None) -> dict:
        symbol = self._resolve_symbol()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "error": "Sem tick disponível"}

        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume if volume is not None else position.volume),
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": self.exec_cfg.deviation,
            "magic": self.exec_cfg.magic,
            "comment": "traderbot_rl_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "error": str(mt5.last_error()), "ticket": position.ticket}
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        return {
            "ok": ok,
            "retcode": result.retcode,
            "comment": result.comment,
            "ticket": position.ticket,
            "volume": request["volume"],
        }

    def execute(self, decision: ExecutionDecision) -> dict:
        """Executa decisão do agente.

        action: 0=HOLD, 1=BUY, 2=SELL
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.exec_cfg.mode.lower() == "paper":
            stop_take_event = self._paper_check_stop_take()
            paper_volume = self.exec_cfg.lot
            sizing = None
            if self.exec_cfg.use_dynamic_position_sizing:
                balance = self.exec_cfg.validation_balance
                sizing = self.calculate_order_volume(
                    balance=balance,
                    stop_loss_pct=self.stop_loss_pct,
                    defensive=False,
                )
                paper_volume = float(sizing["adjusted_volume"])

            opened = False
            open_price = None
            if self.paper_position == 0 and decision.action in (1, 2):
                self.paper_position = 1 if decision.action == 1 else -1
                self.paper_entry_price = self._get_tick_price(side=-self.paper_position)
                self.paper_volume = paper_volume
                opened = True
                open_price = self.paper_entry_price
            return {
                "ok": True,
                "mode": "paper",
                "timestamp": timestamp,
                "action": decision.action,
                "reason": decision.reason,
                "volume": paper_volume,
                "sizing": sizing,
                "opened_position": opened,
                "open_price": open_price,
                "position_snapshot": self.get_position_snapshot(),
                "stop_take_event": stop_take_event,
            }

        if not self.exec_cfg.allow_live_trading:
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "allow_live_trading=false. Execução real bloqueada por segurança.",
            }

        if not self._can_send_now():
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "Aguardando cooldown entre ordens.",
            }

        current_side = self._position_side()
        if current_side != 0:
            return {
                "ok": True,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "message": "Posição já aberta; aguardando stop/take do broker.",
                "position_snapshot": self.get_position_snapshot(),
            }
        target_side = {0: 0, 1: 1, 2: -1}[decision.action]
        if target_side == 0:
            return {
                "ok": True,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "message": "Sem sinal de entrada nesta barra.",
            }

        risk_state = self.get_account_risk_state()
        sizing = self.calculate_order_volume(
            balance=risk_state["balance"],
            stop_loss_pct=self.stop_loss_pct,
            defensive=bool(risk_state["defensive"]),
        )
        if self.exec_cfg.use_dynamic_position_sizing and bool(sizing.get("exceeds_target_risk")):
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "Lote mínimo do broker excede o risco alvo configurado.",
                "sizing": sizing,
                "risk_state": risk_state,
            }
        volume = float(sizing["adjusted_volume"]) if self.exec_cfg.use_dynamic_position_sizing else self.exec_cfg.lot

        responses = []
        if target_side != 0:
            open_type = mt5.ORDER_TYPE_BUY if target_side == 1 else mt5.ORDER_TYPE_SELL
            responses.append(self._send_market_order(open_type, volume))

        self.last_order_ts = time.time()
        ok = all(r.get("ok") for r in responses) if responses else True
        return {
            "ok": ok,
            "mode": self.exec_cfg.mode,
            "timestamp": timestamp,
            "volume": volume,
            "sizing": sizing,
            "risk_state": risk_state,
            "responses": responses,
        }
