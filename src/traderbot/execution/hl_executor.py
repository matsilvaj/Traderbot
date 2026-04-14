from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from typing import Optional

from traderbot.config import ExecutionConfig, HyperliquidConfig
from traderbot.data.hl_loader import HLDataLoader

try:
    import eth_account
    from eth_account.signers.local import LocalAccount
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except ImportError:  # pragma: no cover
    eth_account = None
    LocalAccount = None
    Exchange = None
    Info = None
    constants = None


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


class HyperliquidExecutor:
    """Executor paper/live para Hyperliquid mantendo interface próxima à do MT5."""

    DEFAULT_VOLUME_MIN = 0.0001
    DEFAULT_VOLUME_STEP = 0.0001
    DEFAULT_VOLUME_MAX = 22.0
    DEFAULT_CONTRACT_SIZE = 1.0
    DEFAULT_POINT = 1.0

    def __init__(
        self,
        hl_cfg: HyperliquidConfig,
        exec_cfg: ExecutionConfig,
        stop_loss_pct: float = 0.003,
        take_profit_pct: float = 0.006,
        slippage_pct: float = 0.00010,
    ):
        self.hl_cfg = hl_cfg
        self.exec_cfg = exec_cfg
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.slippage_pct = float(slippage_pct)
        self.last_order_ts = 0.0

        self.loader = HLDataLoader(self.hl_cfg)
        self.info: Optional[Info] = None
        self.exchange: Optional[Exchange] = None
        self.account: Optional[LocalAccount] = None
        self.address: Optional[str] = None

        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0

    def _base_url(self) -> str:
        network = str(self.hl_cfg.network).lower().strip()
        if constants is None:
            return ""
        if network == "testnet":
            return constants.TESTNET_API_URL
        return constants.MAINNET_API_URL

    def _ensure_info(self) -> Info:
        if self.info is None:
            self.loader.connect()
            self.info = self.loader.info
        return self.info

    def _can_send_now(self) -> bool:
        elapsed = time.time() - self.last_order_ts
        return elapsed >= self.exec_cfg.min_seconds_between_orders

    def _clean_secret(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().strip("\"'").strip()
        return value or None

    def connect(self) -> None:
        if Info is None or Exchange is None or LocalAccount is None:
            raise RuntimeError(
                "Pacote hyperliquid-python-sdk não encontrado. Instale as dependências do requirements.txt."
            )

        self.loader.connect()
        self.info = self.loader.info

        wallet_address = self._clean_secret(self.hl_cfg.wallet_address)
        private_key = self._clean_secret(self.hl_cfg.private_key)

        if private_key is not None:
            self.account = eth_account.Account.from_key(private_key)
            self.address = wallet_address or self.account.address
            self.exchange = Exchange(
                self.account,
                self._base_url(),
                account_address=self.address,
            )
        else:
            self.account = None
            self.address = wallet_address
            self.exchange = None

        if self.exec_cfg.mode.lower() != "paper" and self.exec_cfg.allow_live_trading and self.exchange is None:
            raise RuntimeError(
                "Execução live na Hyperliquid requer HL_PRIVATE_KEY no .env."
            )

    def disconnect(self) -> None:
        self.loader.disconnect()
        self.info = None
        self.exchange = None
        self.account = None
        self.address = None

    def fetch_historical(self, end_time: Optional[datetime] = None):
        return self.loader.fetch_historical(end_time=end_time)

    def _get_mid_price(self) -> float:
        info = self._ensure_info()
        mids = info.all_mids()
        symbol = str(self.hl_cfg.symbol)
        price = mids.get(symbol)
        if price is None:
            raise RuntimeError(f"Preço médio indisponível para '{symbol}' na Hyperliquid.")
        return float(price)

    def check_connection(self) -> dict:
        status = self.loader.check_connection()
        status["wallet_address"] = self.address
        status["can_trade"] = self.exchange is not None
        if self.address:
            try:
                user_state = self._ensure_info().user_state(self.address)
                margin = user_state.get("marginSummary", {})
                status["account_value"] = float(margin.get("accountValue", 0.0))
                status["open_positions"] = len(user_state.get("assetPositions", []))
            except Exception as exc:  # pragma: no cover
                status["account_error"] = str(exc)
        return status

    def get_symbol_trading_spec(self) -> SymbolTradingSpec:
        return SymbolTradingSpec(
            symbol=str(self.hl_cfg.symbol),
            volume_min=self.DEFAULT_VOLUME_MIN,
            volume_step=self.DEFAULT_VOLUME_STEP,
            volume_max=self.DEFAULT_VOLUME_MAX,
            contract_size=self.DEFAULT_CONTRACT_SIZE,
            point=self.DEFAULT_POINT,
        )

    def get_account_balance(self) -> float:
        if self.exec_cfg.mode.lower() == "paper":
            return float(self.exec_cfg.validation_balance)
        if not self.address:
            raise RuntimeError("wallet_address não configurado para consultar saldo na Hyperliquid.")

        user_state = self._ensure_info().user_state(self.address)
        margin = user_state.get("marginSummary", {})
        return float(margin.get("accountValue", 0.0))

    def get_account_risk_state(self) -> dict:
        balance = self.get_account_balance()
        return {
            "balance": balance,
            "equity": balance,
            "drawdown_pct": 0.0,
            "defensive": False,
        }

    def _round_volume_to_step(self, volume: float, volume_min: float, volume_step: float) -> float:
        if volume_step <= 0:
            return max(volume, volume_min)
        rounded = floor(volume / volume_step) * volume_step
        return max(rounded, volume_min)

    def calculate_order_volume(self, balance: float, stop_loss_pct: float, defensive: bool = False) -> dict:
        spec = self.get_symbol_trading_spec()
        price = self._get_mid_price()
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
            "symbol": spec.symbol,
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

        current_price = self._get_mid_price()
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

    def get_position_snapshot(self) -> dict:
        symbol = str(self.hl_cfg.symbol)
        if self.exec_cfg.mode.lower() == "paper":
            unrealized_pnl = 0.0
            if self.paper_position != 0 and self.paper_entry_price > 0:
                current_price = self._get_mid_price()
                unrealized_pnl = (
                    (current_price - self.paper_entry_price) * self.paper_position * self.paper_volume
                )
            return {
                "symbol": symbol,
                "side": self.paper_position,
                "volume": self.paper_volume,
                "avg_entry_price": self.paper_entry_price,
                "unrealized_pnl": unrealized_pnl,
            }

        if not self.address:
            return {
                "symbol": symbol,
                "side": 0,
                "volume": 0.0,
                "avg_entry_price": 0.0,
                "unrealized_pnl": 0.0,
            }

        user_state = self._ensure_info().user_state(self.address)
        positions = user_state.get("assetPositions", [])
        for wrapped in positions:
            position = wrapped.get("position", {})
            if position.get("coin") != symbol:
                continue

            szi = float(position.get("szi", 0.0))
            if abs(szi) <= 1e-12:
                break

            entry_px = position.get("entryPx")
            return {
                "symbol": symbol,
                "side": 1 if szi > 0 else -1,
                "volume": abs(szi),
                "avg_entry_price": float(entry_px) if entry_px is not None else 0.0,
                "unrealized_pnl": float(position.get("unrealizedPnl", 0.0)),
            }

        return {
            "symbol": symbol,
            "side": 0,
            "volume": 0.0,
            "avg_entry_price": 0.0,
            "unrealized_pnl": 0.0,
        }

    def _response_ok(self, response) -> bool:
        return isinstance(response, dict) and response.get("status") == "ok"

    def _maybe_close_live_position(self, timestamp: str) -> dict | None:
        snapshot = self.get_position_snapshot()
        if snapshot["side"] == 0 or snapshot["avg_entry_price"] <= 0:
            return None

        current_price = self._get_mid_price()
        move = ((current_price - snapshot["avg_entry_price"]) / max(snapshot["avg_entry_price"], 1e-12)) * snapshot["side"]
        trigger = None
        if move <= -self.stop_loss_pct:
            trigger = "stop_loss"
        elif move >= self.take_profit_pct:
            trigger = "take_profit"

        if trigger is None:
            return None

        if self.exchange is None:
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "Exchange da Hyperliquid não inicializado para encerrar posição.",
            }

        response = self.exchange.market_close(
            str(self.hl_cfg.symbol),
            sz=float(snapshot["volume"]),
            px=current_price,
            slippage=float(self.slippage_pct),
        )
        self.last_order_ts = time.time()
        return {
            "ok": self._response_ok(response),
            "mode": self.exec_cfg.mode,
            "timestamp": timestamp,
            "trigger": trigger,
            "response": response,
            "position_snapshot": self.get_position_snapshot(),
        }

    def execute(self, decision: ExecutionDecision) -> dict:
        """Executa decisão do agente.

        action: 0=HOLD, 1=BUY, 2=SELL
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.exec_cfg.mode.lower() == "paper":
            stop_take_event = self._paper_check_stop_take()
            paper_volume = float(self.exec_cfg.lot)
            sizing = None
            if self.exec_cfg.use_dynamic_position_sizing:
                sizing = self.calculate_order_volume(
                    balance=float(self.exec_cfg.validation_balance),
                    stop_loss_pct=self.stop_loss_pct,
                    defensive=False,
                )
                paper_volume = float(sizing["adjusted_volume"])

            opened = False
            open_price = None
            if self.paper_position == 0 and decision.action in (1, 2):
                self.paper_position = 1 if decision.action == 1 else -1
                self.paper_entry_price = self._get_mid_price()
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

        if self.exchange is None:
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "HL_PRIVATE_KEY ausente ou executor da Hyperliquid não inicializado.",
            }

        stop_take_result = self._maybe_close_live_position(timestamp)
        if stop_take_result is not None:
            return stop_take_result

        if not self._can_send_now():
            return {
                "ok": False,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "error": "Aguardando cooldown entre ordens.",
            }

        current_snapshot = self.get_position_snapshot()
        if current_snapshot["side"] != 0:
            return {
                "ok": True,
                "mode": self.exec_cfg.mode,
                "timestamp": timestamp,
                "message": "Posição já aberta; aguardando stop/take gerenciado pelo bot.",
                "position_snapshot": current_snapshot,
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
                "error": "Lote mínimo configurado excede o risco alvo.",
                "sizing": sizing,
                "risk_state": risk_state,
            }

        volume = float(sizing["adjusted_volume"]) if self.exec_cfg.use_dynamic_position_sizing else float(self.exec_cfg.lot)
        price = self._get_mid_price()
        response = self.exchange.market_open(
            str(self.hl_cfg.symbol),
            target_side == 1,
            volume,
            px=price,
            slippage=float(self.slippage_pct),
        )
        self.last_order_ts = time.time()

        return {
            "ok": self._response_ok(response),
            "mode": self.exec_cfg.mode,
            "timestamp": timestamp,
            "volume": volume,
            "sizing": sizing,
            "risk_state": risk_state,
            "response": response,
            "position_snapshot": self.get_position_snapshot(),
        }
