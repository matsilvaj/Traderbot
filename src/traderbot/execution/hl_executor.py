from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor
from typing import Optional

import pandas as pd

from traderbot.config import EnvironmentConfig, ExecutionConfig, HyperliquidConfig, TIMEFRAME_MINUTES_MAP
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
    action: float
    reason: str = ""
    entry_distance_from_ema_240: float = 0.0
    regime_valid_for_entry: bool = False
    regime_dist_ema_240: float = 0.0
    regime_vol_regime_z: float = 0.0
    reference_price: float = 0.0
    bar_timestamp: str = ""


@dataclass
class SymbolTradingSpec:
    symbol: str
    volume_min: float
    volume_step: float
    volume_max: float
    contract_size: float
    point: float


class HyperliquidExecutor:
    """Paper/live executor for Hyperliquid using continuous actions in [-1, 1]."""

    DEFAULT_VOLUME_MIN = 0.0001
    DEFAULT_VOLUME_STEP = 0.0001
    DEFAULT_VOLUME_MAX = 22.0
    DEFAULT_CONTRACT_SIZE = 1.0
    DEFAULT_POINT = 1.0
    AVAILABLE_TO_TRADE_SOURCE_FIELD = "spot_user_state.tokenToAvailableAfterMaintenance[token=0]"
    AVAILABLE_TO_TRADE_FALLBACK_FIELD = "spot_user_state.balances[coin=USDC].total - hold"

    def __init__(
        self,
        hl_cfg: HyperliquidConfig,
        exec_cfg: ExecutionConfig,
        env_cfg: EnvironmentConfig,
        stop_loss_pct: float = 0.003,
        take_profit_pct: float = 0.006,
        slippage_pct: float = 0.00010,
    ):
        self.hl_cfg = hl_cfg
        self.exec_cfg = exec_cfg
        self.env_cfg = env_cfg
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.slippage_pct = float(slippage_pct)
        self.order_slippage = float(getattr(self.exec_cfg, "order_slippage", 0.05))
        self.last_order_ts = 0.0

        self.loader = HLDataLoader(self.hl_cfg)
        self.info: Optional[Info] = None
        self.exchange: Optional[Exchange] = None
        self.account: Optional[LocalAccount] = None
        self.address: Optional[str] = None

        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0
        self.paper_balance = float(self.env_cfg.initial_balance)
        self.session_peak_balance = float(self.env_cfg.initial_balance)
        self.paper_entry_opened_at: Optional[float] = None
        self.paper_entry_distance_from_ema_240 = 0.0
        self.live_entry_opened_at: Optional[float] = None
        self.live_entry_distance_from_ema_240 = 0.0
        self.paper_cooldown_until_ts = 0.0
        self.live_cooldown_until_ts = 0.0
        self.paper_last_order_bar_timestamp = ""
        self.live_last_order_bar_timestamp = ""
        self.market_cache: Optional[pd.DataFrame] = None
        self.paper_balance_seed_source = "paper_local_balance(config.initial_balance)"

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

    def _call_with_retry(self, label: str, fn):
        attempts = max(1, int(getattr(self.exec_cfg, "api_retry_attempts", 1)))
        delay_seconds = max(0.0, float(getattr(self.exec_cfg, "api_retry_delay_seconds", 0)))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - depends on runtime API failures
                last_exc = exc
                if attempt >= attempts:
                    break
                time.sleep(delay_seconds)
        raise RuntimeError(f"{label} failed after {attempts} attempt(s): {last_exc}") from last_exc

    def _can_send_now(self) -> bool:
        elapsed = time.time() - self.last_order_ts
        return elapsed >= self.exec_cfg.min_seconds_between_orders

    def _has_sent_order_in_bar(self, mode: str, bar_timestamp: str) -> bool:
        if not bar_timestamp:
            return False
        if str(mode).lower() == "paper":
            return bar_timestamp == self.paper_last_order_bar_timestamp
        return bar_timestamp == self.live_last_order_bar_timestamp

    def _mark_order_sent_in_bar(self, mode: str, bar_timestamp: str) -> None:
        if not bar_timestamp:
            return
        if str(mode).lower() == "paper":
            self.paper_last_order_bar_timestamp = bar_timestamp
        else:
            self.live_last_order_bar_timestamp = bar_timestamp

    def _clean_secret(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().strip("\"'").strip()
        return value or None

    def _execution_mode(self) -> str:
        mode = str(getattr(self.exec_cfg, "execution_mode", "paper_local")).strip().lower()
        if mode == "paper":
            return "paper_local"
        if mode == "live":
            return "exchange"
        return mode or "paper_local"

    def _paper_local_balance_source(self) -> str:
        if self.env_cfg.simulation_initial_balance is not None:
            return "paper_local_balance(config.simulation_initial_balance)"
        return "paper_local_balance(config.initial_balance)"

    def _operational_mode_label(self) -> str:
        if self._execution_mode() == "paper_local":
            return "paper_local"
        return str(self.hl_cfg.network).lower().strip()

    def connect(self) -> None:
        if Info is None or Exchange is None or LocalAccount is None:
            raise RuntimeError(
                "Hyperliquid SDK not found. Install the dependencies from requirements.txt."
            )

        self.loader.connect()
        self.info = self.loader.info
        self.paper_balance = float(self.env_cfg.initial_balance)
        self.session_peak_balance = float(self.env_cfg.initial_balance)
        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0
        self.paper_entry_opened_at = None
        self.paper_entry_distance_from_ema_240 = 0.0
        self.live_entry_opened_at = None
        self.live_entry_distance_from_ema_240 = 0.0
        self.paper_cooldown_until_ts = 0.0
        self.live_cooldown_until_ts = 0.0
        self.paper_last_order_bar_timestamp = ""
        self.live_last_order_bar_timestamp = ""
        self.market_cache = None
        self.paper_balance_seed_source = self._paper_local_balance_source()

        wallet_address = self._clean_secret(self.hl_cfg.wallet_address)
        private_key = self._clean_secret(self.hl_cfg.private_key)
        execution_mode = self._execution_mode()

        if private_key is not None:
            self.account = eth_account.Account.from_key(private_key)
            self.address = wallet_address or self.account.address
            if execution_mode == "exchange" and self.exec_cfg.allow_live_trading:
                base_url = self._base_url()
                exchange_kwargs = {
                    "base_url": base_url,
                    "account_address": self.address,
                }
                if str(self.hl_cfg.network).lower().strip() == "testnet":
                    exchange_kwargs["meta"] = self.loader._info_post(base_url, {"type": "meta"})
                    exchange_kwargs["spot_meta"] = self.loader._normalize_spot_meta(
                        self.loader._info_post(base_url, {"type": "spotMeta"})
                    )
                self.exchange = Exchange(
                    self.account,
                    **exchange_kwargs,
                )
            else:
                self.exchange = None
        else:
            self.account = None
            self.address = wallet_address
            self.exchange = None

        if execution_mode == "exchange" and self.exec_cfg.allow_live_trading and self.exchange is None:
            raise RuntimeError("Live trading on Hyperliquid requires HL_PRIVATE_KEY in .env.")

    def disconnect(self) -> None:
        self.loader.disconnect()
        self.info = None
        self.exchange = None
        self.account = None
        self.address = None
        self.market_cache = None

    def fetch_historical(self, end_time: Optional[datetime] = None, bars: Optional[int] = None):
        return self._call_with_retry(
            "fetch_historical",
            lambda: self.loader.fetch_historical(end_time=end_time, bars=bars),
        )

    def get_live_market_window(self) -> pd.DataFrame:
        target_bars = max(int(self.hl_cfg.live_window_bars), 1)
        if self.market_cache is None or self.market_cache.empty:
            self.market_cache = self.fetch_historical(bars=target_bars)
        else:
            self.market_cache = self._call_with_retry(
                "update_market_window",
                lambda: self.loader.update_dataframe(self.market_cache),
            )
        if len(self.market_cache) > target_bars:
            self.market_cache = self.market_cache.tail(target_bars).copy()
        return self.market_cache.copy()

    def _get_mid_price(self) -> float:
        info = self._ensure_info()
        mids = self._call_with_retry("all_mids", info.all_mids)
        symbol = str(self.hl_cfg.symbol)
        price = mids.get(symbol)
        if price is None:
            raise RuntimeError(f"Mid price unavailable for '{symbol}' on Hyperliquid.")
        return float(price)

    def check_connection(self) -> dict:
        status = self.loader.check_connection()
        status["execution_mode"] = self._execution_mode()
        status["base_url"] = self._base_url()
        status["wallet_address"] = self.address
        status["wallet_loaded"] = self.account is not None
        status["can_trade"] = (
            self.exchange is not None
            and self._execution_mode() != "paper_local"
            and bool(self.exec_cfg.allow_live_trading)
        )
        if self.address:
            try:
                user_state = self._call_with_retry(
                    "user_state",
                    lambda: self._ensure_info().user_state(self.address),
                )
                margin = user_state.get("marginSummary", {})
                status["account_value"] = float(margin.get("accountValue", 0.0))
                status["perp_account_value"] = float(margin.get("accountValue", 0.0))
                status["open_positions"] = len(user_state.get("assetPositions", []))
                network_balance_snapshot = self.get_available_to_trade_snapshot()
                status["network_available_to_trade"] = float(network_balance_snapshot["available_to_trade"])
                status["network_available_to_trade_source_field"] = str(
                    network_balance_snapshot["available_to_trade_source_field"]
                )
            except Exception as exc:  # pragma: no cover
                status["account_error"] = str(exc)
        sizing_context = self.get_sizing_balance_context(self._execution_mode())
        status["available_to_trade"] = float(sizing_context["available_to_trade"])
        status["available_to_trade_source_field"] = str(sizing_context["available_to_trade_source_field"])
        return status

    def _spot_user_state(self) -> dict:
        if not self.address:
            return {}
        return self._call_with_retry(
            "spot_user_state",
            lambda: self._ensure_info().spot_user_state(self.address),
        )

    @classmethod
    def _extract_available_usdc_with_source(cls, spot_state: dict) -> tuple[float, str]:
        token_rows = spot_state.get("tokenToAvailableAfterMaintenance", []) or []
        for row in token_rows:
            if isinstance(row, (list, tuple)) and len(row) >= 2 and int(row[0]) == 0:
                return float(row[1]), cls.AVAILABLE_TO_TRADE_SOURCE_FIELD

        balances = spot_state.get("balances", []) or []
        for balance in balances:
            if str(balance.get("coin", "")).upper() == "USDC":
                total = float(balance.get("total", 0.0))
                hold = float(balance.get("hold", 0.0))
                return max(0.0, total - hold), cls.AVAILABLE_TO_TRADE_FALLBACK_FIELD
        return 0.0, cls.AVAILABLE_TO_TRADE_SOURCE_FIELD

    @classmethod
    def _extract_available_usdc(cls, spot_state: dict) -> float:
        return float(cls._extract_available_usdc_with_source(spot_state)[0])

    def get_available_to_trade_snapshot(self) -> dict:
        if not self.address:
            return {
                "available_to_trade": 0.0,
                "available_to_trade_source_field": self.AVAILABLE_TO_TRADE_SOURCE_FIELD,
            }
        spot_state = self._spot_user_state()
        available_to_trade, source_field = self._extract_available_usdc_with_source(spot_state)
        return {
            "available_to_trade": float(available_to_trade),
            "available_to_trade_source_field": str(source_field),
        }

    def get_hyperliquid_available_balance(self) -> float:
        return float(self.get_available_to_trade_snapshot()["available_to_trade"])

    def get_total_account_value(self) -> float:
        if self._execution_mode() == "paper_local":
            return float(self.paper_balance)
        if not self.address:
            raise RuntimeError("wallet_address is required to query Hyperliquid balance.")

        user_state = self._call_with_retry(
            "account_balance_user_state",
            lambda: self._ensure_info().user_state(self.address),
        )
        margin = user_state.get("marginSummary", {})
        balance = float(margin.get("accountValue", 0.0))
        if balance > 0:
            return balance
        return float(self.get_hyperliquid_available_balance())

    def get_symbol_trading_spec(self) -> SymbolTradingSpec:
        return SymbolTradingSpec(
            symbol=str(self.hl_cfg.symbol),
            volume_min=float(getattr(self.env_cfg, "broker_volume_min", self.DEFAULT_VOLUME_MIN)),
            volume_step=float(getattr(self.env_cfg, "broker_volume_step", self.DEFAULT_VOLUME_STEP)),
            volume_max=float(getattr(self.env_cfg, "broker_volume_max", self.DEFAULT_VOLUME_MAX)),
            contract_size=float(getattr(self.env_cfg, "broker_contract_size", self.DEFAULT_CONTRACT_SIZE)),
            point=float(getattr(self.env_cfg, "broker_point", self.DEFAULT_POINT)),
        )

    def get_account_balance(self) -> float:
        if self._execution_mode() == "paper_local":
            return float(self.paper_balance)
        if not self.address:
            raise RuntimeError("wallet_address is required to query Hyperliquid balance.")

        available = float(self.get_hyperliquid_available_balance())
        if available > 0:
            return available
        return float(self.get_total_account_value())

    def _paper_sizing_balance_source(self) -> str:
        return self.paper_balance_seed_source

    @staticmethod
    def _leverage_context() -> dict:
        return {
            "leverage_configured": None,
            "leverage_applied": None,
            "leverage_used": None,
            "margin_estimated_consumed": None,
            "has_explicit_leverage_logic": False,
        }

    def get_sizing_balance_context(self, mode: Optional[str] = None) -> dict:
        mode = str(mode or self._execution_mode()).lower()
        leverage_context = self._leverage_context()

        if mode == "paper_local":
            return {
                "available_to_trade": float(self.paper_balance),
                "available_to_trade_source_field": self._paper_sizing_balance_source(),
                "sizing_balance_used": float(self.paper_balance),
                "sizing_balance_source": self._paper_sizing_balance_source(),
                **leverage_context,
            }

        available_snapshot = self.get_available_to_trade_snapshot()
        available_to_trade = float(available_snapshot["available_to_trade"])
        available_source_field = str(available_snapshot["available_to_trade_source_field"])

        if available_to_trade > 0:
            return {
                "available_to_trade": available_to_trade,
                "available_to_trade_source_field": available_source_field,
                "sizing_balance_used": available_to_trade,
                "sizing_balance_source": available_source_field,
                **leverage_context,
            }

        return {
            "available_to_trade": available_to_trade,
            "available_to_trade_source_field": available_source_field,
            "sizing_balance_used": float(self.get_total_account_value()),
            "sizing_balance_source": "user_state.marginSummary.accountValue",
            **leverage_context,
        }

    def get_account_risk_state(self) -> dict:
        balance = self.get_account_balance()
        self.session_peak_balance = max(float(self.session_peak_balance), float(balance))
        drawdown_pct = (
            (float(balance) - float(self.session_peak_balance)) / max(float(self.session_peak_balance), 1e-12)
        )
        defensive_threshold = abs(float(self.exec_cfg.defensive_drawdown_pct))
        defensive_enabled = defensive_threshold > 0 and float(self.exec_cfg.defensive_risk_multiplier) < 1.0
        defensive = defensive_enabled and drawdown_pct <= -defensive_threshold
        return {
            "balance": balance,
            "equity": balance,
            "drawdown_pct": drawdown_pct,
            "defensive": defensive,
        }

    def _round_volume_to_step(self, volume: float, volume_min: float, volume_step: float) -> float:
        if volume_step <= 0:
            return max(volume, volume_min)
        rounded = floor(volume / volume_step) * volume_step
        return max(rounded, volume_min)

    def _round_volume_up_to_step(self, volume: float, volume_min: float, volume_step: float) -> float:
        if volume_step <= 0:
            return max(volume, volume_min)
        rounded = ceil(volume / volume_step) * volume_step
        return max(rounded, volume_min)

    def _timeframe_seconds(self) -> float:
        minutes = TIMEFRAME_MINUTES_MAP.get(str(self.hl_cfg.timeframe).upper(), 1)
        return max(60.0, float(minutes) * 60.0)

    def _trade_cooldown_seconds(self) -> float:
        return max(0.0, float(max(int(self.env_cfg.trade_cooldown_steps), 0)) * self._timeframe_seconds())

    def _set_trade_cooldown(self, mode: str) -> None:
        cooldown_until = time.time() + self._trade_cooldown_seconds()
        if str(mode).lower() == "paper":
            self.paper_cooldown_until_ts = cooldown_until
        else:
            self.live_cooldown_until_ts = cooldown_until

    def _is_trade_cooldown_active(self, mode: str) -> bool:
        now = time.time()
        if str(mode).lower() == "paper":
            return now < float(self.paper_cooldown_until_ts)
        return now < float(self.live_cooldown_until_ts)

    def _trade_cooldown_remaining_seconds(self, mode: str) -> float:
        now = time.time()
        if str(mode).lower() == "paper":
            return max(0.0, float(self.paper_cooldown_until_ts) - now)
        return max(0.0, float(self.live_cooldown_until_ts) - now)

    def _position_bars(self, opened_at: Optional[float]) -> float:
        if opened_at is None:
            return 0.0
        elapsed = max(0.0, time.time() - opened_at)
        return float(floor(elapsed / self._timeframe_seconds()))

    def _decision_regime_valid(self, decision: ExecutionDecision) -> bool:
        if bool(decision.regime_valid_for_entry):
            return True
        return (
            abs(float(decision.regime_dist_ema_240)) >= float(self.env_cfg.regime_min_abs_dist_ema_240)
            and float(decision.regime_vol_regime_z) > float(self.env_cfg.regime_min_vol_regime_z)
        )

    def _stop_take_prices(self, entry_price: float, side: int) -> tuple[float | None, float | None]:
        if side == 0 or entry_price <= 0:
            return None, None
        stop_price = float(entry_price * (1.0 - self.stop_loss_pct * side))
        take_price = float(entry_price * (1.0 + self.take_profit_pct * side))
        return stop_price, take_price

    def _trade_units(self, volume: float) -> float:
        if not bool(getattr(self.env_cfg, "use_broker_constraints", True)):
            return 1.0
        spec = self.get_symbol_trading_spec()
        return float(volume) * max(float(spec.contract_size), 1.0)

    def _estimate_cost(self, price: float, volume: float) -> float:
        notional_value = float(price) * self._trade_units(volume)
        total_fee_rate = float(self.env_cfg.taker_fee_pct) + float(self.env_cfg.slippage_pct)
        return notional_value * total_fee_rate

    def calculate_order_volume(
        self,
        balance: float,
        dynamic_risk_pct: float,
        price: Optional[float] = None,
        defensive: bool = False,
        available_to_trade: Optional[float] = None,
        available_to_trade_source_field: Optional[str] = None,
        sizing_balance_source: Optional[str] = None,
    ) -> dict:
        spec = self.get_symbol_trading_spec()
        price = float(self._get_mid_price() if price is None else price)
        risk_pct = float(dynamic_risk_pct)
        if defensive:
            risk_pct *= float(self.exec_cfg.defensive_risk_multiplier)

        risk_amount = balance * risk_pct
        stop_distance_price = price * float(self.stop_loss_pct)
        contract_multiplier = max(spec.contract_size, 1.0)
        price_per_volume = price * contract_multiplier
        target_notional = 0.0 if float(self.stop_loss_pct) <= 0 else (risk_amount / float(self.stop_loss_pct))
        raw_volume = 0.0 if price_per_volume <= 0 else (target_notional / price_per_volume)

        adjusted_volume = self._round_volume_to_step(raw_volume, spec.volume_min, spec.volume_step)
        min_notional_usd = float(getattr(self.env_cfg, "broker_min_notional_usd", 10.0))
        blocked_by_min_notional = False
        bumped_to_min_notional = False
        skipped_due_to_min_notional_risk = False
        adjusted_notional = adjusted_volume * price_per_volume
        effective_risk = adjusted_notional * float(self.stop_loss_pct)

        if min_notional_usd > 0 and adjusted_notional < min_notional_usd:
            minimum_viable_volume = self._round_volume_up_to_step(
                min_notional_usd / max(price_per_volume, 1e-12),
                spec.volume_min,
                spec.volume_step,
            )
            minimum_viable_notional = minimum_viable_volume * price_per_volume
            minimum_viable_risk = minimum_viable_notional * float(self.stop_loss_pct)
            adjusted_notional = minimum_viable_notional
            effective_risk = minimum_viable_risk

            if minimum_viable_risk > risk_amount + 1e-12 or bool(getattr(self.env_cfg, "block_trade_on_excess_risk", False)):
                adjusted_volume = 0.0
                blocked_by_min_notional = True
                skipped_due_to_min_notional_risk = minimum_viable_risk > risk_amount + 1e-12
            else:
                adjusted_volume = minimum_viable_volume
                bumped_to_min_notional = True

        if spec.volume_max > 0:
            adjusted_volume = min(adjusted_volume, spec.volume_max)
        if adjusted_volume < spec.volume_min:
            adjusted_volume = 0.0

        tradeable_notional = adjusted_volume * price_per_volume
        if adjusted_volume > 0:
            adjusted_notional = tradeable_notional
            effective_risk = tradeable_notional * float(self.stop_loss_pct)
        exceeds_target_risk = effective_risk > risk_amount if risk_amount > 0 else False

        return {
            "symbol": spec.symbol,
            "price": price,
            "balance": balance,
            "available_to_trade": float(available_to_trade) if available_to_trade is not None else None,
            "available_to_trade_source_field": (
                str(available_to_trade_source_field) if available_to_trade_source_field is not None else None
            ),
            "sizing_balance_used": float(balance),
            "sizing_balance_source": str(sizing_balance_source) if sizing_balance_source is not None else None,
            "risk_pct": risk_pct,
            "risk_amount": risk_amount,
            "stop_loss_pct": self.stop_loss_pct,
            "stop_distance_price": stop_distance_price,
            "target_notional": target_notional,
            "contract_size": spec.contract_size,
            "volume_min": spec.volume_min,
            "volume_step": spec.volume_step,
            "volume_max": spec.volume_max,
            "raw_volume": raw_volume,
            "adjusted_volume": adjusted_volume,
            "min_notional_usd": min_notional_usd,
            "adjusted_notional": adjusted_notional,
            "notional_value": tradeable_notional,
            "blocked_by_min_notional": blocked_by_min_notional,
            "bumped_to_min_notional": bumped_to_min_notional,
            "effective_risk_amount": effective_risk,
            "exceeds_target_risk": exceeds_target_risk,
            "skipped_due_to_min_notional_risk": skipped_due_to_min_notional_risk,
            "leverage_configured": None,
            "leverage_applied": None,
            "leverage_used": None,
            "margin_estimated_consumed": None,
            "has_explicit_leverage_logic": False,
        }

    def _coerce_raw_action(self, action) -> float:
        try:
            raw_action = float(action)
        except (TypeError, ValueError):
            raw_action = 0.0
        return max(-1.0, min(1.0, raw_action))

    def _action_direction(self, raw_action: float) -> str:
        threshold = float(self.env_cfg.action_hold_threshold)
        if raw_action > threshold:
            return "BUY"
        if raw_action < -threshold:
            return "SELL"
        return "HOLD"

    def _dynamic_risk_pct(self, raw_action: float) -> float:
        action_magnitude = abs(raw_action)
        dynamic_risk_pct = action_magnitude * float(self.env_cfg.max_risk_per_trade)
        if self._action_direction(raw_action) != "HOLD":
            dynamic_risk_pct = max(dynamic_risk_pct, float(self.env_cfg.min_risk_per_trade))
        return dynamic_risk_pct

    @staticmethod
    def _is_opposite_signal(trade_direction: str, side: int) -> bool:
        return (side == 1 and trade_direction == "SELL") or (side == -1 and trade_direction == "BUY")

    def _paper_close_position(self, price: float) -> dict:
        if self.paper_position == 0 or self.paper_volume <= 0:
            return {"closed": False, "reason": "no_open_position"}

        gross_pnl = (price - self.paper_entry_price) * self.paper_position * self._trade_units(self.paper_volume)
        close_cost = self._estimate_cost(price, self.paper_volume)
        pnl = gross_pnl - close_cost
        self.paper_balance += pnl
        snapshot = {
            "closed": True,
            "side": self.paper_position,
            "entry_price": self.paper_entry_price,
            "exit_price": price,
            "volume": self.paper_volume,
            "close_cost": close_cost,
            "gross_pnl": gross_pnl,
            "pnl": pnl,
        }
        self.paper_position = 0
        self.paper_entry_price = 0.0
        self.paper_volume = 0.0
        self.paper_entry_opened_at = None
        self.paper_entry_distance_from_ema_240 = 0.0
        self._set_trade_cooldown("paper")
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
        if self._execution_mode() == "paper_local":
            unrealized_pnl = 0.0
            if self.paper_position != 0 and self.paper_entry_price > 0:
                current_price = self._get_mid_price()
                unrealized_pnl = (
                    (current_price - self.paper_entry_price)
                    * self.paper_position
                    * self._trade_units(self.paper_volume)
                )
            return {
                "symbol": symbol,
                "side": self.paper_position,
                "volume": self.paper_volume,
                "avg_entry_price": self.paper_entry_price,
                "unrealized_pnl": unrealized_pnl,
                "time_in_position": self._position_bars(self.paper_entry_opened_at),
                "entry_distance_from_ema_240": float(self.paper_entry_distance_from_ema_240),
            }

        if not self.address:
            return {
                "symbol": symbol,
                "side": 0,
                "volume": 0.0,
                "avg_entry_price": 0.0,
                "unrealized_pnl": 0.0,
                "time_in_position": 0.0,
                "entry_distance_from_ema_240": 0.0,
            }

        user_state = self._call_with_retry(
            "position_snapshot_user_state",
            lambda: self._ensure_info().user_state(self.address),
        )
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
                "time_in_position": self._position_bars(self.live_entry_opened_at),
                "entry_distance_from_ema_240": float(self.live_entry_distance_from_ema_240),
            }

        self.live_entry_opened_at = None
        self.live_entry_distance_from_ema_240 = 0.0
        return {
            "symbol": symbol,
            "side": 0,
            "volume": 0.0,
            "avg_entry_price": 0.0,
            "unrealized_pnl": 0.0,
            "time_in_position": 0.0,
            "entry_distance_from_ema_240": 0.0,
        }

    def _response_ok(self, response) -> bool:
        if not isinstance(response, dict) or response.get("status") != "ok":
            return False
        statuses = ((response.get("response") or {}).get("data") or {}).get("statuses")
        if isinstance(statuses, list) and statuses:
            return all(isinstance(status, dict) and not status.get("error") for status in statuses)
        return True

    def _close_live_position(self, snapshot: dict, price: float, timestamp: str, trigger: str) -> dict:
        if self.exchange is None:
            return {
                "ok": False,
                "mode": self._operational_mode_label(),
                "timestamp": timestamp,
                "error": "Hyperliquid exchange is not initialized for close.",
            }

        response = self._call_with_retry(
            "market_close",
            lambda: self.exchange.market_close(
                str(self.hl_cfg.symbol),
                sz=float(snapshot["volume"]),
                px=price,
                slippage=float(self.order_slippage),
            ),
        )
        self.last_order_ts = time.time()
        if self._response_ok(response):
            self.live_entry_opened_at = None
            self.live_entry_distance_from_ema_240 = 0.0
            self._set_trade_cooldown("live")
        return {
            "ok": self._response_ok(response),
            "mode": self._operational_mode_label(),
            "timestamp": timestamp,
            "trigger": trigger,
            "response": response,
            "position_snapshot": self.get_position_snapshot(),
        }

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

        return self._close_live_position(snapshot, current_price, timestamp, trigger)

    def run_smoke_test(self, side: str = "buy", wait_seconds: float = 3.0, order_slippage: float | None = None) -> dict:
        execution_mode = self._execution_mode()
        network = str(self.hl_cfg.network).lower().strip()
        if execution_mode != "exchange":
            raise RuntimeError("Smoke test requires execution_mode=exchange.")
        if not self.exec_cfg.allow_live_trading:
            raise RuntimeError("Smoke test requires allow_live_trading=true.")
        if self.exchange is None:
            raise RuntimeError("Smoke test requires an initialized Hyperliquid exchange client.")

        precheck = self.check_connection()
        precheck["sizing_context"] = self.get_sizing_balance_context("exchange")
        starting_position = self.get_position_snapshot()
        if int(starting_position.get("side", 0)) != 0:
            raise RuntimeError("Smoke test requires no open position before starting.")

        is_buy = str(side).lower().strip() != "sell"
        bar_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        reference_price = self._get_mid_price()
        regime_abs = max(float(self.env_cfg.regime_min_abs_dist_ema_240), 0.05)
        regime_vol = max(float(self.env_cfg.regime_min_vol_regime_z) + 0.1, 0.5)

        effective_order_slippage = float(self.order_slippage if order_slippage is None else order_slippage)
        previous_order_slippage = float(self.order_slippage)
        self.order_slippage = effective_order_slippage
        try:
            open_result = self.execute(
                ExecutionDecision(
                    action=1.0 if is_buy else -1.0,
                    reason="smoke_test_open",
                    entry_distance_from_ema_240=regime_abs,
                    regime_valid_for_entry=True,
                    regime_dist_ema_240=regime_abs,
                    regime_vol_regime_z=regime_vol,
                    reference_price=reference_price,
                    bar_timestamp=bar_timestamp,
                )
            )

            snapshot_after_open = self.get_position_snapshot()
            close_result = None
            final_snapshot = snapshot_after_open
            opened_position = int(snapshot_after_open.get("side", 0)) != 0
            closed_position = False
            if opened_position:
                time.sleep(max(0.0, float(wait_seconds)))
                close_result = self._close_live_position(
                    snapshot_after_open,
                    self._get_mid_price(),
                    datetime.now(timezone.utc).isoformat(),
                    "smoke_test_manual_close",
                )
                final_snapshot = self.get_position_snapshot()
                closed_position = int(final_snapshot.get("side", 0)) == 0
        finally:
            self.order_slippage = previous_order_slippage

        return {
            "ok": bool(open_result.get("ok", False)) and opened_position and (close_result is not None) and bool(
                close_result.get("ok", False)
            ) and closed_position,
            "network": network,
            "execution_mode": execution_mode,
            "base_url": self._base_url(),
            "side": "buy" if is_buy else "sell",
            "wait_seconds": float(wait_seconds),
            "smoke_slippage_used": float(effective_order_slippage),
            "opened_position": opened_position,
            "closed_position": closed_position,
            "precheck": precheck,
            "open_result": open_result,
            "snapshot_after_open": snapshot_after_open,
            "close_result": close_result,
            "final_snapshot": final_snapshot,
        }

    def execute(self, decision: ExecutionDecision) -> dict:
        """Execute a continuous action in [-1, 1]."""
        timestamp = datetime.now(timezone.utc).isoformat()
        raw_action = self._coerce_raw_action(decision.action)
        trade_direction = self._action_direction(raw_action)
        dynamic_risk_pct = self._dynamic_risk_pct(raw_action)
        regime_valid = self._decision_regime_valid(decision)
        reference_price = (
            float(decision.reference_price)
            if float(decision.reference_price) > 0
            else self._get_mid_price()
        )

        def enrich(payload: dict) -> dict:
            payload.setdefault("mode", self._operational_mode_label())
            payload.setdefault("execution_mode", self._execution_mode())
            payload.setdefault("network", str(self.hl_cfg.network).lower())
            payload.setdefault("base_url", self._base_url())
            payload.setdefault("timestamp", timestamp)
            payload.setdefault("action", raw_action)
            payload.setdefault("trade_direction", trade_direction)
            payload.setdefault("dynamic_risk_pct", dynamic_risk_pct)
            payload.setdefault("reason", decision.reason)
            payload.setdefault("decision_mode", str(self.exec_cfg.decision_mode))
            payload.setdefault("reference_price", reference_price)
            payload.setdefault("bar_timestamp", str(decision.bar_timestamp))
            payload.setdefault("regime_valid_for_entry", bool(regime_valid))
            payload.setdefault("regime_dist_ema_240", float(decision.regime_dist_ema_240))
            payload.setdefault("regime_vol_regime_z", float(decision.regime_vol_regime_z))
            return payload

        def enrich_with_sizing(
            payload: dict,
            sizing: Optional[dict] = None,
            balance_context: Optional[dict] = None,
        ) -> dict:
            balance_context = balance_context or self.get_sizing_balance_context(self._execution_mode())
            payload.setdefault("available_to_trade", float(balance_context.get("available_to_trade", 0.0)))
            payload.setdefault(
                "available_to_trade_source_field",
                balance_context.get("available_to_trade_source_field"),
            )
            payload.setdefault("sizing_balance_used", float(balance_context.get("sizing_balance_used", 0.0)))
            payload.setdefault("sizing_balance_source", balance_context.get("sizing_balance_source"))
            payload.setdefault("leverage_configured", balance_context.get("leverage_configured"))
            payload.setdefault("leverage_applied", balance_context.get("leverage_applied"))
            payload.setdefault("leverage_used", balance_context.get("leverage_used"))
            payload.setdefault("margin_estimated_consumed", balance_context.get("margin_estimated_consumed"))
            payload.setdefault(
                "has_explicit_leverage_logic",
                bool(balance_context.get("has_explicit_leverage_logic", False)),
            )
            if sizing is not None:
                payload.setdefault("risk_pct", float(sizing.get("risk_pct", 0.0)))
                payload.setdefault("risk_amount", float(sizing.get("risk_amount", 0.0)))
                payload.setdefault("stop_loss_pct", float(sizing.get("stop_loss_pct", 0.0)))
                payload.setdefault("stop_distance_price", float(sizing.get("stop_distance_price", 0.0)))
                payload.setdefault("target_notional", float(sizing.get("target_notional", 0.0)))
                payload.setdefault("raw_volume", float(sizing.get("raw_volume", 0.0)))
                payload.setdefault("adjusted_volume", float(sizing.get("adjusted_volume", 0.0)))
                payload.setdefault("adjusted_notional", float(sizing.get("adjusted_notional", 0.0)))
                payload.setdefault("notional_value", float(sizing.get("notional_value", 0.0)))
                payload.setdefault("min_notional_usd", float(sizing.get("min_notional_usd", 0.0)))
                payload.setdefault(
                    "blocked_by_min_notional",
                    bool(sizing.get("blocked_by_min_notional", False)),
                )
                payload.setdefault(
                    "bumped_to_min_notional",
                    bool(sizing.get("bumped_to_min_notional", False)),
                )
                payload.setdefault(
                    "effective_risk_amount",
                    float(sizing.get("effective_risk_amount", 0.0)),
                )
                payload.setdefault(
                    "exceeds_target_risk",
                    bool(sizing.get("exceeds_target_risk", False)),
                )
                payload.setdefault(
                    "skipped_due_to_min_notional_risk",
                    bool(sizing.get("skipped_due_to_min_notional_risk", False)),
                )
            return enrich(payload)

        if self._execution_mode() == "paper_local":
            stop_take_event = self._paper_check_stop_take()
            if stop_take_event is not None:
                self._mark_order_sent_in_bar("paper", str(decision.bar_timestamp))
                return enrich({
                    "ok": True,
                    "volume": 0.0,
                    "sizing": None,
                    "entry_cost": 0.0,
                    "opened_position": False,
                    "open_price": None,
                    "exit_reason": stop_take_event.get("trigger"),
                    "paper_balance": self.paper_balance,
                    "position_snapshot": self.get_position_snapshot(),
                    "stop_take_event": stop_take_event,
                    "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                })

            current_snapshot = self.get_position_snapshot()
            current_side = int(self.paper_position)
            attempted_reversal = self._is_opposite_signal(trade_direction, current_side)
            current_stop, current_take = self._stop_take_prices(
                float(current_snapshot.get("avg_entry_price", 0.0)),
                current_side,
            )
            if current_side != 0:
                if bool(self.env_cfg.force_exit_only_by_tp_sl):
                    return enrich({
                        "ok": True,
                        "message": "Position locked until TP/SL, margin call, or end of episode.",
                        "ignored_signal": trade_direction != "HOLD",
                        "attempted_reversal": attempted_reversal,
                        "prevented_same_candle_reversal": attempted_reversal,
                        "blocked_reason": "open_position_locked",
                        "position_size": float(current_snapshot.get("volume", 0.0)),
                        "stop_loss_price": current_stop,
                        "take_profit_price": current_take,
                        "paper_balance": self.paper_balance,
                        "position_snapshot": current_snapshot,
                        "stop_take_event": None,
                        "force_exit_only_by_tp_sl": True,
                    })

                if attempted_reversal:
                    close_event = self._paper_close_position(self._get_mid_price())
                    self._mark_order_sent_in_bar("paper", str(decision.bar_timestamp))
                    close_event["trigger"] = "agent_close"
                    return enrich({
                        "ok": True,
                        "opened_position": False,
                        "closed_position": bool(close_event.get("closed", False)),
                        "attempted_reversal": True,
                        "prevented_same_candle_reversal": True,
                        "blocked_reason": "prevented_same_candle_reversal",
                        "position_size": 0.0,
                        "close_event": close_event,
                        "exit_reason": close_event.get("trigger"),
                        "paper_balance": self.paper_balance,
                        "position_snapshot": self.get_position_snapshot(),
                        "stop_take_event": None,
                        "force_exit_only_by_tp_sl": False,
                    })

                return enrich({
                    "ok": True,
                    "message": "Position maintained on this bar.",
                    "ignored_signal": trade_direction != "HOLD",
                    "attempted_reversal": False,
                    "prevented_same_candle_reversal": False,
                    "blocked_reason": "open_position_locked" if trade_direction != "HOLD" else None,
                    "position_size": float(current_snapshot.get("volume", 0.0)),
                    "stop_loss_price": current_stop,
                    "take_profit_price": current_take,
                    "paper_balance": self.paper_balance,
                    "position_snapshot": current_snapshot,
                    "stop_take_event": None,
                    "force_exit_only_by_tp_sl": False,
                })

            if trade_direction != "HOLD" and self._is_trade_cooldown_active("paper"):
                return enrich({
                    "ok": False,
                    "error": "Waiting for trade cooldown after the previous exit.",
                    "blocked_reason": "cooldown",
                    "cooldown_remaining_seconds": self._trade_cooldown_remaining_seconds("paper"),
                    "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                })

            if trade_direction != "HOLD" and self._has_sent_order_in_bar("paper", str(decision.bar_timestamp)):
                return enrich({
                    "ok": False,
                    "error": "Order already processed on this bar.",
                    "blocked_reason": "duplicate_cycle",
                    "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                })

            if trade_direction != "HOLD" and not regime_valid:
                return enrich_with_sizing({
                    "ok": True,
                    "message": "Entry blocked by regime filter.",
                    "forced_hold_by_regime_filter": True,
                    "blocked_reason": "regime_filter",
                    "opened_position": False,
                    "position_size": 0.0,
                    "paper_balance": self.paper_balance,
                    "position_snapshot": current_snapshot,
                    "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                })

            sizing = None
            paper_volume = 0.0
            sizing_context = self.get_sizing_balance_context("paper_local")
            if trade_direction != "HOLD":
                sizing = self.calculate_order_volume(
                    balance=float(sizing_context["sizing_balance_used"]),
                    dynamic_risk_pct=dynamic_risk_pct,
                    price=reference_price,
                    defensive=False,
                    available_to_trade=float(sizing_context["available_to_trade"]),
                    available_to_trade_source_field=str(sizing_context["available_to_trade_source_field"]),
                    sizing_balance_source=str(sizing_context["sizing_balance_source"]),
                )
                paper_volume = float(sizing["adjusted_volume"])
                if paper_volume <= 0:
                    blocked_reason = (
                        "min_notional_risk"
                        if bool(sizing.get("skipped_due_to_min_notional_risk"))
                        else ("min_notional" if bool(sizing.get("blocked_by_min_notional")) else "volume_min")
                    )
                    return enrich_with_sizing({
                        "ok": False,
                        "error": "Calculated entry volume is invalid.",
                        "blocked_reason": blocked_reason,
                        "sizing": sizing,
                        "position_snapshot": current_snapshot,
                        "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                    }, sizing=sizing, balance_context=sizing_context)

            opened = False
            open_price = None
            entry_cost = 0.0
            stop_loss_price = None
            take_profit_price = None
            if self.paper_position == 0 and trade_direction in ("BUY", "SELL") and paper_volume > 0:
                self.paper_position = 1 if trade_direction == "BUY" else -1
                self.paper_entry_price = reference_price
                self.paper_volume = paper_volume
                self.paper_entry_opened_at = time.time()
                self.paper_entry_distance_from_ema_240 = float(decision.entry_distance_from_ema_240)
                entry_cost = self._estimate_cost(self.paper_entry_price, self.paper_volume)
                self.paper_balance -= entry_cost
                self.session_peak_balance = max(float(self.session_peak_balance), float(self.paper_balance))
                opened = True
                open_price = self.paper_entry_price
                stop_loss_price, take_profit_price = self._stop_take_prices(self.paper_entry_price, self.paper_position)
                self._mark_order_sent_in_bar("paper", str(decision.bar_timestamp))

            return enrich_with_sizing({
                "ok": True,
                "volume": paper_volume,
                "position_size": paper_volume,
                "sizing": sizing,
                "entry_cost": entry_cost,
                "opened_position": opened,
                "open_price": open_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "paper_balance": self.paper_balance,
                "position_snapshot": self.get_position_snapshot(),
                "stop_take_event": stop_take_event,
                "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
            }, sizing=sizing, balance_context=sizing_context if trade_direction != "HOLD" else None)

        if not self.exec_cfg.allow_live_trading:
            return enrich({
                "ok": False,
                "error": "allow_live_trading=false. Real execution is blocked.",
            })

        if self.exchange is None:
            return enrich({
                "ok": False,
                "error": "HL_PRIVATE_KEY missing or Hyperliquid executor not initialized.",
            })

        stop_take_result = self._maybe_close_live_position(timestamp)
        if stop_take_result is not None:
            self._mark_order_sent_in_bar("live", str(decision.bar_timestamp))
            return enrich(stop_take_result)

        current_snapshot = self.get_position_snapshot()
        current_side = int(current_snapshot.get("side", 0))
        current_stop, current_take = self._stop_take_prices(
            float(current_snapshot.get("avg_entry_price", 0.0)),
            current_side,
        )
        if current_snapshot["side"] != 0:
            attempted_reversal = self._is_opposite_signal(trade_direction, current_side)
            if bool(self.env_cfg.force_exit_only_by_tp_sl):
                return enrich({
                    "ok": True,
                    "message": "Position locked until stop/take, margin call, or end of episode.",
                    "ignored_signal": trade_direction != "HOLD",
                    "attempted_reversal": attempted_reversal,
                    "prevented_same_candle_reversal": attempted_reversal,
                    "blocked_reason": "open_position_locked",
                    "position_size": float(current_snapshot.get("volume", 0.0)),
                    "stop_loss_price": current_stop,
                    "take_profit_price": current_take,
                    "position_snapshot": current_snapshot,
                    "force_exit_only_by_tp_sl": True,
                })

            if attempted_reversal:
                if not self._can_send_now():
                    return enrich({
                        "ok": False,
                        "error": "Waiting for cooldown between orders.",
                        "blocked_reason": "order_cooldown",
                        "attempted_reversal": True,
                        "position_snapshot": current_snapshot,
                    })
                if self._has_sent_order_in_bar("live", str(decision.bar_timestamp)):
                    return enrich({
                        "ok": False,
                        "error": "Order already processed on this bar.",
                        "blocked_reason": "duplicate_cycle",
                        "attempted_reversal": True,
                        "position_snapshot": current_snapshot,
                    })
                result = self._close_live_position(current_snapshot, self._get_mid_price(), timestamp, "agent_close")
                self._mark_order_sent_in_bar("live", str(decision.bar_timestamp))
                result["attempted_reversal"] = True
                result["prevented_same_candle_reversal"] = True
                result["force_exit_only_by_tp_sl"] = False
                return enrich(result)

            return enrich({
                "ok": True,
                "message": "Position maintained on this bar.",
                "blocked_reason": "open_position_locked" if trade_direction != "HOLD" else None,
                "position_size": float(current_snapshot.get("volume", 0.0)),
                "stop_loss_price": current_stop,
                "take_profit_price": current_take,
                "position_snapshot": current_snapshot,
                "force_exit_only_by_tp_sl": False,
            })

        if trade_direction != "HOLD" and self._is_trade_cooldown_active("live"):
            return enrich({
                "ok": False,
                "error": "Waiting for trade cooldown after the previous exit.",
                "blocked_reason": "cooldown",
                "cooldown_remaining_seconds": self._trade_cooldown_remaining_seconds("live"),
                "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
            })

        if trade_direction != "HOLD" and self._has_sent_order_in_bar("live", str(decision.bar_timestamp)):
            return enrich({
                "ok": False,
                "error": "Order already processed on this bar.",
                "blocked_reason": "duplicate_cycle",
                "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
            })

        if trade_direction != "HOLD" and not regime_valid:
            return enrich_with_sizing({
                "ok": True,
                "message": "Entry blocked by regime filter.",
                "forced_hold_by_regime_filter": True,
                "blocked_reason": "regime_filter",
                "opened_position": False,
                "position_size": 0.0,
                "position_snapshot": current_snapshot,
                "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
            })

        if not self._can_send_now():
            return enrich({
                "ok": False,
                "error": "Waiting for cooldown between orders.",
                "blocked_reason": "order_cooldown",
            })

        if trade_direction == "HOLD":
            return enrich({
                "ok": True,
                "message": "No entry signal on this bar.",
                "position_snapshot": current_snapshot,
            })

        risk_state = self.get_account_risk_state()
        sizing_context = self.get_sizing_balance_context("exchange")
        price = reference_price
        sizing = self.calculate_order_volume(
            balance=float(sizing_context["sizing_balance_used"]),
            dynamic_risk_pct=dynamic_risk_pct,
            price=price,
            defensive=bool(risk_state["defensive"]),
            available_to_trade=float(sizing_context["available_to_trade"]),
            available_to_trade_source_field=str(sizing_context["available_to_trade_source_field"]),
            sizing_balance_source=str(sizing_context["sizing_balance_source"]),
        )
        volume = float(sizing["adjusted_volume"])

        if volume <= 0:
            blocked_reason = (
                "min_notional_risk"
                if bool(sizing.get("skipped_due_to_min_notional_risk"))
                else ("min_notional" if bool(sizing.get("blocked_by_min_notional")) else "volume_min")
            )
            return enrich_with_sizing({
                "ok": False,
                "error": "Calculated entry volume is invalid.",
                "blocked_reason": blocked_reason,
                "sizing": sizing,
                "risk_state": risk_state,
            }, sizing=sizing, balance_context=sizing_context)

        response = self._call_with_retry(
            "market_open",
            lambda: self.exchange.market_open(
                str(self.hl_cfg.symbol),
                trade_direction == "BUY",
                volume,
                px=price,
                slippage=float(self.order_slippage),
            ),
        )
        self.last_order_ts = time.time()
        if self._response_ok(response):
            self.live_entry_opened_at = time.time()
            self.live_entry_distance_from_ema_240 = float(decision.entry_distance_from_ema_240)
            self._mark_order_sent_in_bar("live", str(decision.bar_timestamp))
        stop_loss_price, take_profit_price = self._stop_take_prices(
            price,
            1 if trade_direction == "BUY" else -1,
        )

        return enrich_with_sizing({
            "ok": self._response_ok(response),
            "volume": volume,
            "position_size": volume,
            "sizing": sizing,
            "risk_state": risk_state,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "response": response,
            "position_snapshot": self.get_position_snapshot(),
        }, sizing=sizing, balance_context=sizing_context)
