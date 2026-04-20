from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor
from typing import Callable, Optional, TypeVar

import pandas as pd
import requests

try:
    from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

    TENACITY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    RetryCallState = object  # type: ignore[assignment]
    Retrying = None  # type: ignore[assignment]
    retry_if_exception = None  # type: ignore[assignment]
    stop_after_attempt = None  # type: ignore[assignment]
    wait_exponential = None  # type: ignore[assignment]
    TENACITY_AVAILABLE = False

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


T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


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
    RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524})
    RETRYABLE_ERROR_MARKERS = (
        "rate limit",
        "too many requests",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
    )

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
            self._call_with_retry("loader_connect", self.loader.connect)
            self.info = self.loader.info
        return self.info

    @classmethod
    def _extract_retryable_status_code(cls, exc: BaseException) -> int | None:
        for candidate in (
            getattr(exc, "status_code", None),
            getattr(getattr(exc, "response", None), "status_code", None),
        ):
            try:
                if candidate is not None:
                    return int(candidate)
            except (TypeError, ValueError):
                continue

        match = re.search(
            r"\b(408|425|429|500|502|503|504|520|522|524)\b",
            str(exc),
        )
        return int(match.group(1)) if match else None

    @classmethod
    def _is_retryable_network_error(cls, exc: BaseException) -> bool:
        status_code = cls._extract_retryable_status_code(exc)
        if status_code is not None:
            return status_code in cls.RETRYABLE_HTTP_STATUS_CODES

        if isinstance(
            exc,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
                TimeoutError,
                ConnectionError,
            ),
        ):
            return True

        lowered = str(exc).lower()
        return any(marker in lowered for marker in cls.RETRYABLE_ERROR_MARKERS)

    def _log_retry_attempt(self, label: str, retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            return

        exc = retry_state.outcome.exception()
        if exc is None:
            return

        sleep_seconds = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
        attempts = max(1, int(getattr(self.exec_cfg, "api_retry_attempts", 1)))
        LOGGER.warning(
            "API transient failure on %s | attempt=%s/%s | retry_in=%.1fs | error=%s: %s",
            label,
            retry_state.attempt_number,
            attempts,
            sleep_seconds,
            type(exc).__name__,
            exc,
        )

    def _call_with_retry(self, label: str, fn: Callable[[], T]) -> T:
        attempts = max(1, int(getattr(self.exec_cfg, "api_retry_attempts", 1)))
        base_delay = max(1.0, float(getattr(self.exec_cfg, "api_retry_delay_seconds", 1.0)))
        max_delay = max(
            base_delay,
            float(getattr(self.exec_cfg, "api_retry_max_delay_seconds", max(base_delay * 4.0, 8.0))),
        )

        if not TENACITY_AVAILABLE:
            return self._call_with_retry_fallback(
                label=label,
                fn=fn,
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
            )

        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
            retry=retry_if_exception(self._is_retryable_network_error),
            before_sleep=lambda retry_state: self._log_retry_attempt(label, retry_state),
        )

        try:
            return retrying(fn)
        except Exception as exc:  # pragma: no cover - depends on runtime API failures
            if self._is_retryable_network_error(exc):
                raise RuntimeError(
                    f"{label} failed after {attempts} attempt(s) with exponential backoff: {exc}"
                ) from exc
            raise RuntimeError(f"{label} failed without retry because the error is not transient: {exc}") from exc

    def _call_with_retry_fallback(
        self,
        *,
        label: str,
        fn: Callable[[], T],
        attempts: int,
        base_delay: float,
        max_delay: float,
    ) -> T:
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - depends on runtime API failures
                last_exc = exc
                if not self._is_retryable_network_error(exc):
                    raise RuntimeError(
                        f"{label} failed without retry because the error is not transient: {exc}"
                    ) from exc

                if attempt >= attempts:
                    break

                sleep_seconds = min(base_delay * (2 ** (attempt - 1)), max_delay)
                LOGGER.warning(
                    "API transient failure on %s | attempt=%s/%s | retry_in=%.1fs | backend=fallback | error=%s: %s",
                    label,
                    attempt,
                    attempts,
                    sleep_seconds,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"{label} failed after {attempts} attempt(s) with exponential backoff: {last_exc}"
        ) from last_exc

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

        execution_mode = self._execution_mode()
        self._call_with_retry("loader_connect", self.loader.connect)
        self.info = self.loader.info
        self.paper_balance = float(self.env_cfg.initial_balance)
        self.session_peak_balance = (
            float(self.env_cfg.initial_balance) if execution_mode == "paper_local" else 0.0
        )
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
                    exchange_kwargs["meta"] = self._call_with_retry(
                        "exchange_meta",
                        lambda: self.loader._info_post(base_url, {"type": "meta"}),
                    )
                    exchange_kwargs["spot_meta"] = self.loader._normalize_spot_meta(
                        self._call_with_retry(
                            "exchange_spot_meta",
                            lambda: self.loader._info_post(base_url, {"type": "spotMeta"}),
                        )
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
        status = self._call_with_retry("check_connection", self.loader.check_connection)
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
        try:
            position_snapshot = self.get_position_snapshot()
            position_side = int(position_snapshot.get("side", 0) or 0)
            position_volume = float(position_snapshot.get("volume", 0.0) or 0.0)
            position_entry_price = float(position_snapshot.get("avg_entry_price", 0.0) or 0.0)
            position_notional_value = 0.0
            if position_side != 0 and position_volume > 0 and position_entry_price > 0:
                position_notional_value = position_entry_price * self._trade_units(position_volume)

            status["position_snapshot"] = position_snapshot
            status["position_is_open"] = position_side != 0
            status["position_label"] = "LONG" if position_side > 0 else ("SHORT" if position_side < 0 else "FLAT")
            status["position_size"] = position_volume
            status["position_avg_entry_price"] = position_entry_price
            status["position_unrealized_pnl"] = float(position_snapshot.get("unrealized_pnl", 0.0) or 0.0)
            status["position_notional_value"] = float(position_notional_value)

            native_tp_sl = self.get_native_tp_sl_status(position_snapshot)
            status["native_tp_sl"] = native_tp_sl
            status["native_tp_sl_protected"] = bool(native_tp_sl.get("is_fully_protected", False))
            status["stop_loss_price"] = float(
                native_tp_sl.get("stop_loss_price")
                or native_tp_sl.get("expected_stop_loss_price")
                or 0.0
            )
            status["take_profit_price"] = float(
                native_tp_sl.get("take_profit_price")
                or native_tp_sl.get("expected_take_profit_price")
                or 0.0
            )
        except Exception as exc:  # pragma: no cover
            status["position_error"] = str(exc)
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
        if float(self.session_peak_balance) <= 0:
            self.session_peak_balance = float(balance)
        else:
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
            
        raw_stop = entry_price * (1.0 - self.stop_loss_pct * side)
        raw_take = entry_price * (1.0 + self.take_profit_pct * side)
        
        return round(raw_stop, 6), round(raw_take, 6)
    
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
        desired_risk_pct = float(dynamic_risk_pct)
        hard_cap_risk_pct = float(self.env_cfg.max_risk_per_trade)
        if defensive:
            defensive_multiplier = float(self.exec_cfg.defensive_risk_multiplier)
            desired_risk_pct *= defensive_multiplier
            hard_cap_risk_pct *= defensive_multiplier

        risk_amount = balance * desired_risk_pct
        hard_cap_risk_amount = balance * hard_cap_risk_pct
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

            if (
                minimum_viable_risk > hard_cap_risk_amount + 1e-12
                or bool(getattr(self.env_cfg, "block_trade_on_excess_risk", False))
            ):
                adjusted_volume = 0.0
                blocked_by_min_notional = True
                skipped_due_to_min_notional_risk = minimum_viable_risk > hard_cap_risk_amount + 1e-12
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
            "risk_pct": desired_risk_pct,
            "risk_amount": risk_amount,
            "desired_risk_pct": desired_risk_pct,
            "desired_risk_amount": risk_amount,
            "hard_cap_risk_pct": hard_cap_risk_pct,
            "hard_cap_risk_amount": hard_cap_risk_amount,
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
            "within_hard_cap": effective_risk <= hard_cap_risk_amount + 1e-12 if hard_cap_risk_amount > 0 else False,
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
        return action_magnitude * float(self.env_cfg.max_risk_per_trade)

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

    def _native_tp_sl_enabled(self) -> bool:
        return self._execution_mode() == "exchange" and bool(
            getattr(self.exec_cfg, "exchange_native_tp_sl_enabled", True)
        )

    def _symbol_dex(self) -> str:
        symbol = str(self.hl_cfg.symbol).strip()
        return symbol.split(":", 1)[0] if ":" in symbol else ""

    @staticmethod
    def _normalized_coin_name(value: object) -> str:
        text = str(value or "").strip().upper()
        return text.split(":")[-1] if text else ""

    def _order_matches_symbol(self, order: dict) -> bool:
        return self._normalized_coin_name(order.get("coin")) == self._normalized_coin_name(self.hl_cfg.symbol)

    @staticmethod
    def _is_trigger_order(order: dict) -> bool:
        if bool(order.get("isTrigger", False)):
            return True
        if float(order.get("triggerPx", 0.0) or 0.0) > 0:
            return True

        for key in ("orderType", "order_type", "type", "origType", "trigger", "triggerType"):
            value = order.get(key)
            if value and "trigger" in str(value).lower():
                return True
        return False

    def _frontend_open_orders(self) -> list[dict]:
        if not self.address:
            return []
        payload = self._call_with_retry(
            "frontend_open_orders",
            lambda: self._ensure_info().frontend_open_orders(self.address, self._symbol_dex()),
        )
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    def _native_tp_sl_orders(self) -> list[dict]:
        orders: list[dict] = []
        for order in self._frontend_open_orders():
            if not self._order_matches_symbol(order):
                continue
            if not bool(order.get("reduceOnly", False)):
                continue
            if not self._is_trigger_order(order):
                continue
            orders.append(order)
        return orders

    def _classify_native_tp_sl_order(self, order: dict, entry_price: float, side: int) -> str | None:
        trigger_px = float(order.get("triggerPx", 0.0) or 0.0)
        if trigger_px <= 0 or entry_price <= 0 or side == 0:
            return None
        if side > 0:
            if trigger_px < entry_price:
                return "sl"
            if trigger_px > entry_price:
                return "tp"
            return None
        if trigger_px > entry_price:
            return "sl"
        if trigger_px < entry_price:
            return "tp"
        return None

    def _price_tolerance(self) -> float:
        spec = self.get_symbol_trading_spec()
        return max(float(spec.point), 1e-8)

    def _size_tolerance(self) -> float:
        spec = self.get_symbol_trading_spec()
        return max(float(spec.volume_step), 1e-12)

    def get_native_tp_sl_status(self, snapshot: dict | None = None) -> dict:
        snapshot = dict(snapshot or self.get_position_snapshot())
        side = int(snapshot.get("side", 0) or 0)
        entry_price = float(snapshot.get("avg_entry_price", 0.0) or 0.0)
        volume = float(snapshot.get("volume", 0.0) or 0.0)
        expected_stop, expected_take = self._stop_take_prices(entry_price, side)
        orders = self._native_tp_sl_orders() if self._native_tp_sl_enabled() else []

        tp_orders: list[dict] = []
        sl_orders: list[dict] = []
        for order in orders:
            classification = self._classify_native_tp_sl_order(order, entry_price, side)
            if classification == "tp":
                tp_orders.append(order)
            elif classification == "sl":
                sl_orders.append(order)

        def _pick_trigger_price(candidates: list[dict]) -> float | None:
            if len(candidates) != 1:
                return None
            trigger_px = float(candidates[0].get("triggerPx", 0.0) or 0.0)
            return trigger_px if trigger_px > 0 else None

        def _pick_size(candidates: list[dict]) -> float | None:
            if len(candidates) != 1:
                return None
            raw = candidates[0].get("origSz", candidates[0].get("sz", 0.0))
            size = float(raw or 0.0)
            return size if size > 0 else None

        stop_loss_price = _pick_trigger_price(sl_orders)
        take_profit_price = _pick_trigger_price(tp_orders)
        stop_loss_size = _pick_size(sl_orders)
        take_profit_size = _pick_size(tp_orders)
        price_tol = self._price_tolerance()
        size_tol = self._size_tolerance()

        has_stop = stop_loss_price is not None
        has_take = take_profit_price is not None
        stop_matches = (
            has_stop
            and expected_stop is not None
            and abs(float(stop_loss_price) - float(expected_stop)) <= price_tol
            and stop_loss_size is not None
            and abs(float(stop_loss_size) - float(volume)) <= size_tol
        )
        take_matches = (
            has_take
            and expected_take is not None
            and abs(float(take_profit_price) - float(expected_take)) <= price_tol
            and take_profit_size is not None
            and abs(float(take_profit_size) - float(volume)) <= size_tol
        )
        is_fully_protected = side == 0 or (stop_matches and take_matches and len(sl_orders) == 1 and len(tp_orders) == 1)

        return {
            "enabled": self._native_tp_sl_enabled(),
            "has_position": side != 0,
            "position_side": side,
            "position_volume": volume,
            "expected_stop_loss_price": expected_stop,
            "expected_take_profit_price": expected_take,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "stop_loss_orders": sl_orders,
            "take_profit_orders": tp_orders,
            "stop_loss_size": stop_loss_size,
            "take_profit_size": take_profit_size,
            "has_stop_loss": has_stop,
            "has_take_profit": has_take,
            "stop_matches_expected": stop_matches,
            "take_matches_expected": take_matches,
            "is_fully_protected": is_fully_protected,
            "open_orders_count": len(orders),
            "open_orders": orders,
        }

    def _cancel_native_tp_sl_orders(self, orders: list[dict] | None = None) -> dict:
        if self.exchange is None:
            return {
                "ok": False,
                "canceled": 0,
                "error": "Hyperliquid exchange is not initialized for TP/SL cancelation.",
            }

        candidates = orders if orders is not None else self._native_tp_sl_orders()
        cancel_requests = []
        for order in candidates:
            oid = int(order.get("oid", 0) or 0)
            if oid <= 0:
                continue
            cancel_requests.append({"coin": str(order.get("coin") or self.hl_cfg.symbol), "oid": oid})

        if not cancel_requests:
            return {"ok": True, "canceled": 0, "response": None}

        response = self._call_with_retry(
            "cancel_native_tp_sl_orders",
            lambda: self.exchange.bulk_cancel(cancel_requests),
        )
        return {
            "ok": self._response_ok(response),
            "canceled": len(cancel_requests),
            "response": response,
        }
        
    def _slippage_price(self, symbol: str, is_buy: bool, slippage: float, price: float) -> float:
        """Calcula o preço com slippage aplicado para ordens de limite agressivas."""
        if is_buy:
            return price * (1.0 + slippage)
        return price * (1.0 - slippage)

    def _build_native_tp_sl_orders(self, snapshot: dict) -> tuple[list[dict], float | None, float | None]:
        side = int(snapshot.get("side", 0) or 0)
        entry_price = float(snapshot.get("avg_entry_price", 0.0) or 0.0)
        volume = float(snapshot.get("volume", 0.0) or 0.0)
        stop_loss_price, take_profit_price = self._stop_take_prices(entry_price, side)
        if side == 0 or entry_price <= 0 or volume <= 0 or stop_loss_price is None or take_profit_price is None:
            return [], stop_loss_price, take_profit_price

        close_is_buy = side < 0
        sl_limit_px = self._slippage_price(str(self.hl_cfg.symbol), close_is_buy, 0.10, stop_loss_price)
        tp_limit_px = self._slippage_price(str(self.hl_cfg.symbol), close_is_buy, 0.10, take_profit_price)
        orders = [
            {
                "coin": str(self.hl_cfg.symbol),
                "is_buy": close_is_buy,
                "sz": volume,
                "limit_px": float(sl_limit_px),
                "order_type": {
                    "trigger": {
                        "triggerPx": float(stop_loss_price),
                        "isMarket": True,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            },
            {
                "coin": str(self.hl_cfg.symbol),
                "is_buy": close_is_buy,
                "sz": volume,
                "limit_px": float(tp_limit_px),
                "order_type": {
                    "trigger": {
                        "triggerPx": float(take_profit_price),
                        "isMarket": True,
                        "tpsl": "tp",
                    }
                },
                "reduce_only": True,
            },
        ]
        return orders, stop_loss_price, take_profit_price

    def ensure_native_tp_sl(
        self,
        snapshot: dict | None = None,
        *,
        allow_existing_trigger_reconcile: bool = True,
    ) -> dict:
        snapshot = dict(snapshot or self.get_position_snapshot())
        protection = self.get_native_tp_sl_status(snapshot)
        if not protection["enabled"] or not protection["has_position"]:
            return {"ok": True, "changed": False, **protection}
        if protection["is_fully_protected"]:
            return {"ok": True, "changed": False, **protection}
        if not allow_existing_trigger_reconcile and int(protection.get("open_orders_count", 0) or 0) > 0:
            return {
                "ok": False,
                "changed": False,
                "preserved_existing_trigger_orders": True,
                "error": (
                    "Existing native trigger orders were detected and preserved during sync. "
                    "No automatic cancel/recreate was attempted."
                ),
                **protection,
            }
        if self.exchange is None:
            return {
                "ok": False,
                "changed": False,
                "error": "Hyperliquid exchange is not initialized for TP/SL placement.",
                **protection,
            }

        cancel_result = self._cancel_native_tp_sl_orders(protection.get("open_orders"))
        if not bool(cancel_result.get("ok", False)):
            return {
                "ok": False,
                "changed": False,
                "error": "Falha ao cancelar TP/SL nativos antigos antes de recriar a proteção.",
                "cancel_result": cancel_result,
                **protection,
            }

        order_requests, expected_stop, expected_take = self._build_native_tp_sl_orders(snapshot)
        if not order_requests:
            return {
                "ok": False,
                "changed": False,
                "error": "Nao foi possivel construir TP/SL nativos para a posicao aberta.",
                **protection,
            }

        response = self._call_with_retry(
            "place_native_tp_sl_orders",
            lambda: self.exchange.bulk_orders(order_requests, grouping="positionTpsl"),
        )
        updated = self.get_native_tp_sl_status(snapshot)
        return {
            "ok": self._response_ok(response) and bool(updated.get("is_fully_protected", False)),
            "changed": True,
            "response": response,
            "cancel_result": cancel_result,
            "expected_stop_loss_price": expected_stop,
            "expected_take_profit_price": expected_take,
            **updated,
        }

    def _close_live_position(self, snapshot: dict, price: float, timestamp: str, trigger: str) -> dict:
        if self.exchange is None:
            return {
                "ok": False,
                "mode": self._operational_mode_label(),
                "timestamp": timestamp,
                "error": "Hyperliquid exchange is not initialized for close.",
            }

        entry_price = float(snapshot.get("avg_entry_price", 0.0) or 0.0)
        volume = float(snapshot.get("volume", 0.0) or 0.0)
        side = int(snapshot.get("side", 0) or 0)
        gross_pnl = None
        estimated_close_cost = None
        estimated_pnl = None
        if side != 0 and entry_price > 0 and volume > 0:
            gross_pnl = (price - entry_price) * side * self._trade_units(volume)
            estimated_close_cost = self._estimate_cost(price, volume)
            estimated_pnl = gross_pnl - estimated_close_cost

        response = self._call_with_retry(
            "market_close",
            lambda: self.exchange.market_close(
                str(self.hl_cfg.symbol),
                sz=volume,
                px=price,
                slippage=float(self.order_slippage),
            ),
        )
        self.last_order_ts = time.time()
        if self._response_ok(response):
            self.live_entry_opened_at = None
            self.live_entry_distance_from_ema_240 = 0.0
            self._set_trade_cooldown("live")
        final_snapshot = self.get_position_snapshot()
        closed_position = bool(self._response_ok(response)) and int(final_snapshot.get("side", 0) or 0) == 0
        cleanup_result = None
        if closed_position and self._native_tp_sl_enabled():
            cleanup_result = self._cancel_native_tp_sl_orders()
        return {
            "ok": self._response_ok(response),
            "closed_position": closed_position,
            "mode": self._operational_mode_label(),
            "timestamp": timestamp,
            "trigger": trigger,
            "response": response,
            "native_tp_sl_cleanup": cleanup_result,
            "close_event": {
                "closed": closed_position,
                "side": side,
                "entry_price": entry_price,
                "exit_price": float(price),
                "volume": volume,
                "gross_pnl": gross_pnl,
                "close_cost": estimated_close_cost,
                "pnl": estimated_pnl,
                "trigger": trigger,
            },
            "position_snapshot": final_snapshot,
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

    def close_open_position(self, trigger: str = "manual_close") -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot = self.get_position_snapshot()
        if int(snapshot.get("side", 0)) == 0:
            return {
                "ok": False,
                "closed_position": False,
                "mode": self._operational_mode_label(),
                "timestamp": timestamp,
                "trigger": trigger,
                "message": "No open position to close.",
                "position_snapshot": snapshot,
            }

        current_price = self._get_mid_price()
        if self._execution_mode() == "paper_local":
            close_event = self._paper_close_position(current_price)
            return {
                "ok": bool(close_event.get("closed", False)),
                "closed_position": bool(close_event.get("closed", False)),
                "mode": self._operational_mode_label(),
                "timestamp": timestamp,
                "trigger": trigger,
                "close_event": close_event,
                "position_snapshot": self.get_position_snapshot(),
            }

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
                    "closed_position": bool(stop_take_event.get("closed", False)),
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
                        "ok": True,
                        "message": "Entry blocked because the minimum tradeable size would violate the allowed risk.",
                        "error": None,
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

        current_snapshot = self.get_position_snapshot()
        current_side = int(current_snapshot.get("side", 0))
        protection_status = self.get_native_tp_sl_status(current_snapshot)
        current_stop = protection_status.get("stop_loss_price") or protection_status.get("expected_stop_loss_price")
        current_take = protection_status.get("take_profit_price") or protection_status.get("expected_take_profit_price")
        if current_snapshot["side"] != 0:
            if self._native_tp_sl_enabled():
                protection_status = self.ensure_native_tp_sl(
                    current_snapshot,
                    allow_existing_trigger_reconcile=False,
                )
                current_stop = protection_status.get("stop_loss_price") or protection_status.get(
                    "expected_stop_loss_price"
                )
                current_take = protection_status.get("take_profit_price") or protection_status.get(
                    "expected_take_profit_price"
                )
                if not bool(protection_status.get("ok", False)):
                    return enrich({
                        "ok": False,
                        "error": "Open position detected without confirmed native TP/SL protection on Hyperliquid.",
                        "blocked_reason": "native_tp_sl_unprotected",
                        "position_size": float(current_snapshot.get("volume", 0.0)),
                        "stop_loss_price": current_stop,
                        "take_profit_price": current_take,
                        "position_snapshot": current_snapshot,
                        "native_tp_sl": protection_status,
                        "force_exit_only_by_tp_sl": bool(self.env_cfg.force_exit_only_by_tp_sl),
                    })

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
                    "native_tp_sl": protection_status,
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
                result["native_tp_sl"] = protection_status
                return enrich(result)

            return enrich({
                "ok": True,
                "message": "Position maintained on this bar.",
                "blocked_reason": "open_position_locked" if trade_direction != "HOLD" else None,
                "position_size": float(current_snapshot.get("volume", 0.0)),
                "stop_loss_price": current_stop,
                "take_profit_price": current_take,
                "position_snapshot": current_snapshot,
                "native_tp_sl": protection_status,
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
                "ok": True,
                "message": "Entry blocked because the minimum tradeable size would violate the allowed risk.",
                "error": None,
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

        # Calcula SL e TP antes para poder usar na ordem nativa
        stop_loss_price, take_profit_price = self._stop_take_prices(
            price,
            1 if trade_direction == "BUY" else -1,
        )

        if self._response_ok(response):
            self.live_entry_opened_at = time.time()
            self.live_entry_distance_from_ema_240 = float(decision.entry_distance_from_ema_240)
            self._mark_order_sent_in_bar("live", str(decision.bar_timestamp))
            
            # =========================================================
            # INÍCIO DO TP/SL NATIVO NA HYPERLIQUID
            # =========================================================
            close_is_buy = trade_direction == "SELL" # Direção oposta para fechar

            if stop_loss_price is not None and stop_loss_price > 0:
                try:
                    self._call_with_retry(
                        "place_native_stop_loss",
                        lambda: self.exchange.order(
                            str(self.hl_cfg.symbol),
                            close_is_buy,
                            volume,
                            float(stop_loss_price),
                            order_type={"trigger": {"triggerPx": float(stop_loss_price), "isMarket": True, "tpsl": "sl"}},
                            reduce_only=True,
                        ),
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to place native stop loss after retries: %s", exc)

            if take_profit_price is not None and take_profit_price > 0:
                try:
                    self._call_with_retry(
                        "place_native_take_profit",
                        lambda: self.exchange.order(
                            str(self.hl_cfg.symbol),
                            close_is_buy,
                            volume,
                            float(take_profit_price),
                            order_type={"trigger": {"triggerPx": float(take_profit_price), "isMarket": True, "tpsl": "tp"}},
                            reduce_only=True,
                        ),
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to place native take profit after retries: %s", exc)
            
            # =========================================================
            # FIM DO TP/SL NATIVO
            # =========================================================

        final_snapshot = self.get_position_snapshot()
        opened_position = bool(self._response_ok(response)) and int(final_snapshot.get("side", 0) or 0) != 0

        return enrich_with_sizing({
            "ok": self._response_ok(response),
            "volume": volume,
            "position_size": volume,
            "sizing": sizing,
            "risk_state": risk_state,
            "opened_position": opened_position,
            "open_price": float(price),
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "response": response,
            "position_snapshot": final_snapshot,
        }, sizing=sizing, balance_context=sizing_context)
