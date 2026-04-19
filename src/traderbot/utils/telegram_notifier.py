from __future__ import annotations

import logging
import os
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 10

_CLOSE_REASON_LABELS = {
    "agent_close": "Fechamento pelo agente",
    "manual_close": "Fechamento manual",
    "manual_close_from_cli": "Kill switch",
    "runtime_shutdown": "Encerramento do programa",
    "shutdown_after_error": "Encerramento por erro",
    "shutdown_signal": "Encerramento do programa",
    "smoke_test_manual_close": "Fechamento do smoke test",
    "stop_loss": "Stop Loss",
    "take_profit": "Take Profit",
}


class TelegramNotifier:
    """Envia alertas operacionais objetivos para um chat do Telegram."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        logger: logging.Logger | None = None,
    ) -> None:
        self.bot_token = (bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        self.timeout = int(timeout)
        self.logger = logger or logging.getLogger(__name__)

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def notify_startup(self) -> bool:
        return self._send_message("Online")

    def notify_order_executed(
        self,
        *,
        asset: str,
        side: str,
        price: float | int,
        quantity: float | int,
    ) -> bool:
        lines = [
            f"ENTRADA {self._normalize_entry_side_label(side)}",
            f"Quantidade: ${self._format_brl_number(quantity, places=2)}",
            f"Preço: ${self._format_brl_number(price, places=2)}",
        ]
        return self._send_message("\n".join(lines))

    def notify_position_closed(
        self,
        *,
        asset: str,
        pnl: float | int | None,
        side: str | None = None,
        reason: str | None = None,
        exit_price: float | int | None = None,
    ) -> bool:
        lines = [
            "POSIÇÃO FECHADA",
        ]

        reason_label = self._normalize_close_reason(reason)
        if reason_label:
            lines.append(f"Motivo: {reason_label}")

        if pnl not in (None, ""):
            lines.append(f"PnL realizado: ${self._format_decimal(pnl, signed=True, places=2, trim=False)}")

        return self._send_message("\n".join(lines))

    def notify_stopped(self, reason: str | None = None) -> bool:
        return self._send_message("Offline")

    def notify_critical_error(
        self,
        error_message: str | None = None,
        details: str | None = None,
        *,
        status: str | None = None,
    ) -> bool:
        return self._send_message("Erro\nStatus: Offline")

    def _send_message(self, text: str) -> bool:
        if not self.enabled:
            self.logger.warning(
                "TelegramNotifier desabilitado: defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID."
            )
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
        }

        try:
            response = requests.post(url, data=payload, timeout=self.timeout)
            response.raise_for_status()
            response_payload = response.json()
        except requests.RequestException as exc:
            self.logger.exception("Falha ao enviar notificacao para o Telegram: %s", exc)
            return False
        except ValueError:
            self.logger.exception("Resposta invalida ao enviar notificacao para o Telegram.")
            return False

        if not response_payload.get("ok", False):
            error_description = response_payload.get("description", "erro desconhecido")
            self.logger.error(
                "Telegram respondeu com falha ao enviar notificacao: %s",
                error_description,
            )
            return False

        return True

    @staticmethod
    def _normalize_side_label(side: str | None) -> str:
        label = str(side or "").strip().lower()
        if label == "buy":
            return "BUY"
        if label == "sell":
            return "SELL"
        return str(side or "").strip().upper() or "ORDEM"

    @staticmethod
    def _normalize_entry_side_label(side: str | None) -> str:
        label = str(side or "").strip().lower()
        if label in {"buy", "long"}:
            return "LONG"
        if label in {"sell", "short"}:
            return "SHORT"
        return str(side or "").strip().upper() or "ORDEM"

    @staticmethod
    def _normalize_position_side(side: str | None) -> str:
        label = str(side or "").strip().lower()
        if label == "long":
            return "LONG"
        if label == "short":
            return "SHORT"
        return str(side or "").strip().upper()

    @staticmethod
    def _normalize_close_reason(reason: str | None) -> str:
        raw_reason = str(reason or "").strip().lower()
        if not raw_reason:
            return ""
        return _CLOSE_REASON_LABELS.get(raw_reason, raw_reason.replace("_", " ").title())

    @staticmethod
    def _multiply_if_possible(left: float | int, right: float | int) -> float | None:
        try:
            return float(left) * float(right)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_runtime_status(status: str | None) -> str:
        label = str(status or "").strip().lower()
        if label == "online":
            return "Online"
        if label == "offline":
            return "Offline"
        return "Offline"

    @staticmethod
    def _format_brl_number(value: float | int, *, places: int = 2) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        rendered = f"{numeric_value:,.{places}f}"
        return rendered.replace(",", "_").replace(".", ",").replace("_", ".")

    @staticmethod
    def _format_decimal(
        value: float | int,
        *,
        signed: bool = False,
        places: int = 8,
        trim: bool = True,
    ) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        rendered = f"{numeric_value:+.{places}f}" if signed else f"{numeric_value:.{places}f}"
        if trim:
            return rendered.rstrip("0").rstrip(".")
        return rendered
