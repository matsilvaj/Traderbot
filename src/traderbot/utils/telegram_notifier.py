from __future__ import annotations

import logging
import os
from typing import Any

import requests

DEFAULT_TIMEOUT_SECONDS = 10


class TelegramNotifier:
    """Envia alertas operacionais para um chat do Telegram."""

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
        """Notifica que o bot foi inicializado."""
        return self._send_message(
            "\n".join(
                [
                    "[Traderbot] Bot inicializado",
                    "Status: online",
                    "Alerta: notificações Telegram ativas",
                ]
            )
        )

    def notify_order_executed(
        self,
        asset: str,
        side: str,
        price: float | int,
        quantity: float | int,
    ) -> bool:
        """Notifica a execucao de uma nova ordem."""
        side_label = str(side).strip().upper()
        return self._send_message(
            "\n".join(
                [
                    "[Traderbot] Nova ordem executada",
                    f"Ativo: {asset}",
                    f"Lado: {side_label}",
                    f"Preco: {self._format_decimal(price)}",
                    f"Quantidade: {self._format_decimal(quantity)}",
                ]
            )
        )

    def notify_position_closed(
        self,
        asset: str,
        pnl: float | int,
        side: str | None = None,
    ) -> bool:
        """Notifica o fecho de uma posicao com o PnL realizado."""
        pnl_value = float(pnl)
        pnl_label = "lucro" if pnl_value >= 0 else "prejuizo"

        lines = [
            "[Traderbot] Posicao fechada",
            f"Ativo: {asset}",
        ]
        if side:
            lines.append(f"Lado: {str(side).strip().upper()}")
        lines.extend(
            [
                f"Resultado: {pnl_label}",
                f"PnL: {self._format_decimal(pnl_value, signed=True)}",
            ]
        )
        return self._send_message("\n".join(lines))

    def notify_critical_error(
        self,
        error_message: str,
        details: str | None = None,
    ) -> bool:
        """Notifica falhas criticas, como erros de API ou execucao."""
        lines = [
            "[Traderbot] ERRO CRITICO",
            f"Mensagem: {error_message}",
        ]
        if details:
            lines.append(f"Detalhes: {details}")
        return self._send_message("\n".join(lines))

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
    def _format_decimal(value: float | int, signed: bool = False) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        if signed:
            return f"{numeric_value:+.8f}".rstrip("0").rstrip(".")
        return f"{numeric_value:.8f}".rstrip("0").rstrip(".")
