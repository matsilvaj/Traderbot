from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from typing import Any

from traderbot.config import AppConfig

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback runtime when dependency is absent
    OpenAI = None


@dataclass(slots=True)
class LauncherEvent:
    source: str
    event_type: str
    raw_line: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.now)
    severity: str = "info"
    event_code: str | None = None
    details: str | None = None
    network: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    color: str = "blue"
    relevant: bool = True
    fingerprint: str = ""
    raw_detail: str = ""

    def __post_init__(self) -> None:
        if not self.raw_detail:
            self.raw_detail = self.raw_line
        if self.network is None:
            self.network = self._derive_context_field("network")
        if self.symbol is None:
            self.symbol = self._derive_context_field("symbol")
        if self.timeframe is None:
            self.timeframe = self._derive_context_field("timeframe")

    def _derive_context_field(self, field_name: str) -> str | None:
        value = self.payload.get(field_name)
        if value not in (None, ""):
            return str(value)
        match = re.search(rf"{re.escape(field_name)}=([^ |\n]+)", self.raw_line, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).strip()

    @property
    def timestamp(self) -> datetime:
        return self.occurred_at

    @property
    def type(self) -> str:
        return self.event_type

    @property
    def message_raw(self) -> str:
        return self.raw_line

    @property
    def message_human(self) -> str:
        return self.message

    @message_human.setter
    def message_human(self, value: str) -> None:
        self.message = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self.payload


HumanizedEvent = LauncherEvent


STATE_COLORS = {
    "info": "blue",
    "warning": "yellow",
    "error": "red",
    "execution": "purple",
    "risk": "orange",
    "blocked": "gray",
}


class LocalLogInterpreter:
    _number_re = re.compile(r"\b\d+(?:\.\d+)?\b")

    def summarize(self, event: LauncherEvent) -> HumanizedEvent:
        payload = event.payload or {}
        if event.event_type == "health":
            return self._health_event(event, payload)
        if event.event_type == "status":
            return self._status_event(event, payload)
        if event.event_type == "cycle":
            return self._cycle_event(event, payload)
        if event.event_type == "smoke":
            return self._smoke_event(event, payload)
        if event.event_type == "manual_close":
            return self._manual_close_event(event, payload)
        return self._generic_event(event)

    def _status_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        network = str(payload.get("network", "--")).upper()
        available = float(payload.get("available_to_trade", 0.0) or 0.0)
        connected = bool(payload.get("connected"))
        can_trade = bool(payload.get("can_trade"))
        if connected and can_trade:
            short = f"Conectado na {network} com saldo operacional de ${available:,.2f}"
            details = "Check operacional concluido sem bloqueios criticos."
            severity = "info"
        elif connected:
            short = f"Conectado na {network}, mas ainda nao apto para operar"
            details = "A conexao respondeu, porem a conta nao ficou pronta para envio de ordens."
            severity = "warning"
        else:
            short = f"Sem conexao operacional com a {network}"
            details = "O check nao conseguiu validar conectividade suficiente para operar."
            severity = "error"
        return self._build(
            event,
            severity=severity,
            message=short,
            details=details,
            fingerprint=f"status:{network}:{connected}:{can_trade}",
            event_code="healthcheck.status",
        )

    def _cycle_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        cycle_key = self._cycle_fingerprint_suffix(payload)
        if payload.get("error"):
            return self._build(
                event,
                severity="error",
                message="Erro operacional no ciclo do bot",
                details=str(payload.get("error")),
                fingerprint=f"cycle:error:{str(payload.get('error')).lower()}:{cycle_key}",
                event_code="execution.cycle_error",
            )

        if payload.get("opened_trade"):
            direction = str(payload.get("position_label", payload.get("trade_direction", "TRADE"))).upper()
            notional = float(payload.get("adjusted_notional", payload.get("notional_value", 0.0)) or 0.0)
            risk_amount = float(payload.get("risk_amount", 0.0) or 0.0)
            return self._build(
                event,
                severity="execution",
                message=f"Nova posicao {direction} aberta em ${notional:,.2f}",
                details=f"Risco efetivo usado no disparo: ${risk_amount:,.2f}.",
                fingerprint=f"cycle:open:{direction}:{cycle_key}",
                event_code="execution.trade_open",
            )

        if payload.get("closed_trade"):
            exit_reason = self._human_exit_reason(payload.get("exit_reason"))
            return self._build(
                event,
                severity="execution",
                message=f"Posicao encerrada por {exit_reason}",
                details="A posicao foi fechada e o bot voltou a aguardar a proxima oportunidade.",
                fingerprint=f"cycle:close:{str(payload.get('exit_reason', 'unknown')).lower()}:{cycle_key}",
                event_code="execution.trade_close",
            )

        if payload.get("blocked_reason"):
            reason = self._human_block_reason(str(payload.get("blocked_reason")))
            return self._build(
                event,
                severity="blocked",
                message=reason,
                details=None,
                fingerprint=f"cycle:block:{str(payload.get('blocked_reason')).lower()}:{cycle_key}",
                event_code=f"blocked.{str(payload.get('blocked_reason')).lower()}",
            )

        if payload.get("position_is_open"):
            vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
            return self._build(
                event,
                severity="execution",
                message=f"Posicao mantida; ensemble segue em {vote_bucket}",
                details=None,
                fingerprint=f"cycle:hold-open:{vote_bucket}:{cycle_key}",
                event_code="execution.position_held",
            )

        vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
        regime_valid = bool(payload.get("regime_valid"))
        short_message = "Sem entrada nesta barra"
        details = (
            f"Ensemble em {vote_bucket} e regime valido, sem gatilho de entrada nesta vela."
            if regime_valid
            else f"Ensemble em {vote_bucket} e regime invalido nesta vela."
        )
        return self._build(
            event,
            severity="info",
            message=short_message,
            details=details,
            fingerprint=f"cycle:flat:{vote_bucket}:{regime_valid}:{cycle_key}",
            event_code="system.waiting",
        )

    def _smoke_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        ok = bool(payload.get("ok"))
        network = str(payload.get("network", "--")).upper()
        opened = bool(payload.get("opened_position"))
        closed = bool(payload.get("closed_position"))
        return self._build(
            event,
            severity="execution" if ok else "error",
            message=(
                f"Smoke test validado na {network}"
                if ok
                else f"Smoke test com falha na {network}"
            ),
            details=f"Abertura={opened} | fechamento={closed}.",
            fingerprint=f"smoke:{network}:{ok}:{opened}:{closed}",
            event_code="system.smoke_test",
        )

    def _manual_close_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        ok = bool(payload.get("ok"))
        return self._build(
            event,
            severity="risk" if ok else "warning",
            message=(
                "Kill switch executado e posicao encerrada"
                if ok
                else "Kill switch executado sem posicao aberta"
            ),
            details="Fechamento manual enviado pelo launcher.",
            fingerprint=f"manual_close:{ok}",
            event_code="risk.manual_close",
        )

    def _health_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        status = str(payload.get("status", "online" if payload.get("online") else "offline")).lower()
        reason = str(payload.get("reason", "unknown"))
        bot_running = bool(payload.get("bot_running"))
        connection_ok = bool(payload.get("connection_ok"))
        executor_alive = bool(payload.get("executor_alive"))

        online_messages = {
            "check_hyperliquid_ok": "Check operacional confirmou sistema online",
            "bot_running_and_healthcheck_ok": "Bot rodando com saude operacional valida",
        }
        warning_messages = {
            "awaiting_first_healthcheck": "Bot rodando e aguardando o primeiro check",
            "health_check_command_failed": "Check operacional falhou, mas o bot segue rodando",
            "health_check_stale_while_running": "Check operacional esta atrasado",
        }
        offline_messages = {
            "check_hyperliquid_failed": "Check operacional marcou o sistema como offline",
            "connection_check_failed": "Falha confirmada na conectividade operacional",
            "runtime_process_exited": "Runtime foi encerrado inesperadamente",
            "bot_not_running": "Bot parado",
            "executor_not_alive": "Executor indisponivel",
            "network_switch_pending_check": "Trocando de rede e aguardando novo check",
        }

        if status == "online":
            message = online_messages.get(reason, "Sistema online")
            severity = "info"
        elif status == "warning":
            message = warning_messages.get(reason, "Saude operacional instavel")
            severity = "warning"
        else:
            message = offline_messages.get(reason, "Sistema offline")
            severity = "error" if bot_running or connection_ok is False or executor_alive is False else "warning"

        details = f"Motivo tecnico: {reason}"
        return self._build(
            event,
            severity=severity,
            message=message,
            details=details,
            fingerprint=f"health:{status}:{reason}",
            event_code=f"health.{reason}",
        )

    def _generic_event(self, event: LauncherEvent) -> HumanizedEvent:
        message = event.message or event.raw_line
        lowered = message.lower()
        if "diagnostico hyperliquid" in lowered:
            severity = "info"
            details = None
            short = self._diagnostic_message(message)
        elif "executor conectado" in lowered:
            severity = "info"
            details = None
            short = self._executor_connected_message(message)
        elif "entrada bloqueada" in lowered:
            severity = "blocked"
            details = None
            short = "Entrada bloqueada pelo regime atual"
        elif "502" in lowered and ("http" in lowered or "error" in lowered):
            severity = "error"
            details = None
            short = "Falha de conexao com a API (502)"
        elif "erro" in lowered or "traceback" in lowered or "exception" in lowered:
            severity = "error"
            details = None
            short = "Erro ao comunicar com a exchange"
        elif "conectado" in lowered:
            severity = "info"
            details = None
            short = "Conexao operacional atualizada"
        elif "iniciado" in lowered or "inicializando" in lowered:
            severity = "info"
            details = None
            short = "Inicializacao operacional em andamento"
        else:
            severity = "info"
            details = None
            short = self._generic_fallback_message(message)
        noisy_fragments = (
            "iniciando pipeline com comando",
            "overrides de runtime aplicados via cli",
            "comando:",
        )
        is_relevant = not any(fragment in lowered for fragment in noisy_fragments)
        return self._build(
            event,
            severity=severity,
            message=short,
            details=details,
            fingerprint=f"generic:{self._normalize(message)}",
            event_code=f"system.{severity}",
            relevant=is_relevant and "ciclo runtime" not in lowered,
        )

    def _build(
        self,
        event: LauncherEvent,
        *,
        severity: str,
        message: str,
        details: str | None,
        fingerprint: str,
        event_code: str | None = None,
        relevant: bool = True,
    ) -> HumanizedEvent:
        color = STATE_COLORS.get(severity, "blue")
        return HumanizedEvent(
            source=event.source,
            event_type=event.event_type,
            raw_line=event.raw_line,
            message=message,
            payload=dict(event.payload or {}),
            occurred_at=event.occurred_at,
            severity=severity,
            event_code=event_code,
            details=details,
            network=event.network,
            symbol=event.symbol,
            timeframe=event.timeframe,
            color=color,
            relevant=relevant,
            fingerprint=fingerprint,
            raw_detail=event.raw_line,
        )

    def _human_block_reason(self, reason: str) -> str:
        mapping = {
            "regime_filter": "Entrada bloqueada pelo filtro de regime",
            "cooldown": "Entrada bloqueada pelo cooldown operacional",
            "force_exit_only_by_tp_sl": "Sinal ignorado porque a posicao esta travada por TP/SL",
            "min_notional_risk": "Entrada pulada porque o notional minimo violaria o risco maximo",
            "exchange_position_present": "Sinal ignorado porque ja existe posicao aberta na exchange",
            "duplicate_cycle_order": "Sinal ignorado para evitar ordem duplicada no mesmo ciclo",
        }
        return mapping.get(reason, f"Entrada bloqueada: {reason}")

    def _human_exit_reason(self, reason: Any) -> str:
        reason_text = str(reason or "fechamento manual").lower()
        mapping = {
            "take_profit": "take profit",
            "stop_loss": "stop loss",
            "margin_call": "margin call",
            "episode_end": "fim de sessao",
            "manual_close": "fechamento manual",
            "smoke_test_manual_close": "fechamento manual do smoke test",
        }
        return mapping.get(reason_text, reason_text.replace("_", " "))

    def _diagnostic_message(self, message: str) -> str:
        network = self._extract_field(message, "network") or "rede"
        symbol = self._extract_field(message, "symbol") or "ativo"
        timeframe = self._extract_field(message, "timeframe") or "--"
        return f"Diagnostico iniciado para {symbol} em {timeframe} na {network}"

    def _executor_connected_message(self, message: str) -> str:
        network = self._extract_field(message, "network") or "rede"
        available = self._extract_field(message, "available_to_trade")
        if available is not None:
            try:
                return f"Executor conectado na {network} com saldo de ${float(available):,.2f}"
            except ValueError:
                pass
        return f"Executor conectado na {network}"

    def _generic_fallback_message(self, message: str) -> str:
        cleaned = re.sub(r"\s+", " ", message).strip()
        if "|" in cleaned:
            cleaned = cleaned.split("|", 1)[0].strip()
        cleaned = re.sub(r"wallet=0x[a-fA-F0-9]+", "", cleaned)
        cleaned = re.sub(r"base_url=\S+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" |")
        if not cleaned:
            return "Evento operacional atualizado"
        if len(cleaned) > 96:
            cleaned = cleaned[:93].rstrip() + "..."
        return cleaned

    def _extract_field(self, message: str, field: str) -> str | None:
        match = re.search(rf"{re.escape(field)}=([^ |\n]+)", message, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).strip()

    def _shorten(self, message: str) -> str:
        compact = re.sub(r"\s+", " ", message).strip()
        if len(compact) <= 110:
            return compact
        return compact[:107].rstrip() + "..."

    def _normalize(self, message: str) -> str:
        normalized = self._number_re.sub("#", message.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def _cycle_fingerprint_suffix(self, payload: dict[str, Any]) -> str:
        bar_timestamp = str(payload.get("bar_timestamp") or "").strip()
        timestamp = str(payload.get("timestamp") or "").strip()
        return bar_timestamp or timestamp or "unknown_cycle"


class OpenAILogTranslator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.local = LocalLogInterpreter()
        self._cache: dict[str, HumanizedEvent] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.launcher.openai_enabled and self.api_key and OpenAI is not None)

    @property
    def api_key(self) -> str | None:
        env_name = str(self.cfg.launcher.openai_api_key_env or "OPENAI_API_KEY").strip()
        if not env_name:
            return None
        value = os.getenv(env_name)
        if value is None:
            value = os.getenv(f"\ufeff{env_name}")
        if value is None:
            return None
        value = value.strip().strip("\"'")
        return value or None

    def cache_key(self, event: LauncherEvent, fallback: HumanizedEvent) -> str:
        raw_payload = {
            "type": event.type,
            "severity": event.severity,
            "event_code": event.event_code,
            "message_raw": self._normalize_cache_text(event.message_raw),
            "network": event.network,
            "symbol": event.symbol,
            "timeframe": event.timeframe,
            "fallback": {
                "severity": fallback.severity,
                "message_human": fallback.message_human,
                "details": fallback.details,
                "color": fallback.color,
                "relevant": fallback.relevant,
                "event_code": fallback.event_code,
            },
        }
        serialized = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        return sha1(serialized.encode("utf-8")).hexdigest()

    def translate(self, event: LauncherEvent, fallback: HumanizedEvent) -> HumanizedEvent:
        if not self.enabled:
            return fallback

        key = self.cache_key(event, fallback)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            client = OpenAI(api_key=self.api_key, timeout=float(self.cfg.launcher.openai_timeout_seconds))
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error", "execution", "risk", "blocked"]},
                    "message": {"type": "string"},
                    "details": {"type": ["string", "null"]},
                    "color": {"type": "string", "enum": ["blue", "yellow", "red", "purple", "orange", "gray"]},
                    "relevant": {"type": "boolean"},
                },
                "required": ["severity", "message", "details", "color", "relevant"],
            }
            structured_input = {
                "timestamp": event.timestamp.isoformat(),
                "type": event.type,
                "severity": event.severity,
                "event_code": event.event_code,
                "message_raw": event.message_raw,
                "message_human_fallback": fallback.message_human,
                "details_fallback": fallback.details,
                "network": event.network,
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "metadata": event.metadata,
            }
            raw_log = event.message_raw or event.message_human or json.dumps(event.metadata, ensure_ascii=False)
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Voce e um interpretador de logs de um bot de trading.\n\n"
                                    "Sua funcao:\n"
                                    "- traduzir logs tecnicos para linguagem humana\n"
                                    "- resumir ao maximo\n"
                                    "- remover ruido tecnico\n"
                                    "- focar apenas no que importa operacionalmente\n\n"
                                    "REGRAS:\n"
                                    "- maximo 1 frase (ideal)\n"
                                    "- nunca repetir o log original\n"
                                    "- nunca explicar demais\n"
                                    "- remover wallet, base_url, ids, etc\n"
                                    "- linguagem direta e objetiva\n\n"
                                    "CLASSIFICACAO:\n"
                                    "- info\n"
                                    "- warning\n"
                                    "- error\n"
                                    "- execution\n"
                                    "- risk\n"
                                    "- blocked\n\n"
                                    "CORES:\n"
                                    "- info -> blue\n"
                                    "- warning -> yellow\n"
                                    "- error -> red\n"
                                    "- execution -> purple\n"
                                    "- risk -> orange\n"
                                    "- blocked -> gray\n\n"
                                    "FORMATO DE SAIDA OBRIGATORIO JSON:\n"
                                    "{\n"
                                    '  "severity": "info | warning | error | execution | risk | blocked",\n'
                                    '  "message": "mensagem curta e humana",\n'
                                    '  "details": null,\n'
                                    '  "color": "blue | yellow | red | purple | orange | gray",\n'
                                    '  "relevant": true\n'
                                    "}"
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"INPUT:\n\nEVENT:\n{json.dumps(structured_input, ensure_ascii=False)}\n\nLOG:\n{raw_log}",
                            }
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "launcher_event_summary",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            translated = self._parse_translation_payload(response.output_text)
            candidate = HumanizedEvent(
                source=fallback.source,
                event_type=fallback.event_type,
                raw_line=fallback.raw_line,
                payload=dict(fallback.payload or {}),
                occurred_at=fallback.occurred_at,
                severity=str(translated.get("severity", fallback.severity)),
                message=str(translated.get("message", fallback.message_human)),
                event_code=fallback.event_code,
                details=translated.get("details", fallback.details),
                network=fallback.network,
                symbol=fallback.symbol,
                timeframe=fallback.timeframe,
                color=str(translated.get("color", fallback.color)),
                relevant=bool(translated.get("relevant", fallback.relevant)),
                fingerprint=fallback.fingerprint,
                raw_detail=fallback.raw_detail,
            )
            result = candidate if self._is_clean_translation(candidate, raw_log) else fallback
        except Exception:
            result = fallback

        with self._lock:
            self._cache[key] = result
        return result

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.launcher.openai_enabled),
            "api_key_present": bool(self.api_key),
            "client_ready": bool(self.enabled),
            "model": "gpt-4o-mini",
            "api_key_env": self.cfg.launcher.openai_api_key_env,
        }

    def _is_clean_translation(self, event: HumanizedEvent, raw_log: str) -> bool:
        message = (event.message_human or "").strip()
        lowered = message.lower()
        raw_lowered = (raw_log or "").strip().lower()
        if not message:
            return False
        if len(message) > 140:
            return False
        if message.lower() == raw_lowered:
            return False
        forbidden = (
            "wallet=",
            "base_url",
            "traceback",
            "requests.",
            "hyperliquid |",
            "0x",
            "http://",
            "https://",
            "{",
            "}",
            "|",
            "=",
        )
        return not any(fragment in lowered for fragment in forbidden)

    def _parse_translation_payload(self, raw_output: str) -> dict[str, Any]:
        payload = json.loads(raw_output)
        if not isinstance(payload, dict):
            raise ValueError("translation payload is not an object")

        severity = str(payload.get("severity", "")).strip().lower()
        color = str(payload.get("color", "")).strip().lower()
        message = str(payload.get("message", "")).strip()
        relevant = payload.get("relevant")
        details = payload.get("details")

        if severity not in {"info", "warning", "error", "execution", "risk", "blocked"}:
            raise ValueError("invalid severity")
        if color not in {"blue", "yellow", "red", "purple", "orange", "gray"}:
            raise ValueError("invalid color")
        if not message:
            raise ValueError("empty message")
        if relevant is not True and relevant is not False:
            raise ValueError("invalid relevant flag")
        if details is not None and not isinstance(details, str):
            raise ValueError("invalid details")

        return {
            "severity": severity,
            "message": message,
            "details": details,
            "color": color,
            "relevant": bool(relevant),
        }

    def _normalize_cache_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        compact = re.sub(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|z)?", "<ts>", compact)
        compact = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<time>", compact)
        return compact
