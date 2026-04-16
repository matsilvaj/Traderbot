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


@dataclass(slots=True)
class HumanizedEvent:
    severity: str
    message: str
    details: str | None
    color: str
    relevant: bool
    fingerprint: str
    occurred_at: datetime
    raw_detail: str = ""


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
        )

    def _cycle_event(self, event: LauncherEvent, payload: dict[str, Any]) -> HumanizedEvent:
        if payload.get("error"):
            return self._build(
                event,
                severity="error",
                message="Erro operacional no ciclo do bot",
                details=str(payload.get("error")),
                fingerprint=f"cycle:error:{str(payload.get('error')).lower()}",
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
                fingerprint=f"cycle:open:{direction}",
            )

        if payload.get("closed_trade"):
            exit_reason = self._human_exit_reason(payload.get("exit_reason"))
            return self._build(
                event,
                severity="execution",
                message=f"Posicao encerrada por {exit_reason}",
                details="A posicao foi fechada e o bot voltou a aguardar a proxima oportunidade.",
                fingerprint=f"cycle:close:{str(payload.get('exit_reason', 'unknown')).lower()}",
            )

        if payload.get("blocked_reason"):
            reason = self._human_block_reason(str(payload.get("blocked_reason")))
            return self._build(
                event,
                severity="blocked",
                message=reason,
                details=None,
                fingerprint=f"cycle:block:{str(payload.get('blocked_reason')).lower()}",
            )

        if payload.get("position_is_open"):
            vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
            return self._build(
                event,
                severity="execution",
                message=f"Posicao mantida; ensemble segue em {vote_bucket}",
                details=None,
                fingerprint=f"cycle:hold-open:{vote_bucket}",
                relevant=False,
            )

        vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
        regime_valid = bool(payload.get("regime_valid"))
        short_message = "Bot aguardando proximo contexto util"
        details = (
            "Sem entrada nesta barra."
            if regime_valid
            else "Sem entrada nesta barra porque o regime atual segue invalido."
        )
        return self._build(
            event,
            severity="info",
            message=short_message,
            details=f"{details} Ensemble em {vote_bucket}.",
            fingerprint=f"cycle:flat:{vote_bucket}:{regime_valid}",
            relevant=False,
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
        relevant: bool = True,
    ) -> HumanizedEvent:
        color = STATE_COLORS.get(severity, "blue")
        return HumanizedEvent(
            severity=severity,
            message=message,
            details=details,
            color=color,
            relevant=relevant,
            fingerprint=fingerprint,
            occurred_at=event.occurred_at,
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
            "event_type": event.event_type,
            "payload": event.payload,
            "message": event.message,
            "fallback": {
                "severity": fallback.severity,
                "message": fallback.message,
                "details": fallback.details,
                "color": fallback.color,
                "relevant": fallback.relevant,
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
            raw_log = event.raw_line or event.message or json.dumps(event.payload, ensure_ascii=False)
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
                                "text": f"INPUT:\n\nLOG:\n{raw_log}",
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
            translated = json.loads(response.output_text)
            candidate = HumanizedEvent(
                severity=str(translated.get("severity", fallback.severity)),
                message=str(translated.get("message", fallback.message)),
                details=translated.get("details", fallback.details),
                color=str(translated.get("color", fallback.color)),
                relevant=bool(translated.get("relevant", fallback.relevant)),
                fingerprint=fallback.fingerprint,
                occurred_at=fallback.occurred_at,
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
        message = (event.message or "").strip()
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
            "{",
            "}",
            "|",
        )
        return not any(fragment in lowered for fragment in forbidden)
