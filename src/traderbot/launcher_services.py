from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from typing import Any


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


@dataclass(slots=True)
class HumanizedEvent(LauncherEvent):
    market_state: str = ""
    model_interpretation: str = ""
    filters_diagnostic: str = ""
    execution_summary: str = ""
    simple_summary: str = ""


STATE_COLORS = {
    "info": "blue",
    "warning": "yellow",
    "error": "red",
    "execution": "purple",
    "risk": "orange",
    "blocked": "gray",
}


AI_HUMANIZED_DETAILS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "market_state": {"type": "string"},
        "model_interpretation": {"type": "string"},
        "filters_diagnostic": {"type": "string"},
        "execution_summary": {"type": "string"},
        "simple_summary": {"type": "string"},
    },
    "required": [
        "market_state",
        "model_interpretation",
        "filters_diagnostic",
        "execution_summary",
        "simple_summary",
    ],
}


AI_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["info", "warning", "error", "execution", "risk", "blocked"],
        },
        "message": {"type": "string"},
        "details": {"type": ["string", "null"]},
        "color": {
            "type": "string",
            "enum": ["blue", "yellow", "red", "purple", "orange", "gray"],
        },
        "relevant": {"type": "boolean"},
        "humanized_details": AI_HUMANIZED_DETAILS_SCHEMA,
    },
    "required": ["severity", "message", "details", "color", "relevant", "humanized_details"],
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
        market_state = self._cycle_market_state(payload)
        model_interpretation = self._cycle_model_interpretation(payload)
        filters_diagnostic = self._cycle_filters_diagnostic(payload)
        execution_summary = self._cycle_execution_summary(payload)
        if payload.get("error"):
            return self._build(
                event,
                severity="error",
                message="Erro operacional no ciclo do bot",
                details=self._shorten(str(payload.get("error"))),
                fingerprint=f"cycle:error:{str(payload.get('error')).lower()}:{cycle_key}",
                event_code="execution.cycle_error",
                market_state=market_state,
                model_interpretation=model_interpretation,
                filters_diagnostic=filters_diagnostic,
                execution_summary=execution_summary,
                simple_summary="Erro operacional no ciclo do bot",
            )

        if payload.get("opened_trade"):
            direction = str(payload.get("position_label", payload.get("trade_direction", "TRADE"))).upper()
            notional = float(payload.get("adjusted_notional", payload.get("notional_value", 0.0)) or 0.0)
            risk_amount = float(payload.get("risk_amount", 0.0) or 0.0)
            details_parts: list[str] = []
            if notional > 0:
                details_parts.append(f"Notional ${notional:,.2f}")
            if risk_amount > 0:
                details_parts.append(f"Risco ${risk_amount:,.2f}")
            return self._build(
                event,
                severity="execution",
                message=f"Posicao {direction} aberta",
                details=" | ".join(details_parts) if details_parts else None,
                fingerprint=f"cycle:open:{direction}:{cycle_key}",
                event_code="execution.trade_open",
                market_state=market_state,
                model_interpretation=model_interpretation,
                filters_diagnostic=filters_diagnostic,
                execution_summary=execution_summary,
                simple_summary=f"Posicao {direction} aberta",
            )

        if payload.get("closed_trade"):
            exit_reason = self._human_exit_reason(payload.get("exit_reason"))
            return self._build(
                event,
                severity="execution",
                message=f"Posicao encerrada por {exit_reason}",
                details=None,
                fingerprint=f"cycle:close:{str(payload.get('exit_reason', 'unknown')).lower()}:{cycle_key}",
                event_code="execution.trade_close",
                market_state=market_state,
                model_interpretation=model_interpretation,
                filters_diagnostic=filters_diagnostic,
                execution_summary=execution_summary,
                simple_summary=f"Posicao encerrada por {exit_reason}",
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
                market_state=market_state,
                model_interpretation=model_interpretation,
                filters_diagnostic=filters_diagnostic,
                execution_summary=execution_summary,
                simple_summary=reason,
            )

        if payload.get("position_is_open"):
            vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
            return self._build(
                event,
                severity="execution",
                message=f"Posicao mantida ({vote_bucket})",
                details=None,
                fingerprint=f"cycle:hold-open:{vote_bucket}:{cycle_key}",
                event_code="execution.position_held",
                market_state=market_state,
                model_interpretation=model_interpretation,
                filters_diagnostic=filters_diagnostic,
                execution_summary=execution_summary,
                simple_summary=f"Posicao mantida ({vote_bucket})",
            )

        vote_bucket = str(payload.get("vote_bucket", "hold")).upper()
        regime_valid = bool(payload.get("regime_valid"))
        short_message = "Sem entrada na barra"
        details = f"Bias {vote_bucket} | Regime {'OK' if regime_valid else 'BLOQUEADO'}"
        return self._build(
            event,
            severity="info",
            message=short_message,
            details=details,
            fingerprint=f"cycle:flat:{vote_bucket}:{regime_valid}:{cycle_key}",
            event_code="system.waiting",
            market_state=market_state,
            model_interpretation=model_interpretation,
            filters_diagnostic=filters_diagnostic,
            execution_summary=execution_summary,
            simple_summary=f"{short_message} | {details}",
        )

    def _cycle_market_state(self, payload: dict[str, Any]) -> str:
        price = self._coerce_float(self._payload_value(payload, "reference_price", snapshot="market_snapshot"))
        rsi_14 = self._coerce_float(self._payload_value(payload, "rsi_14", snapshot="feature_snapshot"))
        dist_ema_240 = self._coerce_float(
            self._payload_value(payload, "dist_ema_240", "regime_dist_ema_240", snapshot="feature_snapshot")
        )
        atr_pct = self._coerce_float(self._payload_value(payload, "atr_pct", snapshot="feature_snapshot"))
        parts: list[str] = []
        if price is not None:
            parts.append(f"Preco ref ${price:,.2f}")
        if rsi_14 is not None:
            parts.append(f"RSI 14 {rsi_14:.1f}")
        if dist_ema_240 is not None:
            parts.append(f"Dist EMA240 {dist_ema_240 * 100.0:+.2f}%")
        if atr_pct is not None:
            parts.append(f"ATR {atr_pct * 100.0:.2f}%")
        return " | ".join(parts) if parts else "Dado indisponivel"

    def _cycle_model_interpretation(self, payload: dict[str, Any]) -> str:
        vote_bucket = str(self._payload_value(payload, "vote_bucket", snapshot="decision") or "HOLD").upper()
        final_action = self._coerce_float(self._payload_value(payload, "final_action", snapshot="decision"))
        confidence_pct = self._coerce_float(self._payload_value(payload, "confidence_pct", snapshot="decision"))
        votes = self._payload_value(payload, "votes", snapshot="decision")
        parts = [f"Bucket {vote_bucket}"]
        if final_action is not None:
            parts.append(f"Acao final {final_action:+.2f}")
        if confidence_pct is not None:
            parts.append(f"Confianca {confidence_pct:.1f}%")
        vote_summary = self._summarize_votes(votes)
        if vote_summary:
            parts.append(vote_summary)
        return " | ".join(parts) if parts else "Dado indisponivel"

    def _cycle_filters_diagnostic(self, payload: dict[str, Any]) -> str:
        blocked_reason = self._payload_value(payload, "blocked_reason", snapshot="filters")
        regime_valid = self._payload_value(payload, "regime_valid_for_entry", "regime_valid", snapshot="filters")
        if blocked_reason:
            return self._human_block_reason(str(blocked_reason))
        if regime_valid is None:
            return "Dado indisponivel"
        return "Filtros OK" if bool(regime_valid) else "Regime bloqueado"

    def _cycle_execution_summary(self, payload: dict[str, Any]) -> str:
        position_label = str(
            self._payload_value(payload, "position_label", "trade_direction", snapshot="execution") or "FLAT"
        ).upper()
        if self._payload_value(payload, "opened_trade", "opened_position", snapshot="execution"):
            size = self._coerce_float(self._payload_value(payload, "position_size", snapshot="execution"))
            return (
                f"Posicao {position_label} aberta"
                if size is None
                else f"Posicao {position_label} aberta | tamanho {size:.6f}"
            )
        if self._payload_value(payload, "closed_trade", "closed_position", snapshot="execution"):
            exit_reason = self._human_exit_reason(self._payload_value(payload, "exit_reason", snapshot="execution"))
            return f"Posicao encerrada por {exit_reason}"
        if self._payload_value(payload, "position_is_open", snapshot="execution"):
            pnl = self._coerce_float(self._payload_value(payload, "position_unrealized_pnl", snapshot="execution"))
            return (
                f"Posicao {position_label} mantida"
                if pnl is None
                else f"Posicao {position_label} mantida | PnL nao realizado ${pnl:,.2f}"
            )
        return "Sem posicao aberta nesta barra"

    def _payload_value(self, payload: dict[str, Any], *keys: str, snapshot: str | None = None) -> Any:
        snapshot_payload = payload.get(snapshot) if snapshot else None
        if isinstance(snapshot_payload, dict):
            for key in keys:
                if key in snapshot_payload and snapshot_payload[key] is not None:
                    return snapshot_payload[key]
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    def _summarize_votes(self, votes: Any) -> str:
        if isinstance(votes, dict):
            parts = [f"{key.upper()}={value}" for key, value in votes.items()]
            return "Votos " + ", ".join(parts[:4]) if parts else ""
        if isinstance(votes, list):
            return f"Votos {', '.join(str(item) for item in votes[:4])}"
        if votes not in (None, ""):
            return f"Votos {votes}"
        return ""

    def _coerce_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().replace(",", ".")
            if not normalized:
                return None
            try:
                return float(normalized)
            except ValueError:
                return None
        return None

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
        market_state: str = "",
        model_interpretation: str = "",
        filters_diagnostic: str = "",
        execution_summary: str = "",
        simple_summary: str = "",
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
            market_state=market_state,
            model_interpretation=model_interpretation,
            filters_diagnostic=filters_diagnostic,
            execution_summary=execution_summary,
            simple_summary=simple_summary,
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
