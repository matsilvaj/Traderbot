from __future__ import annotations

import json
import logging
import os
import re
import threading
from hashlib import sha1
from typing import Any

from traderbot.config import AppConfig
from traderbot.launcher_services import HumanizedEvent, LauncherEvent, LocalLogInterpreter

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback runtime when dependency is absent
    OpenAI = None


AI_MODEL_NAME = "gpt-4o-mini"
WAITING_FEATURES_MESSAGE = "Aguardando proxima atualizacao de features"
logger = logging.getLogger(__name__)

AI_HUMANIZED_DETAILS_SCHEMA = {
    "type": "object",
    "description": (
        "Campos detalhados do evento. market_state e model_interpretation sao tecnicos e brutos; "
        "simple_summary e o resumo humano curto para a interface principal."
    ),
    "additionalProperties": False,
    "properties": {
        "market_state": {
            "type": ["string", "null"],
            "description": "Dados tecnicos brutos do market_snapshot, como preco, RSI e distancia de EMA.",
        },
        "model_interpretation": {
            "type": ["string", "null"],
            "description": (
                "Dados tecnicos brutos do decision, ensemble_score, model_weights e indicators. "
                "Use lista ou tabela HTML curta se isso organizar melhor os votos."
            ),
        },
        "filters_diagnostic": {
            "type": ["string", "null"],
            "description": "Diagnostico tecnico dos filtros e bloqueios do ciclo.",
        },
        "execution_summary": {
            "type": ["string", "null"],
            "description": "Resumo tecnico da execucao ou do estado da posicao.",
        },
        "simple_summary": {
            "type": ["string", "null"],
            "description": "Resumo humano curto e direto para a interface principal.",
        },
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


class LauncherAIPromptBuilder:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def build_system_prompt(self) -> str:
        return (
            "Siga estas REGRAS RIGIDAS de objetividade:\n"
            "1. PROIBIDO REPETIR: Se uma informacao ja esta no 'simple_summary', nao a repita nos outros campos.\n"
            "2. VOTACAO DOS MODELOS: No campo 'model_interpretation', voce DEVE procurar no metadado 'decision' "
            "pelo score de cada modelo (ex: ppo_s1, ppo_s21) e listar como: 'Model [Nome]: [Voto %]'.\n"
            "2.1. No campo model_interpretation, voce DEVE extrair os valores numericos do dicionario decision. "
            "Se encontrar chaves como ppo_s1, ppo_s21, etc., liste cada uma e seu respectivo valor decimal (voto). "
            "Se houver um ensemble_score, destaque-o.\n"
            "3. ESTADO DO MERCADO: No campo 'market_state', use apenas numeros extraidos de 'market_snapshot' "
            "(Preco, RSI, Distancia EMA). Se nao houver, escreva 'Dados insuficientes'.\n"
            "4. SEM PROSA: Evite frases como 'O modelo analisou e decidiu'. Use 'Decisao: HOLD | Motivo: Baixa Confianca'.\n"
            "5. FILTROS: Em 'filters_diagnostic', liste qual filtro barrou a entrada (ex: 'Filtro: Regime Invalido') ou 'Filtros: OK'.\n"
            "6. CAMPOS SECUNDARIOS: Se nao houver analise tecnica profunda ou votacao detalhada, responda 'N/A' em model_interpretation, "
            "filters_diagnostic e execution_summary.\n"
            "7. PRIORIDADE NUMERICA: Priorize ensemble_score, model_weights e indicators quando estiverem disponiveis.\n"
            "7.1. Se a informacao tecnica for identica ao log principal, nao a repita; aprofunde-se nos indicadores "
            "(RSI, Distancia de EMA, score, pesos e votos) que levaram a decisao.\n"
            "7.2. market_state e model_interpretation sao campos tecnicos brutos; simple_summary e o texto humano final.\n"
            "7.3. Quando houver varios votos de modelos, organize model_interpretation como texto plano ou HTML simples: "
            "'Model X: +0.50 | Model Y: -0.20'.\n"
            "7.4. NUNCA use blocos de codigo markdown, incluindo ```json, ```html ou ``` em model_interpretation ou market_state.\n"
            "7.5. Se nao houver votos dentro de decision, retorne exatamente '" + WAITING_FEATURES_MESSAGE + "' em model_interpretation.\n"
            "8. SAIDA: Responda apenas com JSON valido, sem logs crus, sem emojis e sem texto fora do schema."
        )

    def build_user_payload(
        self,
        event: LauncherEvent,
        fallback: HumanizedEvent,
        priority_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        cycle_metrics = self._resolve_cycle_metrics(metadata)
        market_snapshot = cycle_metrics["market_snapshot"]
        feature_snapshot = cycle_metrics["feature_snapshot"]
        features_snapshot = cycle_metrics["features_snapshot"]
        decision_snapshot = cycle_metrics["decision"]
        formatted_decision = self._format_payload_for_ai(decision_snapshot, signed_floats=True)
        formatted_market_snapshot = self._format_payload_for_ai(market_snapshot)
        formatted_feature_snapshot = self._format_payload_for_ai(feature_snapshot, signed_floats=True)
        formatted_features_snapshot = self._format_payload_for_ai(features_snapshot, signed_floats=True)
        return {
            "launcher_context": {
                "purpose": "cockpit pessoal para acompanhar saude do sistema, comportamento do bot e eventos operacionais",
                "symbol": self.cfg.hyperliquid.symbol,
                "timeframe": self.cfg.hyperliquid.timeframe,
                "network_default": self.cfg.hyperliquid.network,
                "decision_mode": self.cfg.execution.decision_mode,
                "selected_models": list(self.cfg.execution.selected_model_names or []),
                "order_slippage": getattr(self.cfg.execution, "order_slippage", None),
                "max_risk_per_trade": self.cfg.environment.max_risk_per_trade,
                "stop_loss_pct": self.cfg.environment.stop_loss_pct,
                "take_profit_pct": self.cfg.environment.take_profit_pct,
                "action_hold_threshold": self.cfg.environment.action_hold_threshold,
                "force_exit_only_by_tp_sl": self.cfg.environment.force_exit_only_by_tp_sl,
            },
            "event": {
                "timestamp": event.timestamp.isoformat(),
                "type": event.type,
                "severity": event.severity,
                "event_code": event.event_code,
                "message_raw": event.message_raw,
                "network": event.network,
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "metadata": metadata,
                "priority_metrics": priority_metrics or {},
                "decision_explicit": formatted_decision,
                "decision_raw": decision_snapshot,
                "decision_votes_explicit": (priority_metrics or {}).get("decision_votes"),
                "feature_snapshot_explicit": formatted_feature_snapshot,
                "feature_snapshot_raw": feature_snapshot,
                "features_snapshot_explicit": formatted_features_snapshot,
                "features_snapshot_raw": features_snapshot,
                "market_snapshot_explicit": formatted_market_snapshot,
                "market_snapshot_raw": market_snapshot,
                "cycle_metrics": cycle_metrics,
            },
            "fallback": {
                "message": fallback.message_human,
                "details": fallback.details,
                "severity": fallback.severity,
                "relevant": fallback.relevant,
                "humanized_details": {
                    "market_state": fallback.market_state,
                    "model_interpretation": fallback.model_interpretation,
                    "filters_diagnostic": fallback.filters_diagnostic,
                    "execution_summary": fallback.execution_summary,
                    "simple_summary": fallback.simple_summary,
                },
            },
            "task": (
                "Interprete o evento no contexto do bot e do launcher. "
                "Explique o comportamento relevante de forma clara para o usuario final e "
                "preencha humanized_details usando todas as metricas do ciclo disponiveis. "
                "Use decision_explicit de forma obrigatoria para extrair os votos dos modelos, "
                "use market_snapshot_explicit para market_state, use feature_snapshot_explicit e features_snapshot_explicit "
                "para reforcar indicadores numericos, nao use blocos de codigo e retorne "
                f"'{WAITING_FEATURES_MESSAGE}' em model_interpretation quando decision nao trouxer votos."
            ),
        }

    def _format_payload_for_ai(self, payload: Any, *, signed_floats: bool = False) -> Any:
        if isinstance(payload, dict):
            return {
                str(key): self._format_payload_for_ai(value, signed_floats=signed_floats)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._format_payload_for_ai(item, signed_floats=signed_floats) for item in payload]
        if isinstance(payload, float):
            return self._format_float_for_ai(payload, signed=signed_floats)
        if isinstance(payload, bool):
            return str(payload).lower()
        return payload

    def _format_float_for_ai(self, value: float, *, signed: bool = False) -> str:
        if signed:
            return f"{value:+.2f}"
        return f"{value:.2f}"

    def _resolve_cycle_metrics(self, metadata: dict[str, Any]) -> dict[str, Any]:
        market_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("market_snapshot",),
            alias_keys=("market",),
            fallback=self._market_snapshot_from_payload(metadata),
        )
        feature_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("feature_snapshot",),
            alias_keys=("features",),
            fallback=self._feature_snapshot_from_payload(metadata),
        )
        features_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("features_snapshot",),
            alias_keys=("features", "feature_snapshot"),
            fallback=feature_snapshot,
        )
        decision_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("decision",),
            fallback=self._decision_snapshot_from_payload(metadata),
        )
        filters_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("filters",),
            fallback=self._filter_snapshot_from_payload(metadata),
        )
        execution_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("execution",),
            fallback=self._execution_snapshot_from_payload(metadata),
        )
        sizing_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("sizing",),
            fallback=self._sizing_snapshot_from_payload(metadata),
        )
        system_state_snapshot = self._resolve_snapshot(
            metadata,
            primary_keys=("system_state",),
            fallback=self._system_state_from_payload(metadata),
        )
        return {
            "market_snapshot": market_snapshot,
            "feature_snapshot": feature_snapshot,
            "features_snapshot": features_snapshot,
            "decision": decision_snapshot,
            "filters": filters_snapshot,
            "execution": execution_snapshot,
            "sizing": sizing_snapshot,
            "system_state": system_state_snapshot,
        }

    def _resolve_snapshot(
        self,
        payload: dict[str, Any],
        *,
        primary_keys: tuple[str, ...],
        alias_keys: tuple[str, ...] = (),
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self._find_first_mapping(payload, primary_keys + alias_keys)
        if snapshot:
            return snapshot
        if isinstance(fallback, dict) and fallback:
            return dict(fallback)
        return {}

    def _find_first_mapping(self, payload: Any, target_keys: tuple[str, ...]) -> dict[str, Any]:
        value = self._find_first_value(payload, target_keys)
        return dict(value) if isinstance(value, dict) else {}

    def _find_first_value(self, payload: Any, target_keys: tuple[str, ...]) -> Any:
        if isinstance(payload, dict):
            for key in target_keys:
                value = payload.get(key)
                if self._has_meaningful_value(value):
                    return value
            for value in payload.values():
                found = self._find_first_value(value, target_keys)
                if self._has_meaningful_value(found):
                    return found
            return None
        if isinstance(payload, list):
            for item in payload:
                found = self._find_first_value(item, target_keys)
                if self._has_meaningful_value(found):
                    return found
        return None

    def _has_meaningful_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        return True

    def _market_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "bar_timestamp": self._find_first_value(payload, ("bar_timestamp",)),
            "open": self._find_first_value(payload, ("candle_open", "open")),
            "high": self._find_first_value(payload, ("candle_high", "high")),
            "low": self._find_first_value(payload, ("candle_low", "low")),
            "close": self._find_first_value(payload, ("candle_close", "close")),
            "volume": self._find_first_value(payload, ("candle_volume", "volume")),
            "reference_price": self._find_first_value(payload, ("reference_price",)),
        }
        return self._compact_snapshot(snapshot)

    def _feature_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "rsi_14": self._find_first_value(payload, ("rsi_14",)),
            "dist_ema_240": self._find_first_value(payload, ("dist_ema_240", "regime_dist_ema_240")),
            "dist_ema_960": self._find_first_value(payload, ("dist_ema_960",)),
            "atr_pct": self._find_first_value(payload, ("atr_pct",)),
            "volatility_24": self._find_first_value(payload, ("volatility_24",)),
            "vol_regime_z": self._find_first_value(payload, ("vol_regime_z", "regime_vol_regime_z")),
            "range_pct": self._find_first_value(payload, ("range_pct",)),
            "range_compression_10": self._find_first_value(payload, ("range_compression_10",)),
            "range_expansion_3": self._find_first_value(payload, ("range_expansion_3",)),
            "breakout_up_10": self._find_first_value(payload, ("breakout_up_10",)),
            "breakout_down_10": self._find_first_value(payload, ("breakout_down_10",)),
            "volume_zscore_20": self._find_first_value(payload, ("volume_zscore_20",)),
            "volume_ratio_20": self._find_first_value(payload, ("volume_ratio_20",)),
        }
        return self._compact_snapshot(snapshot)

    def _decision_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        final_action = self._find_first_value(payload, ("final_action",))
        confidence_pct = self._find_first_value(payload, ("confidence_pct",))
        final_action_float = self._coerce_float(final_action)
        if confidence_pct is None and final_action_float is not None:
            confidence_pct = abs(final_action_float) * 100.0
        snapshot = {
            "decision_mode": self._find_first_value(payload, ("decision_mode",)),
            "trade_direction": self._find_first_value(payload, ("trade_direction",)),
            "action_hold_threshold": self._find_first_value(payload, ("action_hold_threshold",)),
            "final_action": final_action_float if final_action_float is not None else final_action,
            "signal_bucket": self._find_first_value(payload, ("signal_bucket",)),
            "vote_bucket": self._find_first_value(payload, ("vote_bucket",)),
            "votes": self._find_first_value(payload, ("votes",)),
            "raw_actions": self._find_first_value(payload, ("raw_actions",)),
            "raw_action": self._find_first_value(payload, ("raw_action",)),
            "tie_hold": self._find_first_value(payload, ("tie_hold",)),
            "confidence_pct": confidence_pct,
            "reason": self._find_first_value(payload, ("decision_reason", "reason")),
        }
        return self._compact_snapshot(snapshot)

    def _filter_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "regime_valid_for_entry": self._find_first_value(payload, ("regime_valid_for_entry", "regime_valid")),
            "regime_dist_ema_240": self._find_first_value(payload, ("regime_dist_ema_240",)),
            "regime_vol_regime_z": self._find_first_value(payload, ("regime_vol_regime_z",)),
            "regime_thresholds": self._find_first_value(payload, ("regime_thresholds",)),
            "blocked_reason": self._find_first_value(payload, ("blocked_reason",)),
            "blocked_reason_human": self._find_first_value(payload, ("blocked_reason_human",)),
            "blocked_by_min_notional": self._find_first_value(payload, ("blocked_by_min_notional",)),
            "bumped_to_min_notional": self._find_first_value(payload, ("bumped_to_min_notional",)),
            "skipped_due_to_min_notional_risk": self._find_first_value(payload, ("skipped_due_to_min_notional_risk",)),
            "ignored_signal": self._find_first_value(payload, ("ignored_signal",)),
            "attempted_reversal": self._find_first_value(payload, ("attempted_reversal",)),
            "prevented_same_candle_reversal": self._find_first_value(payload, ("prevented_same_candle_reversal",)),
            "force_exit_only_by_tp_sl": self._find_first_value(payload, ("force_exit_only_by_tp_sl",)),
        }
        return self._compact_snapshot(snapshot)

    def _execution_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "action_taken": self._find_first_value(payload, ("action_taken",)),
            "opened_trade": self._find_first_value(payload, ("opened_trade", "opened_position")),
            "closed_trade": self._find_first_value(payload, ("closed_trade", "closed_position")),
            "position_side": self._find_first_value(payload, ("position_side",)),
            "position_label": self._find_first_value(payload, ("position_label",)),
            "position_is_open": self._find_first_value(payload, ("position_is_open",)),
            "position_size": self._find_first_value(payload, ("position_size", "volume")),
            "position_avg_entry_price": self._find_first_value(payload, ("position_avg_entry_price", "avg_entry_price")),
            "position_unrealized_pnl": self._find_first_value(payload, ("position_unrealized_pnl", "unrealized_pnl")),
            "position_time_in_bars": self._find_first_value(payload, ("position_time_in_bars", "time_in_position")),
            "entry_distance_from_ema_240": self._find_first_value(payload, ("entry_distance_from_ema_240",)),
            "stop_loss_price": self._find_first_value(payload, ("stop_loss_price",)),
            "take_profit_price": self._find_first_value(payload, ("take_profit_price",)),
            "exit_reason": self._find_first_value(payload, ("exit_reason",)),
            "error": self._find_first_value(payload, ("error",)),
            "response_ok": self._find_first_value(payload, ("response_ok", "ok")),
        }
        return self._compact_snapshot(snapshot)

    def _sizing_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "available_to_trade": self._find_first_value(payload, ("available_to_trade",)),
            "available_to_trade_source_field": self._find_first_value(payload, ("available_to_trade_source_field",)),
            "sizing_balance_used": self._find_first_value(payload, ("sizing_balance_used",)),
            "sizing_balance_source": self._find_first_value(payload, ("sizing_balance_source",)),
            "risk_pct": self._find_first_value(payload, ("risk_pct",)),
            "risk_amount": self._find_first_value(payload, ("risk_amount",)),
            "stop_loss_pct": self._find_first_value(payload, ("stop_loss_pct",)),
            "stop_distance_price": self._find_first_value(payload, ("stop_distance_price",)),
            "target_notional": self._find_first_value(payload, ("target_notional",)),
            "raw_volume": self._find_first_value(payload, ("raw_volume",)),
            "adjusted_volume": self._find_first_value(payload, ("adjusted_volume",)),
            "adjusted_notional": self._find_first_value(payload, ("adjusted_notional",)),
            "notional_value": self._find_first_value(payload, ("notional_value",)),
            "min_notional_usd": self._find_first_value(payload, ("min_notional_usd",)),
            "effective_risk_amount": self._find_first_value(payload, ("effective_risk_amount",)),
            "exceeds_target_risk": self._find_first_value(payload, ("exceeds_target_risk",)),
            "leverage_configured": self._find_first_value(payload, ("leverage_configured",)),
            "leverage_applied": self._find_first_value(payload, ("leverage_applied",)),
            "leverage_used": self._find_first_value(payload, ("leverage_used",)),
            "margin_estimated_consumed": self._find_first_value(payload, ("margin_estimated_consumed",)),
            "has_explicit_leverage_logic": self._find_first_value(payload, ("has_explicit_leverage_logic",)),
        }
        return self._compact_snapshot(snapshot)

    def _system_state_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "network": self._find_first_value(payload, ("network",)),
            "symbol": self._find_first_value(payload, ("symbol",)),
            "timeframe": self._find_first_value(payload, ("timeframe",)),
            "execution_mode": self._find_first_value(payload, ("execution_mode",)),
            "operational_mode": self._find_first_value(payload, ("operational_mode", "mode")),
            "decision_mode": self._find_first_value(payload, ("decision_mode",)),
            "order_slippage": self._find_first_value(payload, ("order_slippage",)),
            "model_slippage_pct": self._find_first_value(payload, ("model_slippage_pct",)),
        }
        return self._compact_snapshot(snapshot)

    def _compact_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if not self._has_meaningful_value(value):
                continue
            compact[key] = value
        return compact

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


class OpenAILogTranslator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.local = LocalLogInterpreter()
        self._cache: dict[str, HumanizedEvent] = {}
        self._lock = threading.Lock()
        self._prompt_builder = LauncherAIPromptBuilder(cfg)
        self._last_error: str | None = None

    @property
    def configured_enabled(self) -> bool:
        return bool(getattr(self.cfg.launcher, "openai_enabled", True))

    @property
    def dependency_ready(self) -> bool:
        return OpenAI is not None

    @property
    def enabled(self) -> bool:
        return self.configured_enabled and bool(self.api_key) and self.dependency_ready

    @property
    def model_name(self) -> str:
        model_name = str(self.cfg.launcher.openai_model or AI_MODEL_NAME).strip()
        return model_name or AI_MODEL_NAME

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
            "payload": event.metadata,
            "fallback": {
                "severity": fallback.severity,
                "message_human": fallback.message_human,
                "details": fallback.details,
                "color": fallback.color,
                "relevant": fallback.relevant,
                "event_code": fallback.event_code,
                "market_state": fallback.market_state,
                "model_interpretation": fallback.model_interpretation,
                "filters_diagnostic": fallback.filters_diagnostic,
                "execution_summary": fallback.execution_summary,
                "simple_summary": fallback.simple_summary,
            },
        }
        serialized = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        return sha1(serialized.encode("utf-8")).hexdigest()

    def translate(self, event: LauncherEvent, fallback: HumanizedEvent) -> HumanizedEvent:
        if not self.enabled:
            self._last_error = None
            return fallback

        key = self.cache_key(event, fallback)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        should_cache = True
        try:
            self._last_error = None
            client = OpenAI(api_key=self.api_key, timeout=float(self.cfg.launcher.openai_timeout_seconds))
            priority_metrics = self._extract_priority_metrics(event.metadata)
            prompt_payload = self._prompt_builder.build_user_payload(event, fallback, priority_metrics)
            raw_output = self._request_translation_output(client, prompt_payload)
            translated = self._parse_translation_payload(raw_output)
            humanized_details = translated["humanized_details"]
            preferred_model_interpretation = self._format_priority_model_interpretation(priority_metrics)
            if self._should_override_model_interpretation(
                humanized_details.get("model_interpretation", ""),
                preferred_model_interpretation,
            ):
                humanized_details = dict(humanized_details)
                humanized_details["model_interpretation"] = preferred_model_interpretation
            candidate = HumanizedEvent(
                source=fallback.source,
                event_type=fallback.event_type,
                raw_line=fallback.raw_line,
                payload=dict(event.metadata or fallback.payload or {}),
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
                market_state=str(humanized_details.get("market_state", fallback.market_state)),
                model_interpretation=str(
                    humanized_details.get("model_interpretation", fallback.model_interpretation)
                ),
                filters_diagnostic=str(humanized_details.get("filters_diagnostic", fallback.filters_diagnostic)),
                execution_summary=str(humanized_details.get("execution_summary", fallback.execution_summary)),
                simple_summary=str(humanized_details.get("simple_summary", fallback.simple_summary)),
            )
            result = candidate if self._is_clean_translation(candidate, event.message_raw) else fallback
        except Exception as exc:
            should_cache = False
            self._last_error = str(exc)
            logger.error(
                "Falha ao traduzir evento com OpenAI (type=%s code=%s model=%s): %s",
                event.type,
                event.event_code,
                self.model_name,
                exc,
                exc_info=True,
            )
            result = fallback

        if should_cache:
            with self._lock:
                # Prevencao de memory leak: limitar o cache a 200 itens.
                if len(self._cache) >= 200:
                    # Remove os 50 itens mais antigos para liberar espaco.
                    velhos = list(self._cache.keys())[:50]
                    for k in velhos:
                        del self._cache[k]

                self._cache[key] = result
        return result

    def status_snapshot(self) -> dict[str, Any]:
        reason = None
        if not self.configured_enabled:
            reason = "disabled_in_config"
        elif not self.api_key:
            reason = "missing_api_key"
        elif not self.dependency_ready:
            reason = "openai_dependency_unavailable"
        return {
            "enabled": self.enabled,
            "configured_enabled": self.configured_enabled,
            "api_key_present": bool(self.api_key),
            "dependency_ready": self.dependency_ready,
            "client_ready": bool(self.enabled),
            "model": self.model_name,
            "api_key_env": self.cfg.launcher.openai_api_key_env,
            "reason": reason,
            "last_error": self._last_error,
        }

    def _is_clean_translation(self, event: HumanizedEvent, raw_log: str) -> bool:
        message = (event.message_human or "").strip()
        lowered = message.lower()
        raw_lowered = (raw_log or "").strip().lower()
        if not message:
            return False
        if len(message) > 160:
            return False
        if message.lower() == raw_lowered:
            return False
        forbidden = (
            "wallet=",
            "base_url",
            "traceback",
            "requests.",
            "0x",
            "http://",
            "https://",
            "{",
            "}",
        )
        return not any(fragment in lowered for fragment in forbidden)

    def _extract_priority_metrics(self, metadata: Any) -> dict[str, Any]:
        extracted = {
            "ensemble_score": self._find_metadata_key(metadata, "ensemble_score"),
            "model_weights": self._find_metadata_key(metadata, "model_weights"),
            "indicators": self._find_metadata_key(metadata, "indicators"),
            "decision_votes": self._extract_decision_votes(self._find_metadata_key(metadata, "decision")),
        }
        return {key: value for key, value in extracted.items() if value is not None}

    def _find_metadata_key(self, payload: Any, target_key: str) -> Any:
        if isinstance(payload, dict):
            if target_key in payload:
                return payload[target_key]
            for value in payload.values():
                found = self._find_metadata_key(value, target_key)
                if found is not None:
                    return found
            return None
        if isinstance(payload, list):
            for item in payload:
                found = self._find_metadata_key(item, target_key)
                if found is not None:
                    return found
        return None

    def _format_priority_model_interpretation(self, priority_metrics: dict[str, Any]) -> str:
        lines: list[str] = []

        decision_votes = priority_metrics.get("decision_votes")
        if not isinstance(decision_votes, dict) or not decision_votes:
            return WAITING_FEATURES_MESSAGE

        for model_name, vote_value in list(decision_votes.items())[:8]:
            lines.append(f"- Model {model_name}: {self._format_vote_value(vote_value)}")

        ensemble_score = priority_metrics.get("ensemble_score")
        if ensemble_score is not None:
            lines.append(f"- Ensemble score: {self._format_metric_value(ensemble_score)}")
        else:
            lines.append("- Ensemble score: N/A")

        model_weights = priority_metrics.get("model_weights")
        if model_weights is not None:
            lines.append(f"- Model weights: {self._format_model_weights(model_weights)}")
        else:
            lines.append("- Model weights: N/A")

        indicators = priority_metrics.get("indicators")
        if indicators is not None:
            lines.append(f"- Indicators: {self._format_indicators(indicators)}")
        else:
            lines.append("- Indicators: N/A")

        return "\n".join(lines)

    def _extract_decision_votes(self, decision_payload: Any) -> dict[str, Any] | None:
        if not isinstance(decision_payload, dict):
            return None

        for preferred_key in ("model_votes", "votes_by_model", "per_model_votes", "scores_by_model"):
            preferred_value = decision_payload.get(preferred_key)
            if isinstance(preferred_value, dict):
                normalized_votes = self._normalize_vote_mapping(preferred_value)
                if normalized_votes:
                    return normalized_votes

        collected: dict[str, Any] = {}
        self._collect_decision_votes(decision_payload, collected)
        return collected or None

    def _normalize_vote_mapping(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[str(key)] = value
        return normalized

    def _collect_decision_votes(self, payload: Any, result: dict[str, Any], context: str = "") -> None:
        if len(result) >= 8:
            return
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key)
                nested_context = f"{context}.{key_text}" if context else key_text
                if isinstance(value, (int, float)) and not isinstance(value, bool) and self._looks_like_model_vote(key_text, context):
                    result[key_text] = value
                else:
                    self._collect_decision_votes(value, result, nested_context)
                if len(result) >= 8:
                    return
        elif isinstance(payload, list):
            for item in payload:
                self._collect_decision_votes(item, result, context)
                if len(result) >= 8:
                    return

    def _looks_like_model_vote(self, key_text: str, context: str) -> bool:
        lowered_key = key_text.lower()
        lowered_context = context.lower()
        if lowered_key in {"ensemble_score", "confidence", "score", "final_score", "weight"}:
            return False
        if any(fragment in lowered_context for fragment in ("model_votes", "votes_by_model", "per_model_votes", "scores_by_model")):
            return True
        if lowered_key.startswith(("ppo_", "dqn_", "model_", "agent_")):
            return True
        return bool(re.match(r"^[a-z]+(?:_[a-z0-9]+)+$", lowered_key))

    def _should_override_model_interpretation(self, current_text: str, preferred_text: str) -> bool:
        if not preferred_text:
            return False

        if preferred_text == WAITING_FEATURES_MESSAGE:
            return True

        normalized_current = str(current_text or "").strip()
        if not normalized_current:
            return True

        lowered = normalized_current.lower()
        generic_fragments = (
            "segue analisando",
            "continua analisando",
            "continua avaliando",
            "modelo segue",
            "segue monitorando",
            "mercado em observacao",
        )
        if any(fragment in lowered for fragment in generic_fragments):
            return True

        if not re.search(r"\d", normalized_current) and re.search(r"\d", preferred_text):
            return True
        return False

    def _format_model_weights(self, weights: Any) -> str:
        if isinstance(weights, dict):
            parts = [f"{key}={self._format_metric_value(value)}" for key, value in weights.items()]
            return " | ".join(parts) if parts else "indisponivel"
        if isinstance(weights, list):
            parts = [self._format_metric_value(item) for item in weights[:8]]
            return " | ".join(part for part in parts if part) or "indisponivel"
        return self._format_metric_value(weights)

    def _format_indicators(self, indicators: Any) -> str:
        flattened: list[tuple[str, str]] = []
        self._collect_numeric_pairs(indicators, flattened)
        if flattened:
            return " | ".join(f"{key}={value}" for key, value in flattened[:8])
        return self._format_metric_value(indicators)

    def _format_vote_value(self, value: Any) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
            if -1.0 <= numeric_value <= 1.0:
                return f"{numeric_value * 100:.0f}%"
        return self._format_metric_value(value)

    def _collect_numeric_pairs(
        self,
        payload: Any,
        result: list[tuple[str, str]],
        prefix: str = "",
    ) -> None:
        if len(result) >= 8:
            return
        if isinstance(payload, dict):
            for key, value in payload.items():
                nested_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._collect_numeric_pairs(value, result, nested_prefix)
                if len(result) >= 8:
                    return
            return
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                nested_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                self._collect_numeric_pairs(item, result, nested_prefix)
                if len(result) >= 8:
                    return
            return
        if isinstance(payload, bool):
            return
        if isinstance(payload, (int, float)) and prefix:
            result.append((prefix, self._format_metric_value(payload)))

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if isinstance(value, dict):
            compact = json.dumps(value, ensure_ascii=False, sort_keys=True)
            return compact[:157] + "..." if len(compact) > 160 else compact
        if isinstance(value, list):
            compact = ", ".join(self._format_metric_value(item) for item in value[:6])
            return compact or "indisponivel"
        normalized = str(value).strip()
        return normalized or "indisponivel"

    def _request_translation_output(self, client: Any, prompt_payload: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self._prompt_builder.build_system_prompt()},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        last_error: Exception | None = None
        for response_format in self._response_formats():
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                )
                return self._extract_chat_completion_text(response)
            except Exception as exc:
                last_error = exc
                if response_format.get("type") == "json_schema" and self._should_retry_with_json_mode(exc):
                    logger.warning(
                        "json_schema indisponivel para o modelo %s; repetindo em json_object: %s",
                        self.model_name,
                        exc,
                    )
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI nao retornou resposta para a traducao do evento")

    def _response_formats(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "launcher_event_summary",
                    "schema": AI_OUTPUT_SCHEMA,
                    "strict": True,
                },
            },
            {"type": "json_object"},
        )

    def _should_retry_with_json_mode(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            fragment in message
            for fragment in (
                "json_schema",
                "response_format",
                "schema",
                "unsupported",
                "not supported",
                "invalid parameter",
                "invalid schema",
            )
        )

    def _extract_chat_completion_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("OpenAI chat completion retornou choices vazios")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("OpenAI chat completion retornou mensagem vazia")

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ValueError(f"OpenAI recusou a traducao: {refusal}")

        raw_output = self._coerce_message_content_to_text(getattr(message, "content", None))
        if not raw_output:
            raise ValueError("OpenAI chat completion retornou content vazio")
        return raw_output

    def _coerce_message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            text_value = self._extract_text_part(part)
            if text_value:
                text_parts.append(text_value)
        return "".join(text_parts).strip()

    def _extract_text_part(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            text_value = part.get("text")
            if isinstance(text_value, str):
                return text_value
            if isinstance(text_value, dict):
                nested_value = text_value.get("value")
                if isinstance(nested_value, str):
                    return nested_value
            return ""

        text_value = getattr(part, "text", None)
        if isinstance(text_value, str):
            return text_value
        nested_value = getattr(text_value, "value", None)
        if isinstance(nested_value, str):
            return nested_value
        return ""

    def _parse_translation_payload(self, raw_output: str) -> dict[str, Any]:
        normalized_output = str(raw_output or "").strip()
        if not normalized_output:
            raise ValueError("empty translation payload")
        if normalized_output.startswith("```"):
            normalized_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized_output, flags=re.IGNORECASE)
            normalized_output = normalized_output.strip()

        payload = json.loads(normalized_output)
        if not isinstance(payload, dict):
            raise ValueError("translation payload is not an object")

        severity = str(payload.get("severity", "")).strip().lower()
        color = str(payload.get("color", "")).strip().lower()
        message = str(payload.get("message", "")).strip()
        relevant = payload.get("relevant")
        details = payload.get("details")
        humanized_details = payload.get("humanized_details")

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
        if not isinstance(humanized_details, dict):
            raise ValueError("invalid humanized_details")

        parsed_humanized_details: dict[str, str] = {}
        default_humanized_details = {
            "market_state": "Dados insuficientes",
            "model_interpretation": "N/A",
            "filters_diagnostic": "N/A",
            "execution_summary": "N/A",
            "simple_summary": "N/A",
        }
        for key in (
            "market_state",
            "model_interpretation",
            "filters_diagnostic",
            "execution_summary",
            "simple_summary",
        ):
            value = humanized_details.get(key)
            if value is None:
                parsed_humanized_details[key] = default_humanized_details[key]
                continue
            if not isinstance(value, str):
                raise ValueError(f"invalid {key}")
            normalized_value = self._sanitize_humanized_text(value)
            if not normalized_value:
                parsed_humanized_details[key] = default_humanized_details[key]
                continue
            parsed_humanized_details[key] = normalized_value

        return {
            "severity": severity,
            "message": message,
            "details": details,
            "color": color,
            "relevant": bool(relevant),
            "humanized_details": parsed_humanized_details,
        }

    def _sanitize_humanized_text(self, value: str) -> str:
        normalized_value = str(value or "").strip()
        if normalized_value.startswith("```"):
            normalized_value = re.sub(
                r"^```(?:json|html|markdown|md|text)?\s*|\s*```$",
                "",
                normalized_value,
                flags=re.IGNORECASE,
            ).strip()
        return normalized_value

    def _normalize_cache_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        compact = re.sub(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|z)?", "<ts>", compact)
        compact = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<time>", compact)
        return compact
