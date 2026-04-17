from __future__ import annotations

import json
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


class LauncherAIPromptBuilder:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def build_system_prompt(self) -> str:
        return (
            "Voce e a camada de interpretacao inteligente do launcher pessoal de um bot de trading.\n\n"
            "INSTRUCAO CRITICA:\n"
            "- Voce deve agir como um analista de trading relatando as metricas de forma limpa, direta, estruturada e SEM EMOJIS.\n\n"
            "CONTEXTO DO SISTEMA:\n"
            "- A interface e um launcher em PySide6 usado para operar e monitorar um bot pessoal.\n"
            "- O bot opera na Hyperliquid e usa decisao por ensemble validada em treino e backtest.\n"
            "- Os logs chegam como eventos estruturados do runtime, health check, smoke test e interacoes do launcher.\n"
            "- Seu papel nao e repetir logs tecnicos. Seu papel e explicar o que aconteceu e por que aconteceu.\n\n"
            "O QUE VOCE DEVE FAZER:\n"
            "- transformar logs tecnicos em leitura humana e operacional\n"
            "- explicar o contexto do bot quando houver decisao, bloqueio, erro, check ou mudanca de estado\n"
            "- apontar o motivo principal da acao ou da nao acao\n"
            "- manter a mensagem curta, mas usar details quando precisar explicar o motivo do comportamento\n"
            "- analisar todos os snapshots do ciclo quando estiverem disponiveis: market_snapshot, feature_snapshot/features_snapshot, decision, filters e execution\n"
            "- gerar o objeto humanized_details para preencher a aba Detalhes da interface\n"
            "- tratar a mensagem como algo que aparecera na interface principal do launcher\n\n"
            "HUMANIZED_DETAILS:\n"
            "- humanized_details.market_state: descreva estado do mercado com precos, RSI, distancia de EMA e volume quando houver\n"
            "- humanized_details.model_interpretation: descreva a leitura do modelo, bucket, contagem de votos, confianca e motivo principal\n"
            "- humanized_details.filters_diagnostic: explique bloqueios, validacao do regime e diagnostico dos filtros\n"
            "- humanized_details.execution_summary: resuma status de execucao e posicao, incluindo aberta, fechada, mantida ou FLAT\n"
            "- humanized_details.simple_summary: gere um resumo curto e direto de toda a situacao\n\n"
            "REGRAS:\n"
            "- nunca repetir o log cru\n"
            "- nunca expor wallet, base_url, ids internos ou JSON bruto\n"
            "- nunca responder como chat; responda apenas no JSON exigido\n"
            "- message deve ser curta, direta e util para operacao\n"
            "- details pode ser nulo ou uma explicacao curta do motivo\n"
            "- priorize por que o bot operou, nao operou, bloqueou, confirmou conexao ou entrou em erro\n"
            "- nunca deixe os campos de humanized_details vazios; quando faltar dado, escreva uma frase curta informando isso\n"
            "- se o evento nao for relevante para o usuario final, marque relevant=false\n\n"
            "CLASSIFICACAO:\n"
            "- info: atualizacao normal\n"
            "- warning: atencao ou instabilidade sem falha definitiva\n"
            "- error: erro real ou falha confirmada\n"
            "- execution: decisao, operacao, manutencao de posicao, smoke test\n"
            "- risk: eventos de risco, fechamento manual, kill switch\n"
            "- blocked: entrada ou acao impedida\n\n"
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
            '  "details": "explicacao curta ou null",\n'
            '  "color": "blue | yellow | red | purple | orange | gray",\n'
            '  "relevant": true,\n'
            '  "humanized_details": {\n'
            '    "market_state": "texto limpo sobre o mercado",\n'
            '    "model_interpretation": "texto limpo sobre a leitura do modelo",\n'
            '    "filters_diagnostic": "texto limpo sobre filtros e bloqueios",\n'
            '    "execution_summary": "texto limpo sobre a execucao",\n'
            '    "simple_summary": "resumo curto e direto"\n'
            "  }\n"
            "}"
        )

    def build_user_payload(self, event: LauncherEvent, fallback: HumanizedEvent) -> dict[str, Any]:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        feature_snapshot = metadata.get("feature_snapshot")
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
                "cycle_metrics": {
                    "market_snapshot": metadata.get("market_snapshot"),
                    "feature_snapshot": feature_snapshot,
                    "features_snapshot": feature_snapshot if feature_snapshot is not None else metadata.get("features_snapshot"),
                    "decision": metadata.get("decision"),
                    "filters": metadata.get("filters"),
                    "execution": metadata.get("execution"),
                    "sizing": metadata.get("sizing"),
                    "system_state": metadata.get("system_state"),
                },
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
                "preencha humanized_details usando todas as metricas do ciclo disponiveis."
            ),
        }


class OpenAILogTranslator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.local = LocalLogInterpreter()
        self._cache: dict[str, HumanizedEvent] = {}
        self._lock = threading.Lock()
        self._prompt_builder = LauncherAIPromptBuilder(cfg)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and OpenAI is not None)

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
            return fallback

        key = self.cache_key(event, fallback)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            client = OpenAI(api_key=self.api_key, timeout=float(self.cfg.launcher.openai_timeout_seconds))
            prompt_payload = self._prompt_builder.build_user_payload(event, fallback)
            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._prompt_builder.build_system_prompt(),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(prompt_payload, ensure_ascii=False),
                            }
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "launcher_event_summary",
                        "schema": AI_OUTPUT_SCHEMA,
                        "strict": True,
                    }
                },
            )
            translated = self._parse_translation_payload(response.output_text)
            humanized_details = translated["humanized_details"]
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
                market_state=str(humanized_details.get("market_state", fallback.market_state)),
                model_interpretation=str(
                    humanized_details.get("model_interpretation", fallback.model_interpretation)
                ),
                filters_diagnostic=str(humanized_details.get("filters_diagnostic", fallback.filters_diagnostic)),
                execution_summary=str(humanized_details.get("execution_summary", fallback.execution_summary)),
                simple_summary=str(humanized_details.get("simple_summary", fallback.simple_summary)),
            )
            result = candidate if self._is_clean_translation(candidate, event.message_raw) else fallback
        except Exception:
            result = fallback

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
        return {
            "enabled": True,
            "api_key_present": bool(self.api_key),
            "client_ready": bool(self.enabled),
            "model": self.model_name,
            "api_key_env": self.cfg.launcher.openai_api_key_env,
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

    def _parse_translation_payload(self, raw_output: str) -> dict[str, Any]:
        payload = json.loads(raw_output)
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
        for key in (
            "market_state",
            "model_interpretation",
            "filters_diagnostic",
            "execution_summary",
            "simple_summary",
        ):
            value = humanized_details.get(key)
            if not isinstance(value, str):
                raise ValueError(f"invalid {key}")
            normalized_value = value.strip()
            if not normalized_value:
                raise ValueError(f"empty {key}")
            parsed_humanized_details[key] = normalized_value

        return {
            "severity": severity,
            "message": message,
            "details": details,
            "color": color,
            "relevant": bool(relevant),
            "humanized_details": parsed_humanized_details,
        }

    def _normalize_cache_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        compact = re.sub(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|z)?", "<ts>", compact)
        compact = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<time>", compact)
        return compact
