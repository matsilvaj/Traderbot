# -*- coding: utf-8 -*-
import asyncio
import html
import json
import os
from collections import deque
import subprocess
import sys
from datetime import date, datetime, time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
LOGS_DIR = REPO_ROOT / "logs"
TARGET_NETWORK = "mainnet"
TARGET_EXECUTION_MODE = "exchange"

ACTIVE_RUNTIME_STATUSES = {
    "starting",
    "running",
    "guard_closing_position",
    "guard_close_retry_pending",
}

RUNTIME_STATUS_LABELS = {
    "starting": "INICIANDO",
    "running": "RODANDO",
    "stopping": "PARANDO",
    "stopped": "PARADO",
    "stopped_after_error": "PARADO COM ERRO",
    "closed_by_guard": "ENCERRADO PELO GUARD",
    "guard_closing_position": "GUARD FECHANDO POSIÇÃO",
    "guard_close_retry_pending": "GUARD RETENTANDO",
    "guard_checked_no_position": "GUARD SEM POSIÇÃO",
}

load_dotenv(REPO_ROOT / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "SEU_TOKEN_AQUI")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI")

trade_process = None


def verificar_autorizacao(update: Update) -> bool:
    return str(update.effective_chat.id) == str(CHAT_ID)


def carregar_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def salvar_config(config_data: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False, allow_unicode=True)


def _config_symbol() -> str:
    cfg = carregar_config()
    return str(cfg.get("hyperliquid", {}).get("symbol", "BTC") or "BTC").strip().upper()


def _runtime_state_path(network: str = TARGET_NETWORK) -> Path:
    symbol = _config_symbol().lower()
    return LOGS_DIR / f"runtime_state_{str(network).strip().lower()}_{symbol}.json"


def ler_status_runtime(network: str = TARGET_NETWORK) -> dict | None:
    path = _runtime_state_path(network)
    candidate_paths = [path]
    candidate_paths.extend(
        sorted(
            LOGS_DIR.glob(f"state_{str(network).strip().lower()}_*.json"),
            key=os.path.getmtime,
            reverse=True,
        )
    )

    for candidate in candidate_paths:
        if not candidate.exists():
             continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _num(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def v_num(valor, default: float = 0.0) -> float:
    parsed = _num(valor, None)
    return default if parsed is None else parsed


def _parse_iso_datetime(value) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_local_dt(value, default: str = "--") -> str:
    if value is None:
        return default
    try:
        if value.tzinfo is not None:
            value = value.astimezone()
        return value.strftime("%d/%m %H:%M:%S")
    except Exception:
        return default


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--"
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, sec = divmod(total, 60)
        return f"{minutes}m {sec}s"
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def _format_usd(value, default: str = "--") -> str:
    parsed = _num(value, None)
    if parsed is None:
        return default
    return f"${parsed:,.2f}"


def _format_price(value, default: str = "--") -> str:
    parsed = _num(value, None)
    if parsed is None:
        return default
    return f"${parsed:,.2f}"


def _format_plain_number(value, digits: int = 2, default: str = "--") -> str:
    parsed = _num(value, None)
    if parsed is None:
        return default
    return f"{parsed:.{digits}f}"


def _format_ratio_pct(value, digits: int = 2, default: str = "--") -> str:
    parsed = _num(value, None)
    if parsed is None:
        return default
    return f"{parsed * 100:.{digits}f}%"


def _format_pct(value, digits: int = 1, default: str = "--") -> str:
    parsed = _num(value, None)
    if parsed is None:
        return default
    return f"{parsed:.{digits}f}%"


def _human_signal(bucket: str | None) -> str:
    normalized = str(bucket or "").strip().lower()
    if normalized == "buy":
        return "COMPRAR"
    if normalized == "sell":
        return "VENDER"
    return "AGUARDAR"


def _market_status_label(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "Aguardando dados"

    filters = payload.get("filters") or {}
    regime_ok = filters.get("regime_valid_for_entry", payload.get("regime_valid"))
    dist_ema = _num(
        filters.get("regime_dist_ema_240", payload.get("regime_dist_ema_240")),
        None,
    )

    if regime_ok is False:
        return "Lateral / bloqueado pelo regime"
    if dist_ema is None:
        return "Aguardando leitura"
    if dist_ema > 0:
        return "Alta"
    if dist_ema < 0:
        return "Baixa"
    return "Favorável"


def _human_block_label(reason: str | None) -> str:
    mapping = {
        "regime_filter": "Condição de baixo volume ou lateralização",
        "cooldown": "Aguardando tempo de resfriamento (cooldown)",
        "force_exit_only_by_tp_sl": "Posição aberta sem TP/SL confirmado na corretora",
        "native_tp_sl_unprotected": "Posição aberta sem TP/SL nativo confirmado na corretora",
        "min_notional_risk": "Risco abaixo do mínimo exigido pela corretora",
        "exchange_position_present": "Já existe uma posição aberta na corretora",
        "duplicate_cycle_order": "Bloqueio para evitar ordem duplicada nesta vela",
        "order_cooldown": "Aguardando tempo mínimo entre ordens",
        "open_position_locked": "Sinal ignorado porque já existe uma posição aberta",
        "duplicate_cycle": "Bloqueio para evitar entradas duplicadas na mesma vela",
    }
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "--"
    return mapping.get(normalized, normalized.replace("_", " "))


def _clip(text: str | None, limit: int = 180) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _pid_is_running(pid) -> bool:
    pid_int = int(_num(pid, 0) or 0)
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _runtime_stale_after_seconds() -> float:
    cfg = carregar_config()
    execution = cfg.get("execution", {})
    heartbeat = max(5.0, float(execution.get("runtime_guard_heartbeat_interval_seconds", 15) or 15))
    return max(20.0, heartbeat * 4.0)


def _extract_cycle_payload(runtime_state: dict | None) -> dict:
    if not isinstance(runtime_state, dict):
        return {}
    payload = runtime_state.get("last_cycle_payload")
    return payload if isinstance(payload, dict) else {}


def _build_runtime_snapshot(runtime_state: dict | None, exchange_status: dict | None = None) -> dict:
    payload = _extract_cycle_payload(runtime_state)
    status_code = str((runtime_state or {}).get("status") or "").strip().lower()
    heartbeat_at = _parse_iso_datetime((runtime_state or {}).get("last_heartbeat_at")) or _parse_iso_datetime(
        (runtime_state or {}).get("started_at")
    )
    heartbeat_age_seconds = None
    if heartbeat_at is not None:
        if heartbeat_at.tzinfo is not None:
            heartbeat_age_seconds = max(0.0, (datetime.now(heartbeat_at.tzinfo) - heartbeat_at).total_seconds())
        else:
            heartbeat_age_seconds = max(0.0, (datetime.now() - heartbeat_at).total_seconds())

    runtime_pid = int(_num((runtime_state or {}).get("runtime_pid"), 0) or 0)
    pid_alive = _pid_is_running(runtime_pid) if runtime_pid > 0 else False
    stale_after = _runtime_stale_after_seconds()
    runtime_alive = bool(runtime_state) and status_code in ACTIVE_RUNTIME_STATUSES
    if runtime_alive and heartbeat_age_seconds is not None and heartbeat_age_seconds > stale_after:
        runtime_alive = False
    if runtime_alive and runtime_pid > 0 and not pid_alive:
        runtime_alive = False

    if runtime_alive:
        runtime_label = RUNTIME_STATUS_LABELS.get(status_code, "RODANDO")
        runtime_emoji = "??" if status_code in {"running", "starting"} else "??"
    elif not runtime_state:
        runtime_label = "PARADO"
        runtime_emoji = "??"
    elif status_code in RUNTIME_STATUS_LABELS:
        runtime_label = RUNTIME_STATUS_LABELS[status_code]
        runtime_emoji = "??" if "guard" in status_code or status_code == "stopping" else "??"
    else:
        runtime_label = "SEM HEARTBEAT"
        runtime_emoji = "??"
        
    situation_label = runtime_label
    situation_message = "Nenhum runtime compartilhado encontrado."
    
    blocked_reason_human = str(
        payload.get("blocked_reason_human")
        or (payload.get("filters") or {}).get("blocked_reason_human")
        or ""
    ).strip()
    blocked_reason = str(
        payload.get("blocked_reason")
        or (payload.get("filters") or {}).get("blocked_reason")
        or ""
    ).strip()
    payload_error = str(payload.get("error") or "").strip()
    manual_protection_required = bool(payload.get("manual_protection_required", False))
    manual_protection_message = str(payload.get("manual_protection_message") or "").strip()
    position_label = str(payload.get("position_label") or "FLAT").upper()
    signal_bucket = str(
        (payload.get("decision") or {}).get("vote_bucket")
        or payload.get("vote_bucket")
        or (payload.get("decision") or {}).get("signal_bucket")
        or ""
    ).strip()

    exchange_position_open = bool((exchange_status or {}).get("position_is_open", False))
    native_tp_sl_protected = bool((exchange_status or {}).get("native_tp_sl_protected", True))

    if exchange_position_open and not runtime_alive:
        situation_label = "ALERTA"
        situation_message = "Existe posição aberta na exchange sem heartbeat ativo do runtime."
    elif exchange_position_open and not native_tp_sl_protected:
        situation_label = "ALERTA"
        situation_message = "Existe posição aberta sem TP/SL nativo confirmado na corretora."
    elif manual_protection_required:
        situation_label = "AVISO"
        situation_message = manual_protection_message or "Proteção manual de TP/SL necessária."
    elif payload_error:
        situation_label = "ERRO"
        situation_message = payload_error
    elif blocked_reason_human or blocked_reason:
        situation_label = "BLOQUEADO"
        situation_message = blocked_reason_human or _human_block_label(blocked_reason)
    elif payload.get("position_is_open"):
        situation_label = position_label
        situation_message = f"Posição {position_label} em andamento."
    elif payload:
        situation_label = "AGUARDANDO SINAL" if _human_signal(signal_bucket) == "AGUARDAR" else "FLAT"
        situation_message = "Avaliando o mercado. Sem oportunidade no momento."
    elif runtime_alive:
        situation_label = "INICIANDO" if status_code == "starting" else "RODANDO"
        situation_message = (
            "Engine headless ativa. Aguardando o primeiro ciclo."
            if status_code == "starting"
            else "Engine headless ativa em background."
        )
    elif status_code == "stopped_after_error":
        situation_label = "ERRO"
        situation_message = "Runtime parado após falha."
    elif status_code == "closed_by_guard":
        situation_label = "GUARD"
        situation_message = "Runtime guard encerrou a posição após perder o heartbeat."
    elif status_code == "guard_checked_no_position":
        situation_label = "GUARD"
        situation_message = "Runtime guard foi acionado, mas não havia posição aberta."
    elif status_code == "stopping":
        situation_label = "PARANDO"
        situation_message = "Desligamento do runtime em andamento."
    elif heartbeat_age_seconds is not None and heartbeat_age_seconds > stale_after:
        situation_label = "SEM HEARTBEAT"
        situation_message = f"Último heartbeat recebido há {_format_age(heartbeat_age_seconds)}."
    else:
        situation_label = "PARADO"
        situation_message = "Bot desligado."
        
    return {
        "alive": runtime_alive,
        "runtime_label": runtime_label,
        "runtime_emoji": runtime_emoji,
        "status_code": status_code or "--",
        "situation_label": situation_label,
        "situation_message": situation_message,
        "heartbeat_at": heartbeat_at,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "runtime_pid": runtime_pid,
        "pid_alive": pid_alive,
        "bar_timestamp": _parse_iso_datetime(payload.get("bar_timestamp") or (runtime_state or {}).get("last_bar_timestamp")),
        "event_code": str(payload.get("event_code") or (runtime_state or {}).get("last_runtime_event_code") or "--"),
        "payload": payload,
        "last_cycle_error": str((runtime_state or {}).get("last_cycle_error") or "").strip(),
        "last_cycle_error_at": _parse_iso_datetime((runtime_state or {}).get("last_cycle_error_at")),
        "guard_result": (runtime_state or {}).get("guard_result") if isinstance((runtime_state or {}).get("guard_result"), dict) else {},
    }


def calcular_estatisticas_diarias(network: str = TARGET_NETWORK) -> dict:
    hoje = date.today().isoformat()
    stats = {
        "operacoes": 0,
        "abertas": 0,
        "fechadas": 0,
        "bloqueadas": 0,
        "vitorias": 0,
        "derrotas": 0,
        "pnl_fechado": 0.0,
        "ultimo_evento": "--",
        "ultima_hora": "--",
    }

    target_network = str(network).strip().lower()

    for arq in LOGS_DIR.glob("*.log"):
        try:
            with open(arq, "r", encoding="utf-8") as f:
                for linha in f:
                    if hoje not in linha or "Ciclo runtime HL |" not in linha:
                        continue
                    try:
                        prefix, json_blob = linha.split("Ciclo runtime HL |", 1)
                        payload = json.loads(json_blob.strip())
                    except Exception:
                        continue

                    if str(payload.get("network") or "").strip().lower() != target_network:
                        continue

                    blocked_reason = payload.get("blocked_reason")
                    opened_trade = bool(payload.get("opened_trade"))
                    closed_trade = bool(payload.get("closed_trade"))

                    if blocked_reason:
                        stats["bloqueadas"] += 1
                    if opened_trade:
                        stats["abertas"] += 1
                        stats["operacoes"] += 1
                 
                    if closed_trade:
                        stats["fechadas"] += 1
                        pnl = v_num(payload.get("position_unrealized_pnl"))
                        stats["pnl_fechado"] += pnl
                  
                        if pnl > 0:
                            stats["vitorias"] += 1
                        else:
                            stats["derrotas"] += 1

                    stats["ultimo_evento"] = str(
                        payload.get("event_code")
                        or payload.get("execution_action")
                        or "--"
                    )
                    stats["ultima_hora"] = prefix.split(" |", 1)[0].strip()
        except Exception:
            continue

    total_encerradas = stats["vitorias"] + stats["derrotas"]
    stats["winrate"] = (stats["vitorias"] / total_encerradas * 100.0) if total_encerradas > 0 else 0.0
    return stats


def _extract_hyperliquid_json(output_text: str) -> dict | None:
    marker = "Status Hyperliquid: "
    candidate = None
    for line in output_text.splitlines():
        if marker in line:
            candidate = line.split(marker, 1)[1].strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def consultar_status_hyperliquid(network: str = TARGET_NETWORK) -> tuple[dict | None, str | None]:
    cmd = [
        sys.executable,
        "-m",
        "traderbot.main",
        "--config",
        str(CONFIG_PATH),
        "--network-override",
        str(network),
        "--execution-mode-override",
        TARGET_EXECUTION_MODE,
        "check-hyperliquid",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT),
        )
        stdout, stderr = await process.communicate()
    except Exception as exc:
        return None, f"Falha ao executar check-hyperliquid: {exc}"

    combined_output = "\n".join(
        [
            stdout.decode("utf-8", errors="ignore"),
            stderr.decode("utf-8", errors="ignore"),
        ]
    )
    payload = _extract_hyperliquid_json(combined_output)

    if payload is None:
        if process.returncode != 0:
            return None, f"check-hyperliquid retornou código {process.returncode}"
        return None, "Não foi possível interpretar a resposta da Hyperliquid"

    return payload, None


async def coletar_contexto_painel(network: str = TARGET_NETWORK) -> dict:
    cfg = carregar_config()
    runtime_state = ler_status_runtime(network)

    stats_task = asyncio.to_thread(calcular_estatisticas_diarias, network)
    exchange_task = asyncio.create_task(consultar_status_hyperliquid(network))
    stats, exchange_result = await asyncio.gather(stats_task, exchange_task)

    exchange_status, exchange_error = exchange_result
    runtime = _build_runtime_snapshot(runtime_state, exchange_status)

    return {
        "config": cfg,
        "runtime_state": runtime_state,
        "runtime": runtime,
        "exchange_status": exchange_status,
        "exchange_error": exchange_error,
        "stats": stats,
    }


def _exchange_health_label(exchange_status: dict | None) -> str:
    if not isinstance(exchange_status, dict):
        return "indisponível"

    parts = []
    parts.append("conectada" if exchange_status.get("connected") else "sem conexão")
    parts.append("trade liberado" if exchange_status.get("can_trade") else "trade bloqueado")
    if exchange_status.get("wallet_loaded") is False:
        parts.append("wallet não carregada")
    if exchange_status.get("symbol_found") is False:
        parts.append("símbolo não encontrado")
    if bool((exchange_status.get("account_summary") or {}).get("defensive", False)):
        parts.append("modo defensivo")
    return " | ".join(parts)


def _protection_label(exchange_status: dict | None) -> str:
    if not isinstance(exchange_status, dict):
        return "Sem consulta"

    native = exchange_status.get("native_tp_sl") or {}
    if native.get("enabled") is False:
        return "Desativada"
    if not exchange_status.get("position_is_open"):
        return "Sem posição aberta"
    if exchange_status.get("native_tp_sl_protected"):
        return "OK"

    missing = []
    if not native.get("has_stop_loss"):
        missing.append("SL")
    if not native.get("has_take_profit"):
        missing.append("TP")
    if missing:
        return "Faltando " + "/".join(missing)
    return "Parcial"


def _model_vote_summary(payload: dict) -> str:
    decision = payload.get("decision") or {}
    model_votes = decision.get("model_votes") or payload.get("model_votes") or {}
    hold_threshold = v_num(decision.get("action_hold_threshold"), 0.6)
    if not isinstance(model_votes, dict) or not model_votes:
        return "--"

    fragments = []
    for model_name, raw_action in model_votes.items():
        vote_value = v_num(raw_action, 0.0)
        if vote_value >= hold_threshold:
            emoji = "??"
        elif vote_value <= -hold_threshold:
            emoji = "??"
        else:
            emoji = "??"
        short_name = str(model_name).split("_s")[-1] if "_s" in str(model_name) else str(model_name)
        fragments.append(f"S{short_name} {vote_value:+.3f} {emoji}")
    return " | ".join(fragments)


def _position_details(payload: dict, exchange_status: dict | None) -> dict:
    exchange = exchange_status or {}
    position_label = str(
        exchange.get("position_label")
        or payload.get("position_label")
        or "FLAT"
    ).upper()
    position_open = bool(exchange.get("position_is_open", payload.get("position_is_open", False)))
    return {
        "label": position_label,
        "open": position_open,
        "size": _num(exchange.get("position_size", payload.get("position_size")), None),
        "entry": _num(exchange.get("position_avg_entry_price", payload.get("position_avg_entry_price")), None),
        "pnl": _num(exchange.get("position_unrealized_pnl", payload.get("position_unrealized_pnl")), None),
        "notional": _num(exchange.get("position_notional_value", payload.get("notional_value")), None),
        "tp": _num(exchange.get("take_profit_price", payload.get("take_profit_price")), None),
        "sl": _num(exchange.get("stop_loss_price", payload.get("stop_loss_price")), None),
        "time_in_bars": _num(payload.get("position_time_in_bars"), None),
    }


def _build_alert_lines(runtime: dict, exchange_status: dict | None, exchange_error: str | None) -> list[str]:
    alerts: list[str] = []
    payload = runtime.get("payload") or {}
    blocked_reason = str(
        payload.get("blocked_reason_human")
        or payload.get("blocked_reason")
        or ""
    ).strip()
    if blocked_reason:
        alerts.append(f"?? <b>Bloqueio:</b> {html.escape(_human_block_label(blocked_reason))}")

    manual_protection_message = str(payload.get("manual_protection_message") or "").strip()
    if payload.get("manual_protection_required"):
        alerts.append(
             f"??? <b>Proteção manual:</b> {html.escape(manual_protection_message or 'TP/SL precisa de confirmação manual.')}"
        )

    if runtime.get("last_cycle_error"):
        error_at = _format_local_dt(runtime.get("last_cycle_error_at"), default="--")
        alerts.append(
            f"? <b>Último erro:</b> {html.escape(_clip(runtime.get('last_cycle_error')))}"
            f" ({html.escape(error_at)})"
        )

    guard_result = runtime.get("guard_result") or {}
    if guard_result:
        alerts.append(
            f"?? <b>Guard:</b> {html.escape(str(guard_result.get('message') or guard_result.get('trigger') or 'atuação registrada'))}"
        )

    if isinstance(exchange_status, dict) and exchange_status.get("position_is_open") and not exchange_status.get("native_tp_sl_protected"):
        alerts.append("?? <b>TP/SL nativo:</b> a posição aberta ainda não está totalmente protegida.")

    if exchange_error:
        alerts.append(f"?? <b>Exchange:</b> {html.escape(_clip(exchange_error, limit=140))}")

    return alerts


def _execution_action_label(payload: dict) -> str:
    raw = str(
        payload.get("execution_action")
        or (payload.get("execution") or {}).get("action_taken")
        or ""
    ).strip().lower()
    mapping = {
        "opened_position": "Abriu posição",
        "closed_position": "Fechou posição",
        "blocked_entry": "Entrada bloqueada",
        "no_entry": "Sem entrada",
        "error": "Ciclo com erro",
        "manual_protection_required": "Proteção manual exigida",
    }
    if not raw:
        return "--"
    return mapping.get(raw, raw.replace("_", " "))


def _current_reason_label(runtime: dict) -> str:
    payload = runtime.get("payload") or {}
    blocked = str(
        payload.get("blocked_reason_human")
        or payload.get("blocked_reason")
        or ""
    ).strip()
    if blocked:
        return _human_block_label(blocked)

    manual_protection = str(payload.get("manual_protection_message") or "").strip()
    if payload.get("manual_protection_required"):
        return manual_protection or "Proteção manual de TP/SL necessária."

    payload_error = str(payload.get("error") or "").strip()
    if payload_error:
        return payload_error

    decision_reason = str(
        (payload.get("decision") or {}).get("reason")
        or payload.get("decision_reason")
        or ""
    ).strip()
    if decision_reason:
        return decision_reason

    last_cycle_error = str(runtime.get("last_cycle_error") or "").strip()
    if last_cycle_error:
        return last_cycle_error

    return str(runtime.get("situation_message") or "--")


def _log_file_candidates(filter_text: str | None) -> list[Path]:
    normalized = str(filter_text or "").strip().lower()
    preferred_names = ["traderbot-rl.log", "traderbot-launcher.log", "traderbot-launcher-fatal.log"]
    if "launcher" in normalized:
        preferred_names = ["traderbot-launcher.log", "traderbot-launcher-fatal.log", "traderbot-rl.log"]
    files = [LOGS_DIR / name for name in preferred_names]
    return [path for path in files if path.exists()]


def _log_matchers(filter_text: str | None) -> list[str]:
    normalized = str(filter_text or "").strip().lower()
    if not normalized:
        return []

    presets = {
        "erro": [" erro ", "error", "exception", "traceback", "fatal", "runtime.cycle_error"],
        "error": [" erro ", "error", "exception", "traceback", "fatal", "runtime.cycle_error"],
        "trade": ["execution.trade_open", "execution.trade_close", "opened_position", "closed_position", "manual_close"],
        "guard": ["guard", "closed_by_guard", "guard_close", "guard_checked"],
        "status": ["status hyperliquid", "healthcheck", "heartbeat", "executor conectado"],
        "ciclo": ["ciclo runtime hl", "runtime_cycle", "runtime.no_entry", "blocked.", "execution.trade", "runtime.cycle_error"],
        "launcher": ["launcher", "health_status_change", "bot iniciando", "bot parado"],
    }

    matchers: list[str] = []
    for token in normalized.split():
        token_matchers = presets.get(token)
        if token_matchers:
            matchers.extend(token_matchers)
        else:
            matchers.append(token)
    return [matcher.lower() for matcher in matchers if matcher]


def _line_matches_filter(line: str, matchers: list[str]) -> bool:
    if not matchers:
        return True
    line_lower = line.lower()
    return any(matcher in line_lower for matcher in matchers)


def _tail_matching_lines(path: Path, limit: int, matchers: list[str]) -> list[str]:
    collected: deque[str] = deque(maxlen=max(limit, 1))
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if _line_matches_filter(line, matchers):
                    collected.append(line)
    except OSError:
        return []
    return list(collected)


def ler_logs_recentes(filter_text: str | None = None, limit: int = 8) -> list[str]:
    limit = max(1, min(int(limit), 20))
    matchers = _log_matchers(filter_text)
    files = _log_file_candidates(filter_text)
    collected: list[tuple[str, str]] = []

    for path in files:
        file_limit = max(limit * 3, 20)
        lines = _tail_matching_lines(path, file_limit, matchers)
        short_name = path.name.replace("traderbot-", "").replace(".log", "")
        for line in lines:
            collected.append((line, short_name))

    if not collected:
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for line, short_name in collected:
        formatted = f"[{short_name}] {line}"
        if formatted in seen:
            continue
        seen.add(formatted)
        deduped.append(formatted)

    deduped.sort(key=lambda item: item[5:24] if len(item) >= 24 else item)
    return deduped[-limit:]


def _build_logs_message(filter_text: str | None, lines: list[str], limit: int) -> str:
    normalized_filter = str(filter_text or "").strip()
    title = "?? <b>LOGS RECENTES</b>"
    subtitle = f"Filtro: <b>{html.escape(normalized_filter or 'geral')}</b> | Itens: <b>{len(lines)}</b>"
    help_line = "Use: <code>/logs erro</code>, <code>/logs trade</code>, <code>/logs guard</code>, <code>/logs ciclo</code>"

    if not lines:
        return "\n".join(
            [
                title,
                subtitle,
                help_line,
                "",
                "Nenhuma linha encontrada para esse filtro.",
            ]
        )

    rendered_lines = [html.escape(_clip(line, limit=240)) for line in lines]
    while rendered_lines and len("\n".join(rendered_lines)) > 3200:
        rendered_lines.pop(0)

    return "\n".join(
        [
            title,
            subtitle,
            help_line,
            "",
            "<pre>" + "\n".join(rendered_lines[-limit:]) + "</pre>",
        ]
    )


def _build_dash_message(context: dict) -> str:
    cfg = context["config"]
    runtime = context["runtime"]
    exchange_status = context["exchange_status"] or {}
    stats = context["stats"]
    payload = runtime.get("payload") or {}

    account = exchange_status.get("account_summary") or {}
    decision = payload.get("decision") or {}
    position = _position_details(payload, exchange_status)

    equity = account.get("equity")
    available = account.get("available_to_trade", exchange_status.get("available_to_trade"))
    drawdown = account.get("drawdown_pct")
    price = exchange_status.get("latest_mid", payload.get("reference_price"))
    confidence = _num(decision.get("confidence_pct", payload.get("confidence_pct")), 0.0) or 0.0

    lines = [
        "?? <b>DASHBOARD OPERACIONAL</b>",
        "????????????????????",
        f"{runtime['runtime_emoji']} <b>Runtime:</b> {html.escape(runtime['runtime_label'])}",
        f"?? <b>Situação:</b> {html.escape(runtime['situation_label'])}",
        f"?? <b>Resumo:</b> {html.escape(runtime['situation_message'])}",
        f"?? <b>Exchange:</b> {html.escape(_exchange_health_label(exchange_status))}",
        f"?? <b>Modo:</b> {html.escape(TARGET_NETWORK.upper())} | "
        f"{html.escape(str(cfg.get('hyperliquid', {}).get('symbol', 'BTC')))} | "
        f"{html.escape(str(cfg.get('hyperliquid', {}).get('timeframe', '1h')))}",
        f"?? <b>Heartbeat:</b> {html.escape(_format_local_dt(runtime.get('heartbeat_at')))} "
        f"({html.escape(_format_age(runtime.get('heartbeat_age_seconds')))})",
        "",
        f"?? <b>Patrimônio:</b> {html.escape(_format_usd(equity))}",
        f"?? <b>Disponível:</b> {html.escape(_format_usd(available))}",
        f"?? <b>Drawdown:</b> {html.escape(_format_ratio_pct(drawdown))}",
        f"?? <b>PnL em aberto:</b> {html.escape(_format_usd(position['pnl']))}",
        "",
        f"?? <b>HOJE ({datetime.now().strftime('%d/%m')})</b>",
        f"? <b>Operações do dia:</b> {stats['operacoes']}",
        f"?? <b>Encerradas:</b> {stats['fechadas']}",
        f"?? <b>Bloqueadas:</b> {stats['bloqueadas']}",
        f"?? <b>Winrate:</b> {stats['winrate']:.1f}%",
        f"?? <b>PnL fechado:</b> {html.escape(_format_usd(stats['pnl_fechado']))}",
        "",
        f"?? <b>Mercado:</b> {html.escape(_market_status_label(payload))} | BTC {html.escape(_format_price(price))}",
        f"?? <b>IA:</b> {html.escape(_human_signal(decision.get('vote_bucket') or payload.get('vote_bucket')))} "
        f"({confidence:.1f}%)",
        f"?? <b>Posição:</b> {html.escape(position['label'])}",
    ]

    alerts = _build_alert_lines(runtime, exchange_status, context["exchange_error"])
    if alerts:
        lines.append("")
        lines.append("?? <b>ALERTAS</b>")
        lines.extend(alerts[:3])

    return "\n".join(lines)


def _build_status_message(context: dict) -> str:
    cfg = context["config"]
    runtime = context["runtime"]
    exchange_status = context["exchange_status"] or {}
    exchange_error = context["exchange_error"]
    stats = context["stats"]
    payload = runtime.get("payload") or {}
    decision = payload.get("decision") or {}
    filters = payload.get("filters") or {}
    features = payload.get("feature_snapshot") or {}
    votes = decision.get("votes") or payload.get("votes") or {}
    position = _position_details(payload, exchange_status)
    validation = exchange_status.get("validation_sizing") or {}
    account = exchange_status.get("account_summary") or {}
    protection = _protection_label(exchange_status)

    buy_votes = int(votes.get("buy", 0) or 0) if isinstance(votes, dict) else 0
    hold_votes = int(votes.get("hold", 0) or 0) if isinstance(votes, dict) else 0
    sell_votes = int(votes.get("sell", 0) or 0) if isinstance(votes, dict) else 0
   
    confidence = _num(decision.get("confidence_pct", payload.get("confidence_pct")), 0.0) or 0.0
    risk_pct = _num(validation.get("risk_pct"), None)
    risk_amount = _num(validation.get("risk_amount"), None)
    target_notional = _num(validation.get("adjusted_notional", validation.get("notional_value")), None)
    current_reason = _current_reason_label(runtime)
    execution_action = _execution_action_label(payload)

    lines = [
        "??? <b>STATUS COMPLETO DO BOT</b>",
        "????????????????????",
        f"{runtime['runtime_emoji']} <b>Runtime:</b> {html.escape(runtime['runtime_label'])}",
        f"?? <b>Situação:</b> {html.escape(runtime['situation_label'])}",
        f"?? <b>Resumo:</b> {html.escape(runtime['situation_message'])}",
        f"?? <b>Exchange:</b> {html.escape(_exchange_health_label(exchange_status))}",
        f"?? <b>Modo:</b> {html.escape(TARGET_NETWORK.upper())} | "
        f"{html.escape(str(cfg.get('hyperliquid', {}).get('symbol', 'BTC')))} | "
        f"{html.escape(str(cfg.get('hyperliquid', {}).get('timeframe', '1h')))} | "
        f"{html.escape(TARGET_EXECUTION_MODE)}",
        f"?? <b>Heartbeat:</b> {html.escape(_format_local_dt(runtime.get('heartbeat_at')))} "
        f"({html.escape(_format_age(runtime.get('heartbeat_age_seconds')))})",
        f"??? <b>Última vela:</b> {html.escape(_format_local_dt(runtime.get('bar_timestamp')))}",
        f"??? <b>Último evento:</b> {html.escape(runtime.get('event_code', '--'))}",
        f"?? <b>Motivo atual:</b> {html.escape(_clip(current_reason, 180))}",
        f"?? <b>Ação da engine:</b> {html.escape(execution_action)}",
        "",
        "?? <b>POSIÇÃO E RISCO</b>",
        f"• <b>Posição:</b> {html.escape(position['label'])}",
        f"• <b>PnL aberto:</b> {html.escape(_format_usd(position['pnl']))}",
        f"• <b>Entrada:</b> {html.escape(_format_price(position['entry']))}",
        f"• <b>TP / SL:</b> {html.escape(_format_price(position['tp']))} / {html.escape(_format_price(position['sl']))}",
        f"• <b>Volume:</b> {html.escape(_format_plain_number(position['size'], digits=4))}",
        f"• <b>Notional:</b> {html.escape(_format_usd(position['notional']))}",
        f"• <b>Tempo em posição:</b> {html.escape(_format_plain_number(position['time_in_bars'], digits=0))} candles",
        f"• <b>Patrimônio:</b> {html.escape(_format_usd(account.get('equity')))}",
        f"• <b>Disponível:</b> {html.escape(_format_usd(account.get('available_to_trade', exchange_status.get('available_to_trade'))))}",
        f"• <b>Drawdown:</b> {html.escape(_format_ratio_pct(account.get('drawdown_pct')))}",
        f"• <b>Risco por trade:</b> "
        f"{html.escape(_format_pct(risk_pct * 100.0 if risk_pct is not None else None))} "
        f"({html.escape(_format_usd(risk_amount))})",
        f"• <b>Notional alvo:</b> {html.escape(_format_usd(target_notional))}",
        f"• <b>Proteção nativa:</b> {html.escape(protection)}",
        "",
        "?? <b>MERCADO E IA</b>",
        f"• <b>Mercado:</b> {html.escape(_market_status_label(payload))}",
        f"• <b>Preço BTC:</b> {html.escape(_format_price(exchange_status.get('latest_mid', payload.get('reference_price'))))}",
        f"• <b>Decisão:</b> {html.escape(_human_signal(decision.get('vote_bucket') or payload.get('vote_bucket')))} "
        f"({confidence:.1f}%)",
        f"• <b>Placar:</b> ?? {buy_votes} | ?? {hold_votes} | ?? {sell_votes}",
        f"• <b>RSI / Vol Z:</b> "
        f"{html.escape(_format_plain_number(features.get('rsi_14'), digits=1))} / "
        f"{html.escape(_format_plain_number(features.get('volume_zscore_20'), digits=2))}",
        f"• <b>Dist. EMA240:</b> {html.escape(_format_plain_number(filters.get('regime_dist_ema_240'), digits=5))}",
        f"• <b>Seeds:</b> {html.escape(_model_vote_summary(payload))}",
        "",
        f"?? <b>HOJE ({datetime.now().strftime('%d/%m')})</b>",
        f"• <b>Operações:</b> {stats['operacoes']}",
        f"• <b>Encerradas:</b> {stats['fechadas']} | <b>Bloqueadas:</b> {stats['bloqueadas']}",
        f"• <b>Winrate:</b> {stats['winrate']:.1f}% ({stats['vitorias']}V/{stats['derrotas']}D)",
        f"• <b>PnL do dia:</b> {html.escape(_format_usd(stats['pnl_fechado']))}",
        f"• <b>Última ação/operação:</b> {html.escape(str(stats.get('ultimo_evento', '--')))} "
        f"às {html.escape(str(stats.get('ultima_hora', '--')))}",
    ]

    alerts = _build_alert_lines(runtime, exchange_status, exchange_error)
    if alerts:
        lines.append("")
        lines.append("?? <b>ALERTAS E DIAGNÓSTICO</b>")
        lines.extend(alerts)

    return "\n".join(lines)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    global trade_process

    runtime = _build_runtime_snapshot(ler_status_runtime(TARGET_NETWORK))
    if trade_process is not None and trade_process.poll() is None:
        await update.message.reply_text("O bot já está rodando nesta sessão do painel.")
        return

    if runtime["alive"]:
        await update.message.reply_text(
            f"O bot já está ativo em background.\n\n"
            f"Runtime: {runtime['runtime_label']}\n"
            f"Situação: {runtime['situation_label']} - {runtime['situation_message']}"
        )
        return

    await update.message.reply_text("Iniciando o TraderBot na MAINNET com live trading ativado...")

    cmd = [
        sys.executable,
        "-m",
        "traderbot.main",
        "--config",
        str(CONFIG_PATH),
        "--network-override",
        TARGET_NETWORK,
        "--execution-mode-override",
        TARGET_EXECUTION_MODE,
        "--allow-live-trading",
        "run",
    ]

    trade_process = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    await update.message.reply_text("Bot iniciado com sucesso em background!")


async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    global trade_process

    if trade_process is not None and trade_process.poll() is None:
        await update.message.reply_text("Desligando o bot... (fechando processo desta sessão)")
        trade_process.terminate()
        trade_process.wait()
        trade_process = None
        await update.message.reply_text("Bot encerrado.")
        return

    runtime = _build_runtime_snapshot(ler_status_runtime(TARGET_NETWORK))
    if runtime["alive"]:
        await update.message.reply_text(
            "Existe um runtime ativo em background, mas ele não foi iniciado por esta sessão do painel.\n\n"
            "Para evitar matar o processo errado, o /off desta versão encerra apenas o processo local desta sessão.\n"
            "Use /status para acompanhar o heartbeat e, se quiser, eu ajusto o /off depois para encerrar o PID do runtime."
        )
        return

    await update.message.reply_text("O bot já está desligado.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    loading = await update.message.reply_text("? <b>Consultando runtime, exchange e diagnósticos...</b>", parse_mode="HTML")
    painel = await coletar_contexto_painel(TARGET_NETWORK)
    mensagem = _build_status_message(painel)
    await loading.edit_text(mensagem, parse_mode="HTML")


async def cmd_dash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    loading = await update.message.reply_text("? <b>Montando dashboard operacional...</b>", parse_mode="HTML")
    painel = await coletar_contexto_painel(TARGET_NETWORK)
    mensagem = _build_dash_message(painel)
    await loading.edit_text(mensagem, parse_mode="HTML")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    msg = "\n".join(
        [
            "?? <b>COMANDOS DISPONÍVEIS</b>",
            "????????????????????",
            "• <code>/status</code> painel completo com runtime, posição, motivo, IA, risco e PnL do dia",
            "• <code>/dash</code> resumo rápido operacional e financeiro",
            "• <code>/logs</code> últimas linhas relevantes",
            "• <code>/logs erro</code> apenas erros recentes",
            "• <code>/logs trade</code> aberturas e fechamentos",
            "• <code>/logs guard</code> atuações do runtime guard",
            "• <code>/logs ciclo 12</code> últimos 12 eventos de ciclo",
            "• <code>/config</code> ver ou alterar parâmetros rápidos",
            "• <code>/start</code> iniciar o runtime",
            "• <code>/off</code> parar o processo iniciado por esta sessão",
        ]
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    raw_args = [str(arg).strip() for arg in context.args if str(arg).strip()]
    limit = 8
    filter_parts: list[str] = []
    for arg in raw_args:
        if arg.isdigit():
             limit = max(1, min(int(arg), 20))
        else:
            filter_parts.append(arg)

    filter_text = " ".join(filter_parts).strip() or None
    loading = await update.message.reply_text("? <b>Consultando logs recentes...</b>", parse_mode="HTML")
    lines = await asyncio.to_thread(ler_logs_recentes, filter_text, limit)
    mensagem = _build_logs_message(filter_text, lines, limit)
    await loading.edit_text(mensagem, parse_mode="HTML")


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not verificar_autorizacao(update):
        return

    args = context.args
    if len(args) == 0:
        cfg = carregar_config().get("environment", {})
        msg = "*Configurações Atuais:*\n\n"
        msg += f"1?? Risco Máx: {cfg.get('max_risk_per_trade', 0.0) * 100:.1f}%\n"
        msg += f"2?? Stop Loss: {cfg.get('stop_loss_pct', 0.0) * 100:.1f}%\n"
        msg += f"3?? Take Profit: {cfg.get('take_profit_pct', 0.0) * 100:.1f}%\n"
        msg += f"4?? Filtro Hold: {cfg.get('action_hold_threshold', 0.0)}\n"
        msg += f"5?? Cooldown: {cfg.get('cooldown', 0)} candles\n\n"
        msg += "Para alterar, use:\n`/config [risco|stop|tp|hold|cooldown] [valor]`\nExemplo: `/config cooldown 3`"
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if len(args) < 2:
        await update.message.reply_text("Formato incorreto. Use: `/config [parâmetro] [valor]`", parse_mode="Markdown")
        return

    parametro, valor_str = args[0].lower(), args[1]

    try:
        valor = float(valor_str)
    except ValueError:
        await update.message.reply_text("O valor deve ser um número válido.")
        return

    cfg_data = carregar_config()

    if parametro == "risco":
        cfg_data["environment"]["max_risk_per_trade"] = valor / 100.0
    elif parametro == "stop":
        cfg_data["environment"]["stop_loss_pct"] = valor / 100.0
    elif parametro == "tp":
        cfg_data["environment"]["take_profit_pct"] = valor / 100.0
    elif parametro == "hold":
        cfg_data["environment"]["action_hold_threshold"] = valor
    elif parametro == "cooldown":
        cfg_data["environment"]["cooldown"] = int(valor)
    else:
        await update.message.reply_text("Parâmetro desconhecido. Escolha: risco, stop, tp, hold ou cooldown.")
        return

    salvar_config(cfg_data)
    await update.message.reply_text(
        f"*{parametro.upper()}* alterado para {valor} com sucesso!\n"
        "_Obs: O bot pegará a nova configuração no próximo ciclo de mercado._",
        parse_mode="Markdown",
    )


async def relatorio_diario(context: ContextTypes.DEFAULT_TYPE):
    msg = f"*RELATÓRIO DIÁRIO - {date.today().strftime('%d/%m/%Y')}*\n\n"
    msg += "Fechamento do dia. O bot finalizou suas operações diárias.\n"
    await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")


async def relatorio_mensal(context: ContextTypes.DEFAULT_TYPE):
    if datetime.now().day != 1:
        return
    msg = "?? *FECHAMENTO MENSAL*\n\nResumo do mês que passou...\n"
    await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")


def main():
    if TOKEN == "SEU_TOKEN_AQUI" or CHAT_ID == "SEU_CHAT_ID_AQUI" or not TOKEN:
        print("ALERTA: Configure o TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no seu arquivo .env!")
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("off", cmd_off))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("dash", cmd_dash))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("config", cmd_config))

    app.job_queue.run_daily(relatorio_diario, time(hour=23, minute=55))
    app.job_queue.run_daily(relatorio_mensal, time(hour=9, minute=0))

    print("Painel do Telegram iniciado com sucesso! Aguardando comandos...")
    app.run_polling()


if __name__ == "__main__":
    main()