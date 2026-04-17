from __future__ import annotations

import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import ctypes
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QObject, QProcess, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from traderbot.config import AppConfig, EnvironmentConfig, load_config
from traderbot.launcher_ai import OpenAILogTranslator
from traderbot.launcher_services import HumanizedEvent, LauncherEvent


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LAUNCHER_ICON_PATH = REPO_ROOT / "icon" / "icon.ico"
WINDOWS_APP_ID = "traderbot.launcher.desktop"

HUMANIZED_MARKET_PLACEHOLDER = "Aguardando leitura humanizada do mercado."
HUMANIZED_MODEL_PLACEHOLDER = "Aguardando interpretacao humanizada do modelo."
HUMANIZED_FILTERS_PLACEHOLDER = "Aguardando diagnostico humanizado dos filtros."
HUMANIZED_EXECUTION_PLACEHOLDER = "Aguardando resumo humanizado da execucao."


def _currency(value: Any, prefix: str = "$") -> str:
    try:
        return f"{prefix}{float(value):,.2f}"
    except (TypeError, ValueError):
        return f"{prefix}0.00"


def _price(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def _strength(value: Any) -> str:
    try:
        return f"{abs(float(value)) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def _display_value(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text == "--":
        return fallback
    return text


def _has_env_value(name: str) -> bool:
    if not name:
        return False
    value = os.getenv(name)
    if value is None:
        value = os.getenv(f"\ufeff{name}")
    if value is None:
        return False
    return bool(value.strip().strip("\"'"))


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_number(
    value: Any,
    digits: int = 2,
    *,
    signed: bool = False,
    suffix: str = "",
    fallback: str = "--",
) -> str:
    number = _safe_float(value)
    if number is None:
        return fallback
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:,.{digits}f}{suffix}"


def _optional_price(value: Any, fallback: str = "--") -> str:
    number = _safe_float(value)
    if number is None:
        return fallback
    return f"${number:,.2f}"


def _optional_pct(value: Any, digits: int = 2, *, signed: bool = True, fallback: str = "--") -> str:
    number = _safe_float(value)
    if number is None:
        return fallback
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number * 100.0:.{digits}f}%"


def _optional_ratio(value: Any, digits: int = 2, fallback: str = "--") -> str:
    number = _safe_float(value)
    if number is None:
        return fallback
    return f"{number:.{digits}f}x"


def _json_marker(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _apply_windows_app_identity() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass


@dataclass(frozen=True)
class LauncherMode:
    key: str
    label: str
    network: str
    execution_mode: str
    allow_live_trading: bool


MODES: dict[str, LauncherMode] = {
    "testnet": LauncherMode(
        key="testnet",
        label="Testnet",
        network="testnet",
        execution_mode="exchange",
        allow_live_trading=True,
    ),
    "mainnet": LauncherMode(
        key="mainnet",
        label="Mainnet",
        network="mainnet",
        execution_mode="exchange",
        allow_live_trading=True,
    ),
}


@dataclass
class DashboardState:
    online: bool = False
    can_trade: bool = False
    state_label: str = "AGUARDANDO"
    state_message: str = "Aguardando dados do bot"
    state_style: str = "wait"
    regime_label: str = "indefinido"
    signal_label: str = "nenhum"
    direction_label: str = "nenhuma"
    strength_label: str = "--"
    reference_price: str = "--"
    risk_label: str = "--"
    balance_label: str = "--"
    blocker_label: str = "Sem bloqueio"
    last_decision: str = "Sem decisão no momento."
    last_update_label: str = "--"
    last_valid_check_label: str = "--"
    network_label: str = "TESTNET"
    symbol_label: str = "BTC"
    timeframe_label: str = "1h"
    connection_label: str = "offline"
    execution_mode_label: str = "exchange"
    operations_today_label: str = "0"
    blocked_today_label: str = "0"
    last_trade_reason_label: str = "Nenhuma operação hoje."
    last_skip_reason_label: str = "Nenhum bloqueio recente."
    context_summary_label: str = HUMANIZED_MARKET_PLACEHOLDER
    feature_summary_label: str = HUMANIZED_MODEL_PLACEHOLDER
    humanized_market_state: str = ""
    humanized_model_interpretation: str = ""
    humanized_filters_diagnostic: str = ""
    humanized_execution_summary: str = ""
    humanized_simple_summary: str = ""
    last_cycle_fingerprint: str = ""
    position_label: str = "FLAT"
    position_status: str = "Sem sinal ativo"
    position_size_label: str = "--"
    entry_label: str = "--"
    take_profit_label: str = "--"
    stop_loss_label: str = "--"
    pnl_label: str = "--"
    pnl_style: str = "muted"
    last_raw_detail: str = ""
    market_snapshot: dict[str, Any] = field(default_factory=dict)
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    decision_snapshot: dict[str, Any] = field(default_factory=dict)
    filter_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationalHealthState:
    bot_running: bool = False
    last_healthcheck_at: datetime | None = None
    last_successful_healthcheck_at: datetime | None = None
    connection_ok: bool = False
    executor_alive: bool = False
    last_check_execution_ok: bool | None = None
    status: str = "offline"
    reason: str = "startup"


def _refresh_widget_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _set_widget_property(widget: QWidget, name: str, value: Any) -> None:
    widget.setProperty(name, value)
    _refresh_widget_style(widget)


def _configure_dynamic_label(label: QLabel) -> None:
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    _fit_label_height(label)


def _fit_label_height(label: QLabel) -> None:
    label.updateGeometry()
    hint_height = label.sizeHint().height()
    if hint_height > 0:
        label.setMinimumHeight(hint_height)


class TranslationSignals(QObject):
    finished = Signal(object)


class TranslationTask(QRunnable):
    def __init__(self, translator: OpenAILogTranslator, event: LauncherEvent, fallback: HumanizedEvent):
        super().__init__()
        self.translator = translator
        self.event = event
        self.fallback = fallback
        self.signals = TranslationSignals()

    def run(self) -> None:
        result = self.translator.translate(self.event, self.fallback)
        try:
            self.signals.finished.emit(result)
        except RuntimeError:
            # A janela pode ter sido fechada e destruido o QObject dono do sinal.
            pass


class MetricTile(QFrame):
    def __init__(
        self,
        label: str,
        *,
        compact: bool = False,
        wrap_value: bool = False,
        min_height: int = 136,
    ):
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setMinimumHeight(min_height)
        self._compact = compact
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        self.label = QLabel(label)
        self.label.setObjectName("metaLabel")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.label)

        self.value = QLabel("--")
        self.value.setObjectName("metricValue")
        self.value.setProperty("tone", "default")
        if compact:
            self.value.setProperty("sizeVariant", "compact")
        elif wrap_value:
            self.value.setProperty("sizeVariant", "body")
        else:
            self.value.setProperty("sizeVariant", "default")
        self.value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.value.setWordWrap(True)
        layout.addWidget(self.value)

        self.note = QLabel("")
        self.note.setObjectName("subtleText")
        _configure_dynamic_label(self.note)
        self.note.hide()
        layout.addWidget(self.note)

    def update_tile(self, value: str, *, note: str | None = None, tone: str = "default") -> None:
        self.value.setText(value)
        _fit_label_height(self.value)
        _set_widget_property(self.value, "tone", tone)
        if note:
            self.note.setText(note)
            _fit_label_height(self.note)
            self.note.show()
        else:
            self.note.hide()


class DetailFieldCard(QFrame):
    def __init__(self, title: str, *, min_height: int = 224):
        super().__init__()
        self.setObjectName("PositionCard")
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("fieldLabel")
        layout.addWidget(title_label)

        self.summary = QLabel("--")
        self.summary.setObjectName("metricValue")
        self.summary.setProperty("sizeVariant", "body")
        _configure_dynamic_label(self.summary)
        layout.addWidget(self.summary)

        self.note = QLabel("")
        self.note.setObjectName("HeroHint")
        _configure_dynamic_label(self.note)
        self.note.hide()
        layout.addWidget(self.note)

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)
        _fit_label_height(self.summary)

    def set_note(self, text: str | None) -> None:
        if text:
            self.note.setText(text)
            _fit_label_height(self.note)
            self.note.show()
        else:
            self.note.hide()


class EventDetailsDialog(QDialog):
    def __init__(self, summary: HumanizedEvent, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.setModal(True)
        self.setWindowTitle("Detalhes do evento")
        self.resize(760, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("dialogSurface")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel(self.summary.message_human)
        title.setObjectName("dialogTitle")
        _configure_dynamic_label(title)
        root.addWidget(title)

        meta = QLabel(f"{self.summary.severity.upper()}  {self.summary.timestamp.strftime('%H:%M:%S')}")
        meta.setObjectName("subtleText")
        root.addWidget(meta)

        if self.summary.details:
            details_header = QLabel("Detalhes")
            details_header.setObjectName("fieldLabel")
            root.addWidget(details_header)

            details = QLabel(self.summary.details)
            details.setObjectName("subtleText")
            _configure_dynamic_label(details)
            root.addWidget(details)

        raw_header = QLabel("Log tecnico")
        raw_header.setObjectName("fieldLabel")
        root.addWidget(raw_header)

        raw_box = QPlainTextEdit()
        raw_box.setObjectName("technicalLog")
        raw_box.setReadOnly(True)
        raw_box.setPlainText(self.summary.message_raw or "Sem log tecnico disponivel.")
        root.addWidget(raw_box, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_button = QPushButton("Fechar")
        close_button.setObjectName("primary")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        root.addLayout(footer)


class EventToast(QFrame):
    def __init__(self, summary: HumanizedEvent, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("EventToast")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title = QLabel(summary.message_human)
        title.setObjectName("toastTitle")
        _configure_dynamic_label(title)
        layout.addWidget(title)

        meta = QLabel(summary.timestamp.strftime("%H:%M:%S"))
        meta.setObjectName("subtleText")
        layout.addWidget(meta)

        _set_widget_property(self, "severity", summary.severity)
        self.adjustSize()
        QTimer.singleShot(2400, self.close)


class NotificationItemWidget(QFrame):
    def __init__(self, summary: HumanizedEvent):
        super().__init__()
        self.summary = summary
        self.count = 1
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("event")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._build_ui()
        self.apply_summary(summary)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.dot_label = QLabel("●")
        self.dot_label.setObjectName("severityDot")
        top.addWidget(self.dot_label, 0, Qt.AlignLeft)

        self.severity_badge = QLabel("INFO")
        self.severity_badge.setObjectName("badge")
        top.addWidget(self.severity_badge, 0, Qt.AlignLeft)

        self.time_label = QLabel("--:--")
        self.time_label.setObjectName("metaLabel")
        top.addWidget(self.time_label)

        top.addStretch(1)

        self.count_label = QLabel("")
        self.count_label.setObjectName("badge")
        self.count_label.hide()
        top.addWidget(self.count_label, 0, Qt.AlignRight)

        layout.addLayout(top)

        self.message_label = QLabel("")
        self.message_label.setObjectName("notificationMessage")
        _configure_dynamic_label(self.message_label)
        self.message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.message_label)
        self.setToolTip("Ver log tecnico")

    def apply_summary(self, summary: HumanizedEvent) -> None:
        self.summary = summary
        self.dot_label.setText("•")
        self.severity_badge.setText(summary.severity.upper())
        self.message_label.setText(summary.message_human)
        _fit_label_height(self.message_label)
        widget_hint_height = self.sizeHint().height()
        if widget_hint_height > 0:
            self.setMinimumHeight(widget_hint_height)
        self.message_label.updateGeometry()
        self.updateGeometry()
        self.time_label.setText(summary.timestamp.strftime("%H:%M:%S"))
        _set_widget_property(self, "severity", summary.severity)
        _set_widget_property(self.severity_badge, "severity", summary.severity)
        _set_widget_property(self.dot_label, "severity", summary.severity)
        self._refresh_count()

    def bump(self, summary: HumanizedEvent) -> None:
        self.count += 1
        self.apply_summary(summary)

    def _refresh_count(self) -> None:
        if self.count <= 1:
            self.count_label.hide()
            return
        self.count_label.setText(f"x{self.count}")
        self.count_label.show()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        super().mousePressEvent(event)
        dialog = EventDetailsDialog(self.summary, parent=self.window())
        dialog.exec()


class SettingsDialog(QDialog):
    def __init__(
        self,
        cfg: AppConfig,
        smoke_side: str,
        smoke_wait_seconds: float,
        parent=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        env_defaults = EnvironmentConfig()
        self._dialog_defaults = {
            "risk_pct": float(self.cfg.environment.max_risk_per_trade) * 100.0,
            "hold_threshold": float(env_defaults.action_hold_threshold),
            "regime_min_abs_dist_ema_240": float(env_defaults.regime_min_abs_dist_ema_240),
            "regime_min_vol_regime_z": float(env_defaults.regime_min_vol_regime_z),
            "auto_check_interval": float(self.cfg.launcher.auto_check_interval_seconds),
        }
        self.setModal(True)
        self.setWindowTitle("Configuracoes")
        self.resize(620, 520)
        self.setMinimumSize(560, 480)
        self._build_ui(smoke_side=smoke_side, smoke_wait_seconds=smoke_wait_seconds)

    def _build_ui(self, smoke_side: str, smoke_wait_seconds: float) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel("Configuracoes rapidas")
        self.setObjectName("dialogSurface")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        hint = QLabel("Ajustes operacionais do launcher. Chaves sensiveis continuam fora da interface.")
        hint.setObjectName("subtleText")
        _configure_dynamic_label(hint)
        root.addWidget(hint)

        self.risk_input = QDoubleSpinBox()
        self.risk_input.setRange(0.05, 10.0)
        self.risk_input.setDecimals(2)
        self.risk_input.setSingleStep(0.05)
        self.risk_input.setSuffix(" %")
        self.risk_input.setValue(float(self.cfg.environment.max_risk_per_trade) * 100.0)
        self.risk_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.hold_threshold_input = QDoubleSpinBox()
        self.hold_threshold_input.setRange(0.0, 1.0)
        self.hold_threshold_input.setDecimals(2)
        self.hold_threshold_input.setSingleStep(0.01)
        self.hold_threshold_input.setValue(float(self.cfg.environment.action_hold_threshold))
        self.hold_threshold_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.regime_abs_dist_input = QDoubleSpinBox()
        self.regime_abs_dist_input.setRange(0.0, 1.0)
        self.regime_abs_dist_input.setDecimals(3)
        self.regime_abs_dist_input.setSingleStep(0.005)
        self.regime_abs_dist_input.setValue(float(self.cfg.environment.regime_min_abs_dist_ema_240))
        self.regime_abs_dist_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.regime_vol_input = QDoubleSpinBox()
        self.regime_vol_input.setRange(-3.0, 3.0)
        self.regime_vol_input.setDecimals(2)
        self.regime_vol_input.setSingleStep(0.05)
        self.regime_vol_input.setValue(float(self.cfg.environment.regime_min_vol_regime_z))
        self.regime_vol_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.openai_status_label = QLabel("")
        self.openai_status_label.setObjectName("badge")

        self.api_key_hint = QLabel("")
        self.api_key_hint.setObjectName("subtleText")
        _configure_dynamic_label(self.api_key_hint)

        self.smoke_side_button = QPushButton("SELL" if smoke_side == "sell" else "BUY")
        self.smoke_side_button.setCheckable(True)
        self.smoke_side_button.setChecked(smoke_side == "sell")
        self.smoke_side_button.clicked.connect(self._toggle_smoke_side)

        self.smoke_wait_input = QDoubleSpinBox()
        self.smoke_wait_input.setRange(1.0, 30.0)
        self.smoke_wait_input.setDecimals(1)
        self.smoke_wait_input.setSingleStep(1.0)
        self.smoke_wait_input.setValue(float(smoke_wait_seconds))

        self.autocheck_input = QDoubleSpinBox()
        self.autocheck_input.setRange(15.0, 600.0)
        self.autocheck_input.setDecimals(0)
        self.autocheck_input.setSingleStep(15.0)
        self.autocheck_input.setSuffix(" s")
        self.autocheck_input.setValue(float(self.cfg.launcher.auto_check_interval_seconds))
        self.autocheck_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(14)
        form.setColumnStretch(0, 1)
        form.setColumnStretch(1, 2)

        form.addWidget(self._field_label("Risco maximo por trade"), 0, 0)
        form.addWidget(self.risk_input, 0, 1)
        form.addWidget(self._field_label("Filtro HOLD"), 1, 0)
        form.addWidget(self.hold_threshold_input, 1, 1)
        form.addWidget(self._field_label("Regime min dist EMA240"), 2, 0)
        form.addWidget(self.regime_abs_dist_input, 2, 1)
        form.addWidget(self._field_label("Regime min vol z"), 3, 0)
        form.addWidget(self.regime_vol_input, 3, 1)
        form.addWidget(self._field_label("Auto check"), 4, 0)
        form.addWidget(self.autocheck_input, 4, 1)

        root.addLayout(form)
        openai_row = QHBoxLayout()
        openai_row.setSpacing(12)
        openai_row.addWidget(self._field_label("Resumo inteligente via OpenAI"), 1)
        openai_row.addWidget(self.openai_status_label, 0, Qt.AlignRight)
        root.addLayout(openai_row)
        root.addWidget(self.api_key_hint)

        self.advanced_button = QPushButton("Avancado")
        self.advanced_button.setCheckable(True)
        self.advanced_button.clicked.connect(self._toggle_advanced)
        root.addWidget(self.advanced_button, 0, Qt.AlignLeft)

        self.advanced_frame = QFrame()
        self.advanced_frame.setObjectName("AdvancedFrame")
        advanced_layout = QGridLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(0, 4, 0, 0)
        advanced_layout.setHorizontalSpacing(14)
        advanced_layout.setVerticalSpacing(14)
        advanced_layout.setColumnStretch(0, 1)
        advanced_layout.setColumnStretch(1, 2)
        advanced_layout.addWidget(self._field_label("Smoke side"), 0, 0)
        advanced_layout.addWidget(self.smoke_side_button, 0, 1)
        advanced_layout.addWidget(self._field_label("Wait smoke"), 1, 0)
        advanced_layout.addWidget(self.smoke_wait_input, 1, 1)
        self.advanced_frame.hide()
        root.addWidget(self.advanced_frame)

        footer = QHBoxLayout()

        self.reset_button = QPushButton("Voltar ao padrao")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        footer.addWidget(self.reset_button)

        footer.addStretch(1)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        footer.addWidget(self.cancel_button)

        self.save_button = QPushButton("Salvar")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.accept)
        footer.addWidget(self.save_button)

        root.addStretch(1)
        root.addLayout(footer)
        self._refresh_openai_state()

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _toggle_smoke_side(self) -> None:
        self.smoke_side_button.setText("SELL" if self.smoke_side_button.isChecked() else "BUY")

    def _toggle_advanced(self) -> None:
        expanded = self.advanced_button.isChecked()
        self.advanced_button.setText("Ocultar avancado" if expanded else "Avancado")
        self.advanced_frame.setVisible(expanded)

    def _reset_to_defaults(self) -> None:
        self.risk_input.setValue(float(self._dialog_defaults["risk_pct"]))
        self.hold_threshold_input.setValue(float(self._dialog_defaults["hold_threshold"]))
        self.regime_abs_dist_input.setValue(float(self._dialog_defaults["regime_min_abs_dist_ema_240"]))
        self.regime_vol_input.setValue(float(self._dialog_defaults["regime_min_vol_regime_z"]))
        self.autocheck_input.setValue(float(self._dialog_defaults["auto_check_interval"]))

    def _refresh_openai_state(self) -> None:
        key_name = self.cfg.launcher.openai_api_key_env or "OPENAI_API_KEY"
        has_key = _has_env_value(key_name)
        if has_key:
            self.openai_status_label.setText("Ativo")
            _set_widget_property(self.openai_status_label, "severity", "execution")
            self.api_key_hint.setText(f"OpenAI fica sempre ativa no launcher. Chave detectada via {key_name}.")
        else:
            self.openai_status_label.setText("Sem chave")
            _set_widget_property(self.openai_status_label, "severity", "warning")
            self.api_key_hint.setText(
                f"OpenAI fica sempre ativa no launcher, mas nenhuma chave foi detectada em {key_name}. "
                "Sem isso, o resumo local continua como fallback."
            )
        _fit_label_height(self.api_key_hint)

    def smoke_side(self) -> str:
        return "sell" if self.smoke_side_button.isChecked() else "buy"

    def smoke_wait_seconds(self) -> float:
        return float(self.smoke_wait_input.value())

    def risk_pct(self) -> float:
        return float(self.risk_input.value()) / 100.0

    def hold_threshold(self) -> float:
        return float(self.hold_threshold_input.value())

    def regime_min_abs_dist_ema_240(self) -> float:
        return float(self.regime_abs_dist_input.value())

    def regime_min_vol_regime_z(self) -> float:
        return float(self.regime_vol_input.value())

    def auto_check_interval(self) -> int:
        return int(self.autocheck_input.value())



class TraderBotLauncher(QMainWindow):
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        super().__init__()
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.current_mode = "mainnet" if str(self.cfg.hyperliquid.network).lower() == "mainnet" else "testnet"
        self.smoke_side = "buy"
        self.smoke_wait_seconds = 3.0
        self.pending_emergency_close = False
        self._active_task_context: dict[str, Any] = {"label": None, "silent": False}
        self._notification_widgets: dict[str, NotificationItemWidget] = {}
        self._notification_order: list[str] = []
        self._thread_pool = QThreadPool.globalInstance()
        self.state = DashboardState(network_label=self._current_mode().label.upper())
        self.health_state = OperationalHealthState()
        self.last_connection_signature: tuple[Any, ...] | None = None
        self.runtime_stop_requested = False
        self._awaiting_initial_snapshot = True
        self._initial_check_attempts = 0
        self._startup_check_scheduled = False
        self.stats_day = date.today()
        self._active_toast: EventToast | None = None
        self._run_stderr_buffer = ""
        self._task_stderr_buffer = ""

        self.log_translator = OpenAILogTranslator(self.cfg)

        self.run_process = self._build_process(
            self._handle_run_output,
            self._handle_run_stderr,
            self._handle_run_finished,
        )
        self.task_process = self._build_process(
            self._handle_task_output,
            self._handle_task_stderr,
            self._handle_task_finished,
        )

        self.setWindowTitle("Trader Laucher")
        if LAUNCHER_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(LAUNCHER_ICON_PATH)))
        self.resize(1280, 820)
        self.setMinimumSize(960, 680)

        self._build_ui()
        self._hydrate_static_snapshot()
        self._apply_health_status("offline", "startup", emit_event=False)
        self._refresh_dashboard()
        self._apply_responsive_layout()
        self._update_controls()

        self.health_timer = QTimer(self)
        self.health_timer.setInterval(int(self.cfg.launcher.auto_check_interval_seconds) * 1000)
        self.health_timer.timeout.connect(self._auto_check)
        self.health_timer.start()

    def _build_process(self, output_handler, stderr_handler, finished_handler) -> QProcess:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(output_handler)
        process.readyReadStandardError.connect(stderr_handler)
        process.finished.connect(finished_handler)
        return process

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.central_surface = central
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(26, 24, 26, 24)
        root.setSpacing(16)

        root.addWidget(self._build_topbar())
        root.addWidget(self._build_main_navigation(), 0, Qt.AlignHCenter)
        root.addWidget(self._build_dashboard(), 1)

    def _build_topbar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TopBar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        left = QHBoxLayout()
        left.setSpacing(8)
        self.mode_badge = QLabel(self._current_mode().label.upper())
        self.mode_badge.setObjectName("ModeBadge")
        left.addWidget(self.mode_badge, 0, Qt.AlignLeft)
        layout.addLayout(left, 1)

        center = QFrame()
        center.setObjectName("SegmentFrame")
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(6, 6, 6, 6)
        center_layout.setSpacing(8)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key in ("testnet", "mainnet"):
            button = QPushButton(MODES[key].label)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, mode_key=key: self._switch_mode(mode_key))
            center_layout.addWidget(button)
            self.mode_buttons[key] = button
        layout.addWidget(center, 0, Qt.AlignCenter)

        right = QHBoxLayout()
        right.setSpacing(10)
        self.health_chip = QLabel("offline")
        self.health_chip.setObjectName("HealthChip")
        right.addWidget(self.health_chip, 0, Qt.AlignRight)

        self.refresh_button = QPushButton("")
        self.refresh_button.setObjectName("IconButton")
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_button.clicked.connect(lambda: self._run_check(silent=False))
        right.addWidget(self.refresh_button)

        self.settings_button = QPushButton("")
        self.settings_button.setObjectName("IconButton")
        self.settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.settings_button.clicked.connect(self._open_settings)
        right.addWidget(self.settings_button)
        layout.addLayout(right, 1)

        return frame

    def _build_dashboard(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        main_card = QFrame()
        main_card.setObjectName("HeroCard")
        main_layout = QVBoxLayout(main_card)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.main_stack = QStackedWidget()
        self.main_stack.setObjectName("MainStack")
        self.page_keys = ["dashboard", "details", "events", "terminal"]
        self.page_indexes: dict[str, int] = {}
        for key, builder in (
            ("dashboard", self._build_dashboard_tab),
            ("details", self._build_details_tab),
            ("events", self._build_events_tab),
            ("terminal", self._build_terminal_tab),
        ):
            page = builder()
            self.page_indexes[key] = self.main_stack.addWidget(page)
        main_layout.addWidget(self.main_stack)

        layout.addWidget(main_card, 1)
        return container

    def _build_main_navigation(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("SegmentFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.page_buttons: dict[str, QPushButton] = {}
        for key, label in (
            ("dashboard", "Dashboard"),
            ("details", "Detalhes"),
            ("events", "Eventos"),
            ("terminal", "Terminal"),
        ):
            button = QPushButton(label)
            button.setObjectName("TabButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, page_key=key: self._switch_main_page(page_key))
            layout.addWidget(button)
            self.page_buttons[key] = button

        self.current_page = "dashboard"
        self.unread_events = 0
        self._update_navigation_state()
        return frame

    def _build_dashboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        hero_top = QHBoxLayout()
        hero_top.setSpacing(10)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(8)

        self.state_pill = QLabel("AGUARDANDO")
        self.state_pill.setObjectName("StatePill")
        hero_left.addWidget(self.state_pill, 0, Qt.AlignLeft)

        self.state_message = QLabel("Aguardando dados do bot")
        self.state_message.setObjectName("HeroTitle")
        _configure_dynamic_label(self.state_message)
        hero_left.addWidget(self.state_message)

        hero_top.addLayout(hero_left, 1)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.run_button = QPushButton("INICIAR BOT")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.setProperty("dashboardAction", True)
        self.run_button.setProperty("runtimeState", "idle")
        self.run_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.run_button.clicked.connect(self._toggle_run)
        actions.addWidget(self.run_button)

        self.smoke_button = QPushButton("Smoke test")
        self.smoke_button.setObjectName("SecondaryButton")
        self.smoke_button.setProperty("dashboardAction", True)
        self.smoke_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.smoke_button.clicked.connect(self._run_smoke)
        actions.addWidget(self.smoke_button)

        self.kill_button = QPushButton("Kill switch")
        self.kill_button.setObjectName("DangerGhostButton")
        self.kill_button.setProperty("dashboardAction", True)
        self.kill_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.kill_button.clicked.connect(self._trigger_kill_switch)
        actions.addWidget(self.kill_button)

        hero_top.addLayout(actions, 0)
        layout.addLayout(hero_top)

        hero_bottom = QHBoxLayout()
        hero_bottom.setSpacing(14)

        balance_stack = QVBoxLayout()
        balance_stack.setSpacing(4)
        self.balance_caption = QLabel("Saldo operacional")
        self.balance_caption.setObjectName("MetricLabel")
        balance_stack.addWidget(self.balance_caption)
        self.balance_inline = QLabel("--")
        self.balance_inline.setObjectName("BalanceValue")
        _configure_dynamic_label(self.balance_inline)
        balance_stack.addWidget(self.balance_inline)
        hero_bottom.addLayout(balance_stack)
        hero_bottom.addStretch(1)
        layout.addLayout(hero_bottom)

        self.dashboard_metrics_grid = QGridLayout()
        self.dashboard_metrics_grid.setContentsMargins(0, 0, 0, 0)
        self.dashboard_metrics_grid.setHorizontalSpacing(12)
        self.dashboard_metrics_grid.setVerticalSpacing(12)

        self.signal_tile = MetricTile("Sinal", compact=True, min_height=118)
        self.reason_tile = MetricTile("Motivo", wrap_value=True, min_height=144)
        self.price_tile = MetricTile("Preço de referência", compact=True, min_height=118)
        self.risk_tile = MetricTile("Risco efetivo", compact=True, min_height=118)
        self.position_tile = MetricTile("Posição atual", compact=True, min_height=118)
        self.pnl_tile = MetricTile("PnL aberto", compact=True, min_height=118)
        self.dashboard_metric_tiles = [
            self.signal_tile,
            self.reason_tile,
            self.price_tile,
            self.risk_tile,
            self.position_tile,
            self.pnl_tile,
        ]
        layout.addLayout(self.dashboard_metrics_grid)

        position = QFrame()
        position.setObjectName("PositionCard")
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(18, 18, 18, 18)
        position_layout.setSpacing(12)

        position_top = QHBoxLayout()
        position_top.setSpacing(12)

        position_meta = QVBoxLayout()
        position_meta.setSpacing(6)

        self.position_pill = QLabel("FLAT")
        self.position_pill.setObjectName("PositionPill")
        position_meta.addWidget(self.position_pill, 0, Qt.AlignLeft)

        self.position_status = QLabel("Sem sinal ativo")
        self.position_status.setObjectName("PositionStatus")
        _configure_dynamic_label(self.position_status)
        position_meta.addWidget(self.position_status)

        position_top.addLayout(position_meta, 1)

        pnl_box = QVBoxLayout()
        pnl_box.setSpacing(4)
        pnl_label = QLabel("PnL aberto")
        pnl_label.setObjectName("MetricLabel")
        pnl_box.addWidget(pnl_label)
        self.pnl_value = QLabel("--")
        self.pnl_value.setObjectName("PnlValue")
        _configure_dynamic_label(self.pnl_value)
        pnl_box.addWidget(self.pnl_value)
        position_top.addLayout(pnl_box)

        size_box = QVBoxLayout()
        size_box.setSpacing(4)
        size_label = QLabel("Tamanho")
        size_label.setObjectName("MetricLabel")
        size_box.addWidget(size_label)
        self.position_size_value = QLabel("--")
        self.position_size_value.setObjectName("PositionNotional")
        _configure_dynamic_label(self.position_size_value)
        size_box.addWidget(self.position_size_value)
        position_top.addLayout(size_box)

        position_layout.addLayout(position_top)

        self.levels_grid = QGridLayout()
        self.levels_grid.setContentsMargins(0, 0, 0, 0)
        self.levels_grid.setHorizontalSpacing(12)
        self.levels_grid.setVerticalSpacing(10)

        self.entry_box, self.entry_value = self._build_level_box("Entrada")
        self.tp_box, self.tp_value = self._build_level_box("Take profit")
        self.sl_box, self.sl_value = self._build_level_box("Stop loss")
        self.level_boxes = [self.entry_box, self.tp_box, self.sl_box]
        position_layout.addLayout(self.levels_grid)

        layout.addWidget(position, 1)

        return tab

    def _build_details_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QFrame.NoFrame)
        self.details_scroll.setObjectName("DetailsScroll")
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.viewport().setObjectName("DetailsViewport")

        details_host = QWidget()
        details_host.setObjectName("DetailsHost")
        details_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        content_layout = QVBoxLayout(details_host)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(18)

        self.details_overview_grid = QGridLayout()
        self.details_overview_grid.setContentsMargins(0, 0, 0, 0)
        self.details_overview_grid.setHorizontalSpacing(14)
        self.details_overview_grid.setVerticalSpacing(14)
        content_layout.addLayout(self.details_overview_grid)

        hero_card = QFrame()
        hero_card.setObjectName("PositionCard")
        hero_card.setMinimumHeight(152)
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(8)

        hero_eyebrow = QLabel("Última decisão relevante")
        hero_eyebrow.setObjectName("fieldLabel")
        hero_layout.addWidget(hero_eyebrow)

        self.details_decision_label = QLabel("Sem decisão no momento.")
        self.details_decision_label.setObjectName("PositionStatus")
        _configure_dynamic_label(self.details_decision_label)
        hero_layout.addWidget(self.details_decision_label)

        self.details_decision_reason_label = QLabel("Sem contexto recente.")
        self.details_decision_reason_label.setObjectName("HeroHint")
        _configure_dynamic_label(self.details_decision_reason_label)
        hero_layout.addWidget(self.details_decision_reason_label)

        self.details_decision_meta_label = QLabel(
            "Sinal nenhum • Direção nenhuma • Regime indefinido • Posição FLAT"
        )
        self.details_decision_meta_label.setObjectName("subtleText")
        _configure_dynamic_label(self.details_decision_meta_label)
        hero_layout.addWidget(self.details_decision_meta_label)

        self.details_overview_cards = [hero_card]

        self.details_metrics_grid = QGridLayout()
        self.details_metrics_grid.setContentsMargins(0, 0, 0, 0)
        self.details_metrics_grid.setHorizontalSpacing(14)
        self.details_metrics_grid.setVerticalSpacing(14)

        self.connection_tile = MetricTile("Conexão", compact=True, min_height=92)
        self.heartbeat_tile = MetricTile("Último check válido", compact=True, min_height=92)
        self.operations_today_tile = MetricTile("Operações hoje", compact=True, min_height=92)
        self.blocked_today_tile = MetricTile("Bloqueios hoje", compact=True, min_height=92)
        self.details_metric_tiles = [
            self.connection_tile,
            self.heartbeat_tile,
            self.operations_today_tile,
            self.blocked_today_tile,
        ]
        content_layout.addLayout(self.details_metrics_grid)

        technical_header = QHBoxLayout()
        technical_header.setSpacing(10)

        technical_copy = QVBoxLayout()
        technical_copy.setSpacing(4)
        technical_title = QLabel("Analise tecnica do ultimo ciclo")
        technical_title.setObjectName("fieldLabel")
        technical_copy.addWidget(technical_title)

        technical_hint = QLabel(
            "Cards alimentados diretamente pelos textos humanizados da IA para o ultimo ciclo."
        )
        technical_hint.setObjectName("HeroHint")
        _configure_dynamic_label(technical_hint)
        technical_copy.addWidget(technical_hint)
        technical_header.addLayout(technical_copy, 1)

        self.details_prev_button = QPushButton("Anterior")
        self.details_prev_button.setObjectName("SecondaryButton")
        self.details_prev_button.clicked.connect(self._show_previous_detail_card)
        technical_header.addWidget(self.details_prev_button, 0, Qt.AlignRight)

        self.details_carousel_label = QLabel("1/1")
        self.details_carousel_label.setObjectName("metaLabel")
        technical_header.addWidget(self.details_carousel_label, 0, Qt.AlignRight)

        self.details_next_button = QPushButton("Proximo")
        self.details_next_button.setObjectName("SecondaryButton")
        self.details_next_button.clicked.connect(self._show_next_detail_card)
        technical_header.addWidget(self.details_next_button, 0, Qt.AlignRight)
        content_layout.addLayout(technical_header)

        self.details_technical_stack = QStackedWidget()
        self.details_technical_stack.currentChanged.connect(self._update_details_carousel_navigation)
        content_layout.addWidget(self.details_technical_stack)

        self.indicators_card = DetailFieldCard("Leitura do Mercado")
        self.intelligence_card = DetailFieldCard("Interpretacao do Modelo")
        self.filters_card = DetailFieldCard("Diagnostico dos Filtros")

        self.details_technical_cards = [
            self.indicators_card,
            self.intelligence_card,
            self.filters_card,
        ]
        self._details_carousel_signature: tuple[int, int] | None = None
        self._rebuild_details_carousel(items_per_page=3, columns=3)

        def build_detail_section(title: str, *, min_height: int = 122) -> tuple[QFrame, QLabel, QLabel]:
            card = QFrame()
            card.setObjectName("PositionCard")
            card.setMinimumHeight(min_height)
            section_layout = QVBoxLayout(card)
            section_layout.setContentsMargins(18, 16, 18, 16)
            section_layout.setSpacing(8)

            title_label = QLabel(title)
            title_label.setObjectName("fieldLabel")
            section_layout.addWidget(title_label)

            value_label = QLabel("--")
            value_label.setObjectName("metricValue")
            value_label.setProperty("sizeVariant", "body")
            _configure_dynamic_label(value_label)
            section_layout.addWidget(value_label)

            note_label = QLabel("")
            note_label.setObjectName("HeroHint")
            _configure_dynamic_label(note_label)
            note_label.hide()
            section_layout.addWidget(note_label)

            return card, value_label, note_label

        self.details_sections_grid = QGridLayout()
        self.details_sections_grid.setContentsMargins(0, 0, 0, 0)
        self.details_sections_grid.setHorizontalSpacing(14)
        self.details_sections_grid.setVerticalSpacing(14)

        self.details_context_card, self.details_context_value, self.details_context_note = build_detail_section(
            "Contexto da decisão"
        )
        self.details_trade_card, self.details_trade_value, self.details_trade_note = build_detail_section(
            "Última operação"
        )
        self.details_skip_card, self.details_skip_value, self.details_skip_note = build_detail_section(
            "Última não entrada"
        )
        self.details_features_card, self.details_features_value, self.details_features_note = build_detail_section(
            "Leitura das features",
            min_height=128,
        )
        self.details_section_cards = [
            self.details_context_card,
            self.details_trade_card,
            self.details_skip_card,
            self.details_features_card,
        ]
        content_layout.addLayout(self.details_sections_grid)

        self.details_scroll.setWidget(details_host)
        layout.addWidget(self.details_scroll, 1)

        return tab

    def _build_events_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("EventsTab")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        self.events_empty_label = QLabel("Nenhum evento relevante ainda.")
        self.events_empty_label.setObjectName("HeroHint")
        _configure_dynamic_label(self.events_empty_label)
        layout.addWidget(self.events_empty_label)

        self.events_scroll = QScrollArea()
        self.events_scroll.setWidgetResizable(True)
        self.events_scroll.setFrameShape(QFrame.NoFrame)
        self.events_scroll.setObjectName("EventsScroll")
        self.events_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.events_scroll.viewport().setObjectName("EventsViewport")

        self.notification_host = QWidget()
        self.notification_host.setObjectName("EventsHost")
        self.notification_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.notifications_layout = QVBoxLayout(self.notification_host)
        self.notifications_layout.setContentsMargins(0, 0, 0, 0)
        self.notifications_layout.setSpacing(12)
        self.notifications_layout.setAlignment(Qt.AlignTop)
        self.notifications_layout.addStretch(1)
        self.events_scroll.setWidget(self.notification_host)
        layout.addWidget(self.events_scroll, 1)
        return tab

    def _build_terminal_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("TerminalTab")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        terminal_card = QFrame()
        terminal_card.setObjectName("PositionCard")
        terminal_layout = QVBoxLayout(terminal_card)
        terminal_layout.setContentsMargins(18, 16, 18, 16)
        terminal_layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)

        title = QLabel("Terminal")
        title.setObjectName("PanelTitle")
        header.addWidget(title)
        header.addStretch(1)

        self.clear_terminal_button = QPushButton("Limpar terminal")
        self.clear_terminal_button.setObjectName("SecondaryButton")
        self.clear_terminal_button.clicked.connect(self._clear_terminal_output)
        header.addWidget(self.clear_terminal_button)
        terminal_layout.addLayout(header)

        hint = QLabel("Espelho bruto do stdout e stderr do runtime e dos comandos auxiliares.")
        hint.setObjectName("HeroHint")
        _configure_dynamic_label(hint)
        terminal_layout.addWidget(hint)

        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setObjectName("technicalLog")
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.terminal_output.setPlaceholderText("Aguardando saida dos processos...")
        self.terminal_output.document().setMaximumBlockCount(6000)
        terminal_layout.addWidget(self.terminal_output, 1)

        layout.addWidget(terminal_card, 1)
        return tab

    def _build_level_box(self, title: str) -> tuple[QFrame, QLabel]:
        wrapper = QFrame()
        wrapper.setObjectName("LevelBox")
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(14, 12, 14, 12)
        inner.setSpacing(6)

        label = QLabel(title)
        label.setObjectName("MetricLabel")
        inner.addWidget(label)

        value = QLabel("--")
        value.setObjectName("LevelValue")
        _configure_dynamic_label(value)
        inner.addWidget(value)
        return wrapper, value

    def _hydrate_static_snapshot(self) -> None:
        self._sync_mode_buttons()
        self._update_events_empty_state()

    def _switch_main_page(self, page_key: str) -> None:
        if page_key not in self.page_indexes:
            return
        self.current_page = page_key
        self.main_stack.setCurrentIndex(self.page_indexes[page_key])
        if page_key == "events":
            self.unread_events = 0
        self._update_navigation_state()

    def _clear_layout(self, layout: QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _detach_layout_items(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                self._detach_layout_items(child_layout)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _relayout_grid_items(self, layout: QGridLayout, widgets: list[QWidget], columns: int) -> None:
        columns = max(1, columns)
        self._clear_layout(layout)
        for index, widget in enumerate(widgets):
            row = index // columns
            column = index % columns
            layout.addWidget(widget, row, column)
        for column in range(columns):
            layout.setColumnStretch(column, 1)
        rows = (len(widgets) + columns - 1) // columns
        for row in range(rows):
            layout.setRowStretch(row, 1)

    def _apply_responsive_layout(self) -> None:
        width = max(self.width(), self.size().width())
        dashboard_width = max(width - 120, 760)

        density = "compact" if width < 1320 else "regular"
        _set_widget_property(self.central_surface, "density", density)

        if dashboard_width >= 1180:
            metrics_columns = 3
        elif dashboard_width >= 760:
            metrics_columns = 2
        else:
            metrics_columns = 1

        detail_overview_columns = 2 if dashboard_width >= 980 else 1
        detail_columns = 4 if dashboard_width >= 1280 else 2 if dashboard_width >= 760 else 1
        detail_section_columns = 2 if dashboard_width >= 760 else 1
        level_columns = 3 if dashboard_width >= 980 else 1
        detail_carousel_items = 3 if dashboard_width >= 1340 else 2 if dashboard_width >= 940 else 1
        detail_carousel_columns = min(detail_carousel_items, 3)

        self._relayout_grid_items(self.dashboard_metrics_grid, self.dashboard_metric_tiles, metrics_columns)
        if hasattr(self, "details_overview_grid"):
            self._relayout_grid_items(self.details_overview_grid, self.details_overview_cards, detail_overview_columns)
        self._relayout_grid_items(self.details_metrics_grid, self.details_metric_tiles, detail_columns)
        if hasattr(self, "details_technical_stack"):
            self._rebuild_details_carousel(detail_carousel_items, detail_carousel_columns)
        self._relayout_grid_items(self.details_sections_grid, self.details_section_cards, detail_section_columns)
        self._relayout_grid_items(self.levels_grid, self.level_boxes, level_columns)
        if hasattr(self, "notifications_layout"):
            self._trim_notifications()

    def _rebuild_details_carousel(self, items_per_page: int, columns: int) -> None:
        cards = getattr(self, "details_technical_cards", [])
        if not hasattr(self, "details_technical_stack") or not cards:
            return

        signature = (max(1, items_per_page), max(1, columns))
        if self._details_carousel_signature == signature and self.details_technical_stack.count():
            self._update_details_carousel_navigation()
            return

        current_index = max(self.details_technical_stack.currentIndex(), 0)
        while self.details_technical_stack.count():
            page = self.details_technical_stack.widget(0)
            page_layout = page.layout()
            if page_layout is not None:
                self._detach_layout_items(page_layout)
            self.details_technical_stack.removeWidget(page)
            page.deleteLater()

        for start in range(0, len(cards), signature[0]):
            chunk = cards[start : start + signature[0]]
            page = QWidget()
            page_layout = QGridLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setHorizontalSpacing(14)
            page_layout.setVerticalSpacing(14)
            for index, card in enumerate(chunk):
                row = index // signature[1]
                column = index % signature[1]
                page_layout.addWidget(card, row, column)
            for column in range(signature[1]):
                page_layout.setColumnStretch(column, 1)
            self.details_technical_stack.addWidget(page)

        total_pages = self.details_technical_stack.count()
        if total_pages:
            self.details_technical_stack.setCurrentIndex(min(current_index, total_pages - 1))
        self._details_carousel_signature = signature
        self._update_details_carousel_navigation()

    def _show_previous_detail_card(self) -> None:
        if not hasattr(self, "details_technical_stack"):
            return
        current_index = self.details_technical_stack.currentIndex()
        if current_index > 0:
            self.details_technical_stack.setCurrentIndex(current_index - 1)

    def _show_next_detail_card(self) -> None:
        if not hasattr(self, "details_technical_stack"):
            return
        current_index = self.details_technical_stack.currentIndex()
        if current_index < self.details_technical_stack.count() - 1:
            self.details_technical_stack.setCurrentIndex(current_index + 1)

    def _update_details_carousel_navigation(self) -> None:
        if not hasattr(self, "details_technical_stack"):
            return
        total_pages = self.details_technical_stack.count()
        current_page = self.details_technical_stack.currentIndex() + 1 if total_pages else 0
        multi_page = total_pages > 1
        self.details_prev_button.setVisible(multi_page)
        self.details_next_button.setVisible(multi_page)
        self.details_carousel_label.setVisible(multi_page)
        self.details_prev_button.setEnabled(current_page > 1)
        self.details_next_button.setEnabled(current_page < total_pages)
        self.details_carousel_label.setText(f"{current_page}/{total_pages}" if total_pages else "--/--")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "main_stack"):
            self._apply_responsive_layout()
        if hasattr(self, "_active_toast") and self._active_toast_visible():
            self._position_toast(self._active_toast)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Interceta o fecho da janela para evitar processos zumbis e proteger posições abertas."""
        bot_running = self.run_process.state() != QProcess.NotRunning
        position_open = self.state.position_label in {"LONG", "SHORT"}

        if bot_running and position_open:
            answer = QMessageBox.warning(
                self,
                "Atenção: Posição Aberta!",
                "Existe uma posição aberta na exchange!\n\n"
                "Se fechar o Launcher agora, o bot irá parar e a posição ficará à deriva.\n"
                "Deseja fechar mesmo assim?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return

        # Limpar requisicoes da IA que ficaram presas na fila e dar no maximo 1 segundo para terminarem
        self._thread_pool.clear()
        self._thread_pool.waitForDone(1000)

        if bot_running:
            self.run_process.terminate()
            self.run_process.waitForFinished(2000)

        if self.task_process.state() != QProcess.NotRunning:
            self.task_process.terminate()
            self.task_process.waitForFinished(2000)

        event.accept()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._startup_check_scheduled:
            return
        self._startup_check_scheduled = True
        self._schedule_initial_check(900)

    def _sync_mode_buttons(self) -> None:
        mode = self._current_mode()
        self.mode_badge.setText(mode.label.upper())
        for key, button in self.mode_buttons.items():
            button.setChecked(key == self.current_mode)

    def _has_complete_initial_snapshot(self) -> bool:
        return (
            self.health_state.last_successful_healthcheck_at is not None
            and self.state.balance_label != "--"
        )

    def _schedule_initial_check(self, delay_ms: int) -> None:
        QTimer.singleShot(delay_ms, self._run_initial_check)

    def _run_initial_check(self) -> None:
        if not self._awaiting_initial_snapshot:
            return
        if self._has_complete_initial_snapshot():
            self._awaiting_initial_snapshot = False
            return
        if self.task_process.state() != QProcess.NotRunning:
            self._schedule_initial_check(1200)
            return
        self._initial_check_attempts += 1
        self._run_check(silent=True)

    def _update_controls(self) -> None:
        run_busy = self.run_process.state() != QProcess.NotRunning
        task_busy = self.task_process.state() != QProcess.NotRunning
        task_label = str(self._active_task_context.get("label") or "")
        for button in self.mode_buttons.values():
            button.setEnabled(not run_busy and not task_busy)
        self.refresh_button.setEnabled(not task_busy)
        self.smoke_button.setEnabled(not task_busy and not run_busy)
        self.settings_button.setEnabled(not task_busy and not run_busy)
        self.kill_button.setEnabled(not task_busy)
        self.run_button.setEnabled(run_busy or not task_busy)
        if run_busy and self.runtime_stop_requested:
            run_label = "PARANDO..."
            run_state = "stopping"
        elif run_busy and self.state.state_label == "INICIANDO":
            run_label = "INICIANDO..."
            run_state = "starting"
        elif run_busy:
            run_label = "BOT RODANDO"
            run_state = "running"
        else:
            run_label = "INICIAR BOT"
            run_state = "idle"
        self.run_button.setText(run_label)
        _set_widget_property(self.run_button, "runtimeState", run_state)
        self.smoke_button.setText("Executando..." if task_busy and task_label == "Smoke test" else "Smoke test")

    def _current_mode(self) -> LauncherMode:
        return MODES[self.current_mode]

    def _build_base_args(self, mode: LauncherMode) -> list[str]:
        args = ["-u", "-m", "traderbot.main", "--config", str(self.config_path)]
        args.extend(["--network-override", mode.network])
        args.extend(["--execution-mode-override", mode.execution_mode])
        if mode.allow_live_trading:
            args.append("--allow-live-trading")
        return args

    def _resolve_process_python_executable(self) -> str:
        executable = Path(sys.executable)
        name = executable.name.lower()

        candidates: list[Path] = []
        if name == "pythonw.exe":
            candidates.append(executable.with_name("python.exe"))
        elif name == "pyw.exe":
            py_path = shutil.which("py")
            python_path = shutil.which("python")
            if py_path:
                candidates.append(Path(py_path))
            if python_path:
                candidates.append(Path(python_path))

        candidates.append(executable)
        python_on_path = shutil.which("python")
        if python_on_path:
            candidates.append(Path(python_on_path))

        seen: set[str] = set()
        for candidate in candidates:
            candidate_str = str(candidate)
            if not candidate_str:
                continue
            normalized = candidate_str.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists() and candidate.name.lower() in {"python.exe", "py.exe", "python", "py"}:
                return candidate_str

        return str(executable)

    def _new_event(
        self,
        *,
        source: str,
        event_type: str,
        raw_line: str,
        message: str = "",
        payload: dict[str, Any] | None = None,
        severity: str = "info",
        event_code: str | None = None,
    ) -> LauncherEvent:
        metadata = dict(payload or {})
        network = str(metadata.get("network") or self._current_mode().network)
        symbol = str(metadata.get("symbol") or self.cfg.hyperliquid.symbol)
        timeframe = str(metadata.get("timeframe") or self.cfg.hyperliquid.timeframe)
        metadata.setdefault("network", network)
        metadata.setdefault("symbol", symbol)
        metadata.setdefault("timeframe", timeframe)
        return LauncherEvent(
            source=source,
            event_type=event_type,
            raw_line=raw_line,
            message=message,
            payload=metadata,
            severity=severity,
            event_code=event_code,
            network=network,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _start_process(self, process: QProcess, args: list[str], *, label: str, silent: bool = False) -> bool:
        process.setWorkingDirectory(str(REPO_ROOT))
        self._reset_process_stderr_buffer(process)
        self._active_task_context = {"label": label, "silent": silent}
        process.start(self._resolve_process_python_executable(), args)
        if not process.waitForStarted(3000):
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=f"Não foi possível iniciar {label}.",
                message=f"Não foi possível iniciar {label}.",
                event_code="system.process_start_failed",
                severity="error",
            )
            self._push_event(event)
            return False
        if not silent:
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=f"{label} iniciado.",
                message=f"{label} iniciado.",
                event_code="system.process_started",
            )
            self._push_event(event)
        self._update_controls()
        return True

    def _reset_process_stderr_buffer(self, process: QProcess) -> None:
        if process is self.run_process:
            self._run_stderr_buffer = ""
            return
        if process is self.task_process:
            self._task_stderr_buffer = ""

    def _append_process_stderr(self, process: QProcess) -> None:
        payload = bytes(process.readAllStandardError()).decode("utf-8", errors="ignore")
        if not payload:
            return
        self._append_terminal_output(payload)
        if process is self.run_process:
            self._run_stderr_buffer += payload
            return
        if process is self.task_process:
            self._task_stderr_buffer += payload

    def _consume_process_stderr(self, process: QProcess) -> str:
        if process is self.run_process:
            payload = self._run_stderr_buffer
            self._run_stderr_buffer = ""
            return payload
        if process is self.task_process:
            payload = self._task_stderr_buffer
            self._task_stderr_buffer = ""
            return payload
        return ""

    def _stderr_summary_line(self, stderr_buffer: str, *, fallback: str) -> str:
        for line in reversed(stderr_buffer.splitlines()):
            candidate = self._strip_prefix(line.strip())
            if candidate:
                return candidate
        return fallback

    def _show_process_error_dialog(self, *, label: str, summary: str, stderr_buffer: str, exit_code: int) -> None:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle(f"{label} com erro")
        dialog.setText(summary)
        dialog.setInformativeText(f"{label} finalizado com falha (exit={exit_code}).")
        dialog.setDetailedText(stderr_buffer)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def _emit_process_error(
        self,
        *,
        process_label: str,
        source: str,
        exit_code: int,
        stderr_buffer: str,
        event_code: str,
        show_dialog: bool,
    ) -> None:
        fallback = f"{process_label} finalizado com codigo {exit_code}."
        summary = self._stderr_summary_line(stderr_buffer, fallback=fallback)
        event = self._new_event(
            source=source,
            event_type="generic",
            raw_line=stderr_buffer or fallback,
            message=summary,
            payload={
                "process_label": process_label,
                "exit_code": exit_code,
                "stderr_summary": summary,
            },
            severity="error",
            event_code=event_code,
        )
        prebuilt = HumanizedEvent(
            source=event.source,
            event_type=event.event_type,
            raw_line=event.raw_line,
            message=summary,
            payload=dict(event.payload or {}),
            occurred_at=event.occurred_at,
            severity="error",
            event_code=event.event_code,
            details=f"{process_label} finalizado com falha (exit={exit_code}).",
            network=event.network,
            symbol=event.symbol,
            timeframe=event.timeframe,
            color="red",
            relevant=True,
            fingerprint=f"{event_code}:{process_label.lower()}:{summary.lower()}",
            raw_detail=stderr_buffer or fallback,
            execution_summary=f"{process_label} finalizado com falha (exit={exit_code}).",
            simple_summary=summary,
        )
        self._push_event(event, prebuilt=prebuilt)
        if show_dialog:
            self._show_process_error_dialog(
                label=process_label,
                summary=summary,
                stderr_buffer=stderr_buffer or fallback,
                exit_code=exit_code,
            )

    def _sync_runtime_stopped_ui(self, *, exit_code: int, stop_requested: bool) -> None:
        self.health_state.bot_running = False
        self.health_state.executor_alive = False

        self._refresh_connectivity_health(emit_event=not stop_requested)

        if exit_code != 0 and not stop_requested:
            summary = "Bot parado após falha do runtime. Revise o erro para retomar a operação."
            style = "error"
        else:
            summary = "Bot parado. Inicie o runtime para voltar a operar."
            style = "blocked"

        self.state.state_label = "PARADO"
        self.state.state_message = summary
        self.state.state_style = style
        self.state.last_decision = summary
        self.state.humanized_simple_summary = summary
        self.state.humanized_execution_summary = summary
        self.state.last_cycle_fingerprint = ""

        if self.state.position_label in {"LONG", "SHORT"}:
            self.state.position_status = f"Bot parado; revisar posição {self.state.position_label} na exchange."
        else:
            self.state.position_status = "Bot parado. Nenhum runtime ativo."

        self._refresh_dashboard()
        self._update_controls()

    def _run_check(self, *, silent: bool = False) -> None:
        if self.task_process.state() != QProcess.NotRunning:
            return
        args = self._build_base_args(self._current_mode()) + ["check-hyperliquid"]
        self._start_process(self.task_process, args, label="Check Hyperliquid", silent=silent)

    def _auto_check(self) -> None:
        self._run_check(silent=True)

    def _run_smoke(self) -> None:
        mode = self._current_mode()
        args = self._build_base_args(mode) + [
            "smoke-hyperliquid",
            "--side",
            self.smoke_side,
            "--wait-seconds",
            str(self.smoke_wait_seconds),
            "--network",
            mode.network,
        ]
        if mode.network == "mainnet":
            args.append("--allow-mainnet")
        self._start_process(self.task_process, args, label="Smoke test", silent=False)

    def _toggle_run(self) -> None:
        if self.run_process.state() == QProcess.NotRunning:
            self.runtime_stop_requested = False
            args = self._build_base_args(self._current_mode()) + ["run"]
            if self._start_process(self.run_process, args, label="Runtime", silent=False):
                self.health_state.bot_running = True
                self.health_state.executor_alive = True
                self.state.state_label = "INICIANDO"
                self.state.state_message = "Runtime iniciado. Primeira avaliacao em andamento."
                self.state.state_style = "wait"
                self.state.last_decision = "Runtime iniciado. Primeira avaliacao em andamento."
                self.state.humanized_simple_summary = ""
                self.state.humanized_execution_summary = ""
                self.state.last_cycle_fingerprint = ""
                if self.state.position_label not in {"LONG", "SHORT"}:
                    self.state.position_status = "Aguardando primeira avaliacao do runtime."
                self._update_controls()
                self._refresh_dashboard()
                self._push_event(
                    self._new_event(
                        source="launcher",
                        event_type="generic",
                        raw_line="Runtime iniciado. Primeira avaliacao em andamento.",
                        message="Runtime iniciado. Primeira avaliacao em andamento.",
                        severity="info",
                        event_code="system.runtime_started",
                    )
                )
            return

        if self.state.position_label in {"LONG", "SHORT"}:
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line="Existe posição aberta. Use o kill switch antes de parar o bot.",
                message="Existe posição aberta. Use o kill switch antes de parar o bot.",
                severity="warning",
                event_code="risk.stop_blocked_open_position",
            )
            self._push_event(event)
            return

        self.runtime_stop_requested = True
        self._update_controls()
        self.run_process.terminate()

        # Fallback de seguranca para garantir o encerramento apos 3 segundos se o terminate falhar
        QTimer.singleShot(
            3000,
            lambda: self.run_process.kill() if self.run_process.state() != QProcess.NotRunning else None,
        )

    def _switch_mode(self, mode_key: str) -> None:
        if mode_key == self.current_mode:
            return
        if self.run_process.state() != QProcess.NotRunning or self.task_process.state() != QProcess.NotRunning:
            return
        self.current_mode = mode_key
        self.state.network_label = self._current_mode().label.upper()
        self._sync_mode_buttons()
        self.health_state = OperationalHealthState(reason="network_switch_pending_check")
        self._apply_health_status("offline", "network_switch_pending_check", emit_event=False)
        self.last_connection_signature = None
        self.state.last_decision = f"{self._current_mode().label} selecionada. Atualizando check..."
        self.state.last_update_label = "--"
        self._refresh_dashboard()
        self._run_check(silent=False)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.cfg,
            smoke_side=self.smoke_side,
            smoke_wait_seconds=self.smoke_wait_seconds,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        self.smoke_side = dialog.smoke_side()
        self.smoke_wait_seconds = dialog.smoke_wait_seconds()
        self._save_settings(
            max_risk_per_trade=dialog.risk_pct(),
            action_hold_threshold=dialog.hold_threshold(),
            regime_min_abs_dist_ema_240=dialog.regime_min_abs_dist_ema_240(),
            regime_min_vol_regime_z=dialog.regime_min_vol_regime_z(),
            auto_check_interval=dialog.auto_check_interval(),
        )

    def _save_settings(
        self,
        *,
        max_risk_per_trade: float,
        action_hold_threshold: float,
        regime_min_abs_dist_ema_240: float,
        regime_min_vol_regime_z: float,
        auto_check_interval: int,
    ) -> None:
        path = Path(self.config_path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        raw.setdefault("environment", {})
        raw["environment"]["max_risk_per_trade"] = float(max_risk_per_trade)
        raw["environment"]["action_hold_threshold"] = float(action_hold_threshold)
        raw["environment"]["regime_min_abs_dist_ema_240"] = float(regime_min_abs_dist_ema_240)
        raw["environment"]["regime_min_vol_regime_z"] = float(regime_min_vol_regime_z)

        raw.setdefault("launcher", {})
        raw["launcher"].pop("openai_enabled", None)
        raw["launcher"]["auto_check_interval_seconds"] = int(auto_check_interval)

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)

        self.cfg = load_config(path)
        self.log_translator = OpenAILogTranslator(self.cfg)
        self.health_timer.setInterval(int(self.cfg.launcher.auto_check_interval_seconds) * 1000)
        self._hydrate_static_snapshot()
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        if self.state.decision_snapshot:
            final_action = float(self.state.decision_snapshot.get("final_action", 0.0) or 0.0)
            self.state.signal_label = self._signal_label(final_action)
        self._refresh_dashboard()
        key_name = self.cfg.launcher.openai_api_key_env or "OPENAI_API_KEY"
        has_key = _has_env_value(key_name)
        self._push_event(
            self._new_event(
                source="launcher",
                event_type="generic",
                raw_line="Configuracoes salvas.",
                message=(
                    f"Configuracoes salvas. Risco {self.cfg.environment.max_risk_per_trade * 100.0:.2f}%, "
                    f"filtro HOLD {self.cfg.environment.action_hold_threshold:.2f}, "
                    f"regime dist>={self.cfg.environment.regime_min_abs_dist_ema_240:.3f}, "
                    f"regime vol>={self.cfg.environment.regime_min_vol_regime_z:.2f}, "
                    f"auto check {int(self.cfg.launcher.auto_check_interval_seconds)}s e OpenAI "
                    f"{'pronta' if has_key else 'sem chave'}."
                ),
                event_code="system.settings_saved",
            )
        )

    def _trigger_kill_switch(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Confirmar kill switch",
            "Encerrar a posição atual e parar o bot?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.pending_emergency_close = True
        if self.run_process.state() != QProcess.NotRunning:
            self.run_process.kill()
            QTimer.singleShot(900, self._execute_pending_kill_switch)
        else:
            self._execute_pending_kill_switch()

    def _execute_pending_kill_switch(self) -> None:
        if not self.pending_emergency_close:
            return

        # Se houver uma tarefa rodando (ex: healthcheck travado), matamos sem piedade
        # para libertar a via para o comando de emergencia.
        if self.task_process.state() != QProcess.NotRunning:
            self.task_process.kill()
            self.task_process.waitForFinished(1000)

        self.pending_emergency_close = False
        args = self._build_base_args(self._current_mode()) + ["close-hyperliquid-position"]
        self._start_process(self.task_process, args, label="Kill switch", silent=False)

    def _clear_terminal_output(self) -> None:
        if hasattr(self, "terminal_output"):
            self.terminal_output.clear()

    def _append_terminal_output(self, text: str) -> None:
        if not text or not hasattr(self, "terminal_output"):
            return
        self.terminal_output.moveCursor(QTextCursor.End)
        self.terminal_output.insertPlainText(text)
        self.terminal_output.moveCursor(QTextCursor.End)
        self.terminal_output.ensureCursorVisible()

    def _handle_run_output(self) -> None:
        while self.run_process.canReadLine():
            raw_payload = bytes(self.run_process.readLine()).decode("utf-8", errors="ignore")
            self._append_terminal_output(raw_payload)
            self._parse_line(raw_payload.rstrip("\r\n"), source="run")

    def _handle_run_stderr(self) -> None:
        self._append_process_stderr(self.run_process)

    def _handle_task_output(self) -> None:
        while self.task_process.canReadLine():
            raw_payload = bytes(self.task_process.readLine()).decode("utf-8", errors="ignore")
            self._append_terminal_output(raw_payload)
            self._parse_line(raw_payload.rstrip("\r\n"), source="task")

    def _handle_task_stderr(self) -> None:
        self._append_process_stderr(self.task_process)

    def _handle_run_finished(self, exit_code: int, _status) -> None:
        self._handle_run_output()
        run_remainder = bytes(self.run_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        self._append_terminal_output(run_remainder)
        for raw_line in run_remainder.splitlines():
            self._parse_line(raw_line, source="run")
        self._handle_run_stderr()
        stderr_buffer = self._consume_process_stderr(self.run_process)
        stop_requested = self.runtime_stop_requested
        self.runtime_stop_requested = False
        if exit_code != 0 and stderr_buffer.strip():
            self._emit_process_error(
                process_label="Runtime",
                source="run",
                exit_code=exit_code,
                stderr_buffer=stderr_buffer,
                event_code="system.runtime_process_failed",
                show_dialog=True,
            )
        else:
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=f"Runtime finalizado (exit={exit_code}).",
                message=f"Runtime finalizado (exit={exit_code}).",
                severity="warning" if exit_code else "info",
                event_code="system.runtime_finished",
            )
            self._push_event(event)
        self._sync_runtime_stopped_ui(exit_code=exit_code, stop_requested=stop_requested)

    def _handle_task_finished(self, exit_code: int, _status) -> None:
        self._handle_task_output()
        task_remainder = bytes(self.task_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        self._append_terminal_output(task_remainder)
        for raw_line in task_remainder.splitlines():
            self._parse_line(raw_line, source="task")
        self._handle_task_stderr()
        stderr_buffer = self._consume_process_stderr(self.task_process)
        label = str(self._active_task_context.get("label") or "")
        silent = bool(self._active_task_context.get("silent"))
        self._update_controls()
        if label == "Check Hyperliquid" and exit_code != 0:
            self.health_state.last_healthcheck_at = datetime.now()
            self.health_state.last_check_execution_ok = False
            self._refresh_connectivity_health(emit_event=True, forced_reason="health_check_command_failed")
        if label == "Check Hyperliquid" and self._awaiting_initial_snapshot:
            if self._has_complete_initial_snapshot():
                self._awaiting_initial_snapshot = False
            elif self._initial_check_attempts < 2:
                self._schedule_initial_check(1500)
            else:
                self._awaiting_initial_snapshot = False
        if exit_code != 0 and stderr_buffer.strip():
            self._emit_process_error(
                process_label=label or "Processo auxiliar",
                source="task",
                exit_code=exit_code,
                stderr_buffer=stderr_buffer,
                event_code="system.helper_process_failed",
                show_dialog=not silent,
            )
        elif exit_code != 0 and not silent:
            self._push_event(
                self._new_event(
                    source="launcher",
                    event_type="generic",
                    raw_line=f"Processo auxiliar finalizado com codigo {exit_code}.",
                    message=f"Processo auxiliar finalizado com codigo {exit_code}.",
                    severity="warning",
                    event_code="system.helper_process_finished",
                )
            )
        if self.pending_emergency_close:
            self._execute_pending_kill_switch()

    def _parse_line(self, raw_line: str, source: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        status_payload = _json_marker(line, "Status Hyperliquid: ")
        if status_payload is not None:
            self._apply_status_payload(status_payload, source=source)
            return

        cycle_payload = _json_marker(line, "Ciclo runtime HL | ")
        if cycle_payload is not None:
            self._apply_cycle_payload(cycle_payload, source=source)
            return

        smoke_payload = _json_marker(line, "Smoke test Hyperliquid | ")
        if smoke_payload is not None:
            self._apply_smoke_payload(smoke_payload, source=source)
            return

        close_payload = _json_marker(line, "Fechamento manual Hyperliquid | ")
        if close_payload is not None:
            self._apply_manual_close_payload(close_payload, source=source)
            return

        plain = self._strip_prefix(line)
        if "Iniciando pipeline com comando: run" in plain:
            self.state.state_label = "RODANDO"
            self.state.state_message = "Runtime ativo. Aguardando proximo ciclo."
            self.state.state_style = "wait"
            self.state.last_decision = "Runtime iniciado. Aguardando proximo ciclo."
            self._update_controls()
            self._refresh_dashboard()

        event = self._new_event(
            source=source,
            event_type="generic",
            raw_line=line,
            message=plain,
            event_code="system.raw_log",
        )
        self._push_event(event)

    def _ensure_daily_rollover(self) -> None:
        today = date.today()
        if self.stats_day == today:
            return
        self.stats_day = today
        self.state.operations_today_label = "0"
        self.state.blocked_today_label = "0"
        self.state.last_trade_reason_label = "Nenhuma operação hoje."
        self.state.last_skip_reason_label = "Nenhum bloqueio recente."

    def _increment_counter_label(self, current_value: str) -> str:
        try:
            return str(int(current_value) + 1)
        except (TypeError, ValueError):
            return "1"

    def _apply_status_payload(self, payload: dict[str, Any], *, source: str) -> None:
        check_at = datetime.now()
        connected = bool(payload.get("connected"))
        can_trade = bool(payload.get("can_trade"))
        health_ok = connected and can_trade
        self.health_state.last_healthcheck_at = check_at
        self.health_state.last_check_execution_ok = True
        self.health_state.connection_ok = health_ok
        if health_ok:
            self.health_state.last_successful_healthcheck_at = check_at
        self.state.online = health_ok
        self.state.can_trade = can_trade
        self.state.network_label = str(payload.get("network", self._current_mode().label)).upper()
        self.state.balance_label = _currency(payload.get("available_to_trade", 0.0))
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        self.state.last_update_label = check_at.strftime("%H:%M:%S")
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)
        runtime_running = self.run_process.state() != QProcess.NotRunning
        if connected and can_trade and not runtime_running:
            self.state.state_label = "PARADO"
            self.state.state_message = "Conexão validada, mas o bot está parado."
            self.state.state_style = "blocked"
            if self.state.position_label in {"LONG", "SHORT"}:
                self.state.position_status = f"Bot parado; revisar posição {self.state.position_label} na exchange."
            else:
                self.state.position_status = "Bot parado. Nenhum runtime ativo."
        self._refresh_dashboard()
        self._refresh_connectivity_health(
            emit_event=True,
            forced_reason="check_hyperliquid_ok" if health_ok else "check_hyperliquid_failed",
        )
        if self._has_complete_initial_snapshot():
            self._awaiting_initial_snapshot = False

        signature = (
            payload.get("connected"),
            payload.get("can_trade"),
            round(float(payload.get("available_to_trade", 0.0) or 0.0), 2),
            payload.get("network"),
        )
        should_notify = not bool(self._active_task_context.get("silent")) or signature != self.last_connection_signature
        self.last_connection_signature = signature
        if should_notify:
            event = self._new_event(
                source=source,
                event_type="status",
                raw_line=json.dumps(payload, ensure_ascii=False),
                payload=payload,
                severity="info" if connected and can_trade else "warning" if connected else "error",
                event_code="healthcheck.status",
            )
            self._push_event(event)

    def _apply_smoke_payload(self, payload: dict[str, Any], *, source: str) -> None:
        event = self._new_event(
            source=source,
            event_type="smoke",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            severity="execution" if payload.get("ok") else "error",
            event_code="system.smoke_test",
        )
        self._push_event(event)

    def _apply_manual_close_payload(self, payload: dict[str, Any], *, source: str) -> None:
        if payload.get("ok"):
            self.state.position_label = "FLAT"
            self.state.position_status = "Sem posição aberta"
            self.state.position_size_label = "--"
            self.state.entry_label = "--"
            self.state.take_profit_label = "--"
            self.state.stop_loss_label = "--"
            self.state.pnl_label = "--"
            self.state.pnl_style = "muted"
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Posição encerrada manualmente."
            self.state.state_style = "wait"
            self.state.position_status = "Sem sinal ativo"
            self._refresh_dashboard()

        event = self._new_event(
            source=source,
            event_type="manual_close",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            severity="risk" if payload.get("ok") else "warning",
            event_code="risk.manual_close",
        )
        self._push_event(event)

    def _push_event(self, event: LauncherEvent, *, prebuilt: HumanizedEvent | None = None) -> None:
        summary = prebuilt or self.log_translator.local.summarize(event)
        if summary.relevant:
            widget, is_new = self._upsert_notification(summary)
            if is_new:
                if self.current_page != "events":
                    self.unread_events += 1
                    self._update_navigation_state()
                    self._show_event_toast(summary)
                self._update_events_empty_state()
            if self.log_translator.enabled:
                task = TranslationTask(self.log_translator, event, summary)
                task.signals.finished.connect(lambda result, target=widget: self._apply_translated_notification(target, result))
                self._thread_pool.start(task)

        if summary.severity == "error":
            self.state.state_label = "ERRO"
            self.state.state_message = summary.message_human
            self.state.state_style = "error"
            self._refresh_dashboard()

    def _upsert_notification(self, summary: HumanizedEvent) -> tuple[NotificationItemWidget, bool]:
        existing = self._notification_widgets.get(summary.fingerprint)
        if existing is not None:
            existing.bump(summary)
            self.notifications_layout.removeWidget(existing)
            self.notifications_layout.insertWidget(0, existing)
            return existing, False

        widget = NotificationItemWidget(summary)
        self._notification_widgets[summary.fingerprint] = widget
        self._notification_order.insert(0, summary.fingerprint)
        self.notifications_layout.insertWidget(0, widget)

        self._trim_notifications()
        return widget, True

    def _notification_limit(self) -> int:
        return max(12, int(self.cfg.launcher.notification_limit))

    def _trim_notifications(self) -> None:
        limit = self._notification_limit()
        while len(self._notification_order) > limit:
            stale_fingerprint = self._notification_order.pop()
            stale = self._notification_widgets.pop(stale_fingerprint, None)
            if stale is not None:
                self.notifications_layout.removeWidget(stale)
                stale.deleteLater()
        self._update_events_empty_state()

    def _update_events_empty_state(self) -> None:
        has_events = bool(self._notification_order)
        self.events_empty_label.setVisible(not has_events)

    def _position_toast(self, toast: EventToast) -> None:
        try:
            toast.adjustSize()
            x = max(18, self.width() - toast.width() - 28)
            y = max(18, self.height() - toast.height() - 32)
            toast.move(x, y)
        except RuntimeError:
            if toast is self._active_toast:
                self._active_toast = None

    def _active_toast_visible(self) -> bool:
        if self._active_toast is None:
            return False
        try:
            return self._active_toast.isVisible()
        except RuntimeError:
            self._active_toast = None
            return False

    def _show_event_toast(self, summary: HumanizedEvent) -> None:
        if self._active_toast_visible():
            try:
                self._active_toast.close()
            except RuntimeError:
                self._active_toast = None
        self._active_toast = EventToast(summary, self)
        self._position_toast(self._active_toast)
        self._active_toast.show()
        self._active_toast.raise_()

    def _apply_translated_notification(self, widget: NotificationItemWidget, result: HumanizedEvent) -> None:
        try:
            if widget is None:
                return
            widget.apply_summary(result)
            self._apply_humanized_cycle_details(result)
        except RuntimeError:
            # O widget C++ ja foi destruido pelo _trim_notifications antes da IA terminar a traducao.
            # Ignoramos silenciosamente para evitar crash do PySide6.
            pass

    def _apply_humanized_cycle_details(self, result: HumanizedEvent) -> None:
        if result.event_type != "cycle":
            return
        if not result.fingerprint or result.fingerprint != self.state.last_cycle_fingerprint:
            return
        if self.run_process.state() == QProcess.NotRunning:
            return

        updated = False
        for state_key, value in (
            ("humanized_market_state", result.market_state),
            ("humanized_model_interpretation", result.model_interpretation),
            ("humanized_filters_diagnostic", result.filters_diagnostic),
            ("humanized_execution_summary", result.execution_summary),
            ("humanized_simple_summary", result.simple_summary),
        ):
            normalized = str(value or "").strip()
            if not normalized or getattr(self.state, state_key) == normalized:
                continue
            setattr(self.state, state_key, normalized)
            updated = True

        simple_summary = str(result.simple_summary or "").strip()
        if simple_summary:
            if self.state.last_decision != simple_summary:
                self.state.last_decision = simple_summary
                updated = True
            if self.state.state_message != simple_summary:
                self.state.state_message = simple_summary
                updated = True

        if updated:
            self._refresh_dashboard()

    def _apply_health_status(self, status: str, reason: str, *, emit_event: bool = False) -> None:
        previous_status = self.health_state.status
        self.health_state.status = status
        self.health_state.reason = reason
        self.state.online = status == "online"
        text = status
        self.health_chip.setText(text)
        _set_widget_property(self.health_chip, "state", text)
        if emit_event and previous_status != status:
            self._push_event(
                self._new_event(
                    source="launcher",
                    event_type="health",
                    raw_line=f"health_status_change | status={status} | reason={reason}",
                    payload={
                        "status": status,
                        "reason": reason,
                        "bot_running": self.health_state.bot_running,
                        "connection_ok": self.health_state.connection_ok,
                        "executor_alive": self.health_state.executor_alive,
                        "last_healthcheck_at": self.health_state.last_healthcheck_at.isoformat() if self.health_state.last_healthcheck_at else None,
                        "last_successful_healthcheck_at": self.health_state.last_successful_healthcheck_at.isoformat() if self.health_state.last_successful_healthcheck_at else None,
                    },
                    message=f"health_status_change | status={status} | reason={reason}",
                    severity="info" if status == "online" else "warning" if status == "warning" else "error",
                    event_code=f"health.{reason}",
                )
            )

    def _derive_health_status(self, *, forced_reason: str | None = None) -> tuple[str, str]:
        state = self.health_state
        threshold = max(30, int(self.cfg.launcher.auto_check_interval_seconds) * 2)

        if state.last_healthcheck_at is None:
            return "offline", forced_reason or "awaiting_first_healthcheck"

        age_seconds = (datetime.now() - state.last_healthcheck_at).total_seconds()

        if state.last_check_execution_ok is False and age_seconds <= threshold:
            return "offline", forced_reason or "health_check_command_failed"

        if age_seconds > threshold:
            return "offline", forced_reason or "health_check_stale"

        if state.connection_ok:
            return "online", forced_reason if forced_reason == "check_hyperliquid_ok" else "connection_healthy"

        return "offline", forced_reason if forced_reason == "check_hyperliquid_failed" else "connection_check_failed"

    def _refresh_connectivity_health(self, *, emit_event: bool = False, forced_reason: str | None = None) -> None:
        status, reason = self._derive_health_status(forced_reason=forced_reason)
        self._apply_health_status(status, reason, emit_event=emit_event)

    def _snapshot_dict(self, payload: dict[str, Any], key: str) -> dict[str, Any]:
        snapshot = payload.get(key)
        return dict(snapshot) if isinstance(snapshot, dict) else {}

    def _market_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._snapshot_dict(payload, "market_snapshot")
        if snapshot:
            return snapshot
        return {
            "bar_timestamp": payload.get("bar_timestamp"),
            "open": payload.get("candle_open"),
            "high": payload.get("candle_high"),
            "low": payload.get("candle_low"),
            "close": payload.get("candle_close"),
            "volume": payload.get("candle_volume"),
            "reference_price": payload.get("reference_price"),
        }

    def _feature_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._snapshot_dict(payload, "feature_snapshot")
        if snapshot:
            return snapshot
        return {
            "rsi_14": payload.get("rsi_14"),
            "dist_ema_240": payload.get("dist_ema_240", payload.get("regime_dist_ema_240")),
            "dist_ema_960": payload.get("dist_ema_960"),
            "atr_pct": payload.get("atr_pct"),
            "volatility_24": payload.get("volatility_24"),
            "vol_regime_z": payload.get("vol_regime_z", payload.get("regime_vol_regime_z")),
            "range_pct": payload.get("range_pct"),
            "range_compression_10": payload.get("range_compression_10"),
            "range_expansion_3": payload.get("range_expansion_3"),
            "breakout_up_10": payload.get("breakout_up_10"),
            "breakout_down_10": payload.get("breakout_down_10"),
            "volume_zscore_20": payload.get("volume_zscore_20"),
            "volume_ratio_20": payload.get("volume_ratio_20"),
        }

    def _decision_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._snapshot_dict(payload, "decision")
        if snapshot:
            return snapshot
        final_action = float(payload.get("final_action", 0.0) or 0.0)
        return {
            "decision_mode": payload.get("decision_mode"),
            "vote_bucket": payload.get("vote_bucket"),
            "votes": payload.get("votes"),
            "raw_actions": payload.get("raw_actions"),
            "reason": payload.get("decision_reason"),
            "final_action": final_action,
            "tie_hold": payload.get("tie_hold"),
            "confidence_pct": abs(final_action) * 100.0,
        }

    def _filter_snapshot_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._snapshot_dict(payload, "filters")
        if snapshot:
            return snapshot
        blocked_reason = payload.get("blocked_reason")
        return {
            "regime_valid_for_entry": payload.get("regime_valid"),
            "regime_dist_ema_240": payload.get("regime_dist_ema_240"),
            "regime_vol_regime_z": payload.get("regime_vol_regime_z"),
            "regime_thresholds": {
                "min_abs_dist_ema_240": float(self.cfg.environment.regime_min_abs_dist_ema_240),
                "min_vol_regime_z": float(self.cfg.environment.regime_min_vol_regime_z),
            },
            "blocked_reason": blocked_reason,
            "blocked_reason_human": self._human_block_label(str(blocked_reason)) if blocked_reason else None,
        }

    def _apply_cycle_payload(self, payload: dict[str, Any], *, source: str) -> None:
        self._ensure_daily_rollover()
        self.health_state.bot_running = True
        self.health_state.executor_alive = True
        self.state.last_update_label = datetime.now().strftime("%H:%M:%S")

        market_snapshot = self._market_snapshot_from_payload(payload)
        feature_snapshot = self._feature_snapshot_from_payload(payload)
        decision_snapshot = self._decision_snapshot_from_payload(payload)
        filter_snapshot = self._filter_snapshot_from_payload(payload)

        self.state.market_snapshot = market_snapshot
        self.state.feature_snapshot = feature_snapshot
        self.state.decision_snapshot = decision_snapshot
        self.state.filter_snapshot = filter_snapshot

        final_action = decision_snapshot.get("final_action", payload.get("final_action", 0.0))
        vote_bucket = decision_snapshot.get("vote_bucket", payload.get("vote_bucket", "--"))
        blocked_reason = filter_snapshot.get("blocked_reason", payload.get("blocked_reason"))

        self.state.reference_price = _optional_price(
            market_snapshot.get("reference_price", payload.get("reference_price", 0.0))
        )
        self.state.risk_label = _pct(payload.get("dynamic_risk_pct", payload.get("risk_pct", 0.0)))
        self.state.strength_label = _strength(final_action)
        self.state.regime_label = "valido" if filter_snapshot.get("regime_valid_for_entry") else "bloqueado"
        self.state.signal_label = self._signal_label(float(final_action or 0.0))
        self.state.direction_label = str(vote_bucket or "--").upper()
        self.state.balance_label = _currency(payload.get("available_to_trade", 0.0))
        self.state.blocker_label = self._human_block_label(str(blocked_reason)) if blocked_reason else "Sem bloqueio"
        self.state.context_summary_label = HUMANIZED_MARKET_PLACEHOLDER
        self.state.feature_summary_label = HUMANIZED_MODEL_PLACEHOLDER
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)
        self.state.humanized_market_state = ""
        self.state.humanized_model_interpretation = ""
        self.state.humanized_filters_diagnostic = ""
        self.state.humanized_execution_summary = ""
        self.state.humanized_simple_summary = ""
        self.state.last_cycle_fingerprint = ""

        position_label = str(payload.get("position_label", "FLAT")).upper()
        self.state.position_label = position_label
        self.state.position_size_label = _currency(payload.get("adjusted_notional", payload.get("notional_value", 0.0)))
        self.state.entry_label = _price(payload.get("position_avg_entry_price", 0.0))
        self.state.take_profit_label = _price(payload.get("take_profit_price", 0.0))
        self.state.stop_loss_label = _price(payload.get("stop_loss_price", 0.0))
        pnl_value = float(payload.get("position_unrealized_pnl", 0.0) or 0.0)
        self.state.pnl_label = _currency(pnl_value)
        self.state.pnl_style = "success" if pnl_value >= 0 else "error"

        if payload.get("error"):
            self.state.state_label = "ERRO"
            self.state.state_message = "Execucao com erro. Revisar operacao."
            self.state.state_style = "error"
        elif blocked_reason:
            self.state.state_label = "BLOQUEADO"
            self.state.state_message = self._human_block_label(str(blocked_reason))
            self.state.state_style = "blocked"
        elif payload.get("position_is_open"):
            self.state.state_label = position_label
            self.state.state_message = f"Posicao {position_label} em andamento."
            self.state.state_style = "long" if position_label == "LONG" else "short"
        elif self.state.signal_label == "HOLD":
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Sem entrada nesta barra."
            self.state.state_style = "wait"
        else:
            self.state.state_label = "FLAT"
            self.state.state_message = "Sem posicao aberta."
            self.state.state_style = "flat"

        if payload.get("position_is_open"):
            self.state.position_status = f"{position_label} aberta no momento"
        elif blocked_reason:
            self.state.position_status = "Sem posicao; entrada bloqueada"
        else:
            self.state.position_status = "Sem sinal ativo"

        self._refresh_dashboard()

        event = self._new_event(
            source=source,
            event_type="cycle",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            severity="execution",
            event_code="execution.cycle",
        )
        summary = self.log_translator.local.summarize(event)
        summary_text = str(summary.simple_summary or "").strip() or summary.message_human
        if payload.get("opened_trade"):
            self.state.operations_today_label = self._increment_counter_label(self.state.operations_today_label)
            self.state.last_trade_reason_label = summary.message_human
        elif payload.get("closed_trade"):
            self.state.last_trade_reason_label = summary.message_human
        elif blocked_reason:
            self.state.blocked_today_label = self._increment_counter_label(self.state.blocked_today_label)
            self.state.last_skip_reason_label = summary.message_human
        else:
            self.state.last_skip_reason_label = summary.message_human

        self.state.last_decision = summary_text
        self.state.state_message = summary_text
        self.state.last_cycle_fingerprint = summary.fingerprint
        self._refresh_dashboard()
        self._push_event(event, prebuilt=summary)

    def _refresh_dashboard(self) -> None:
        self.state.symbol_label = str(self.cfg.hyperliquid.symbol)
        self.state.timeframe_label = str(self.cfg.hyperliquid.timeframe)
        self.state.execution_mode_label = str(self._current_mode().execution_mode)
        self.state.connection_label = self.health_state.status
        self.state.last_valid_check_label = (
            self.health_state.last_successful_healthcheck_at.strftime("%H:%M:%S")
            if self.health_state.last_successful_healthcheck_at is not None
            else "--"
        )
        self.mode_badge.setText(self.state.network_label)
        self.state_pill.setText(self.state.state_label)
        _set_widget_property(self.state_pill, "status", self.state.state_style)
        dashboard_summary = _display_value(self.state.humanized_simple_summary, self.state.state_message)
        self.state_message.setText(dashboard_summary)
        self.balance_inline.setText(self.state.balance_label)
        _fit_label_height(self.state_message)
        _fit_label_height(self.balance_inline)
        self.state_message.updateGeometry()

        self.signal_tile.update_tile(_display_value(self.state.signal_label, "nenhum"), tone="default")
        reason_value = _display_value(
            self.state.humanized_execution_summary,
            self.state.blocker_label if self.state.blocker_label != "Sem bloqueio" else dashboard_summary,
        )
        self.reason_tile.update_tile(_display_value(reason_value, "indefinido"), tone="default")
        self.price_tile.update_tile(self.state.reference_price, tone="default")
        self.risk_tile.update_tile(self.state.risk_label, tone="warning")
        self.position_tile.update_tile(_display_value(self.state.position_label, "FLAT"), tone="default")
        self.pnl_tile.update_tile(self.state.pnl_label, tone=self.state.pnl_style)

        self.position_pill.setText(self.state.position_label)
        position_style = "long" if self.state.position_label == "LONG" else "short" if self.state.position_label == "SHORT" else "flat"
        _set_widget_property(self.position_pill, "status", position_style)
        self.position_status.setText(self.state.position_status)
        _fit_label_height(self.position_status)
        self.position_status.updateGeometry()
        self.pnl_value.setText(self.state.pnl_label)
        _fit_label_height(self.pnl_value)
        _set_widget_property(self.pnl_value, "tone", self.state.pnl_style)
        self.position_size_value.setText(self.state.position_size_label)
        _fit_label_height(self.position_size_value)
        self.entry_value.setText(self.state.entry_label)
        _fit_label_height(self.entry_value)
        self.tp_value.setText(self.state.take_profit_label)
        _fit_label_height(self.tp_value)
        self.sl_value.setText(self.state.stop_loss_label)
        _fit_label_height(self.sl_value)

        self.connection_tile.update_tile(_display_value(self.state.connection_label, "offline"), tone="default")
        self.heartbeat_tile.update_tile(_display_value(self.state.last_valid_check_label, "--"), tone="default")
        self.operations_today_tile.update_tile(_display_value(self.state.operations_today_label, "0"), tone="default")
        self.blocked_today_tile.update_tile(_display_value(self.state.blocked_today_label, "0"), tone="default")

        def set_detail_block(value_label: QLabel, value_text: str, note_label: QLabel, note_text: str | None = None) -> None:
            value_label.setText(value_text)
            _fit_label_height(value_label)
            if note_text:
                note_label.setText(note_text)
                _fit_label_height(note_label)
                note_label.show()
            else:
                note_label.hide()

        current_reason = self.state.blocker_label if self.state.blocker_label != "Sem bloqueio" else dashboard_summary
        trade_fallback = self.state.last_trade_reason_label
        if trade_fallback.startswith("Nenhuma "):
            trade_fallback = HUMANIZED_EXECUTION_PLACEHOLDER
        filters_fallback = self.state.last_skip_reason_label
        if filters_fallback == "Nenhum bloqueio recente.":
            filters_fallback = HUMANIZED_FILTERS_PLACEHOLDER
        detail_simple_summary = _display_value(self.state.humanized_simple_summary, self.state.last_decision)
        detail_execution_summary = _display_value(self.state.humanized_execution_summary, current_reason)
        detail_market_state = _display_value(self.state.humanized_market_state, HUMANIZED_MARKET_PLACEHOLDER)
        detail_model_interpretation = _display_value(
            self.state.humanized_model_interpretation,
            HUMANIZED_MODEL_PLACEHOLDER,
        )
        detail_filters_diagnostic = _display_value(
            self.state.humanized_filters_diagnostic,
            filters_fallback,
        )
        detail_trade_summary = _display_value(self.state.humanized_execution_summary, trade_fallback)

        self.details_decision_label.setText(detail_simple_summary)
        _fit_label_height(self.details_decision_label)
        self.details_decision_reason_label.setText(detail_execution_summary)
        _fit_label_height(self.details_decision_reason_label)
        self.details_decision_meta_label.setText(
            f"Sinal {self.state.signal_label} | Direcao {self.state.direction_label} | Regime {self.state.regime_label} | Posicao {self.state.position_label}"
        )
        _fit_label_height(self.details_decision_meta_label)

        self.indicators_card.set_summary(detail_market_state)
        self.indicators_card.set_note(None)
        self.intelligence_card.set_summary(detail_model_interpretation)
        self.intelligence_card.set_note(None)
        self.filters_card.set_summary(detail_filters_diagnostic)
        self.filters_card.set_note(None)

        set_detail_block(
            self.details_context_value,
            detail_market_state,
            self.details_context_note,
            f"Preco de referencia {self.state.reference_price} | Risco {self.state.risk_label}",
        )
        set_detail_block(
            self.details_trade_value,
            detail_trade_summary,
            self.details_trade_note,
            f"Posicao atual {self.state.position_label} | Tamanho {self.state.position_size_label}",
        )
        set_detail_block(
            self.details_skip_value,
            detail_filters_diagnostic,
            self.details_skip_note,
            f"Bloqueio atual: {self.state.blocker_label}",
        )
        set_detail_block(
            self.details_features_value,
            detail_model_interpretation,
            self.details_features_note,
            f"Entrada {self.state.entry_label} | TP {self.state.take_profit_label} | SL {self.state.stop_loss_label}",
        )

    def _update_navigation_state(self) -> None:
        labels = {
            "dashboard": "Dashboard",
            "details": "Detalhes",
            "events": "Eventos",
            "terminal": "Terminal",
        }
        for key, button in self.page_buttons.items():
            button.setChecked(key == self.current_page)
            unread = key == "events" and self.unread_events > 0
            button.setText(f"{labels[key]} \u2022" if unread else labels[key])
            _set_widget_property(button, "unread", unread)

    def _signal_label(self, final_action: float) -> str:
        threshold = float(self.cfg.environment.action_hold_threshold)
        if final_action >= threshold:
            return "BUY"
        if final_action <= -threshold:
            return "SELL"
        return "HOLD"

    def _human_block_label(self, reason: str) -> str:
        mapping = {
            "regime_filter": "Entrada bloqueada pelo filtro de regime",
            "cooldown": "Entrada bloqueada pelo cooldown",
            "force_exit_only_by_tp_sl": "Sinal ignorado: posição travada por TP/SL",
            "min_notional_risk": "Entrada pulada por risco mínimo x notional",
            "exchange_position_present": "Entrada ignorada: posição já existe",
            "duplicate_cycle_order": "Entrada ignorada para evitar ordem duplicada",
        }
        return mapping.get(reason, reason.replace("_", " "))

    def _strip_prefix(self, line: str) -> str:
        if " | " not in line:
            return line
        parts = line.split(" | ", 3)
        if len(parts) < 4:
            return line
        return parts[3].strip()


def _run_launcher_with_pid_lock(app: QApplication) -> int:
    pid_file = REPO_ROOT / "launcher.pid"
    global _launcher_lock
    _launcher_lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Tenta travar a porta para garantir que e a unica instancia
        _launcher_lock.bind(("127.0.0.1", 54321))
        # Se conseguiu, salva o PID atual no arquivo
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    except socket.error:
        # Se a porta falhou, existe um zumbi. Vamos tentar ler o PID dele.
        old_pid = None
        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
            except ValueError:
                pass

        msg = "O TraderBot Launcher ja esta rodando (provavelmente preso invisivel em segundo plano).\n\n"
        if old_pid:
            msg += f"Deseja forcar o encerramento da instancia fantasma (PID: {old_pid}) para liberar o sistema?"
        else:
            msg += "Nao encontrei o PID para matar automaticamente. Voce tera que fechar o 'pythonw.exe' pelo Gerenciador de Tarefas do Windows."

        answer = QMessageBox.warning(
            None,
            "Instancia Fantasma Detectada",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if answer == QMessageBox.Yes and old_pid:
            try:
                if os.name == "nt":  # Se for Windows
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(old_pid)],
                        check=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:  # Se for Linux/Mac
                    os.kill(old_pid, signal.SIGKILL)
                QMessageBox.information(
                    None,
                    "Resolvido",
                    "Instancia fantasma encerrada com sucesso!\nVoce ja pode abrir o Launcher normalmente.",
                )
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "Erro",
                    f"Nao foi possivel matar o processo fantasma.\nDetalhes: {e}\n\nAbra o Gerenciador de Tarefas (Ctrl+Shift+Esc), procure por 'pythonw.exe' e finalize a tarefa manualmente.",
                )
        return 1

    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    if LAUNCHER_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(LAUNCHER_ICON_PATH)))
    theme_path = REPO_ROOT / "theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    window = TraderBotLauncher()
    window.show()
    try:
        return app.exec()
    finally:
        try:
            _launcher_lock.close()
        except Exception:
            pass
        try:
            if pid_file.exists():
                pid_file.unlink()
        except OSError:
            pass


def main() -> int:
    _apply_windows_app_identity()
    app = QApplication(sys.argv)
    app.setApplicationName("Trader Laucher")
    app.setApplicationDisplayName("Trader Laucher")
    return _run_launcher_with_pid_lock(app)


if __name__ == "__main__":
    raise SystemExit(main())
