from __future__ import annotations
from traderbot.utils.logger import setup_logger
from traderbot.utils.telegram_notifier import TelegramNotifier

import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import traceback
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from traderbot.config import AppConfig, EnvironmentConfig, load_config
from traderbot.launcher_services import HumanizedEvent, LauncherEvent, is_launcher_silenced_cycle_issue
from traderbot.runtime_guard import read_runtime_state, runtime_state_path_for


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LAUNCHER_ICON_PATH = REPO_ROOT / "icon" / "icon.ico"
WINDOWS_APP_ID = "traderbot.launcher.desktop"

def _currency(value: Any, prefix: str = "$") -> str:
    try:
        return f"{prefix}{float(value):,.2f}"
    except (TypeError, ValueError):
        return f"{prefix}0.00"


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


def _json_marker(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _apply_windows_app_identity() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass


def _apply_windows_taskbar_icon(window: QWidget) -> None:
    if os.name != "nt" or not LAUNCHER_ICON_PATH.exists():
        return

    try:
        hwnd = int(window.winId())
        if hwnd <= 0:
            return

        image_icon = 1
        wm_seticon = 0x0080
        icon_small = 0
        icon_big = 1
        lr_loadfromfile = 0x0010
        lr_defaultsize = 0x0040
        gclp_hicon = -14
        gclp_hiconsm = -34

        load_image = ctypes.windll.user32.LoadImageW
        send_message = ctypes.windll.user32.SendMessageW
        set_class_long_ptr = getattr(ctypes.windll.user32, "SetClassLongPtrW", None)
        if set_class_long_ptr is None:
            set_class_long_ptr = ctypes.windll.user32.SetClassLongW

        big_icon = load_image(
            None,
            str(LAUNCHER_ICON_PATH),
            image_icon,
            0,
            0,
            lr_loadfromfile | lr_defaultsize,
        )
        small_icon = load_image(
            None,
            str(LAUNCHER_ICON_PATH),
            image_icon,
            16,
            16,
            lr_loadfromfile,
        )

        if big_icon:
            send_message(hwnd, wm_seticon, icon_big, big_icon)
            set_class_long_ptr(hwnd, gclp_hicon, big_icon)
        if small_icon:
            send_message(hwnd, wm_seticon, icon_small, small_icon)
            set_class_long_ptr(hwnd, gclp_hiconsm, small_icon)

        setattr(window, "_windows_taskbar_icons", (big_icon, small_icon))
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
    state_label: str = "CARREGANDO..."
    state_message: str = "Carregando Informações..."
    state_style: str = "wait"
    regime_label: str = "indefinido"
    signal_label: str = "nenhum"
    direction_label: str = "nenhuma"
    strength_label: str = "--"
    reference_currency: str = "--"
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
    wins_today: int = 0 
    losses_today: int = 0
    blocked_today_label: str = "0"
    market_status_label: str = "Aguardando..."
    drawdown_label: str = "--"
    ai_bias_label: str = "--"
    ensemble_score_label: str = "--"
    humanized_execution_summary: str = ""
    humanized_simple_summary: str = ""
    last_cycle_fingerprint: str = ""
    position_label: str = "FLAT"
    position_status: str = "Nenhuma Ordem Aberta"
    position_size_label: str = "--"
    entry_label: str = "--"
    take_profit_label: str = "--"
    stop_loss_label: str = "--"
    native_tp_sl_status: dict[str, Any] = field(default_factory=dict)
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
    last_runtime_state_at: datetime | None = None
    runtime_state_ok: bool = False
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


def _configure_readonly_browser(browser: QTextBrowser) -> None:
    browser.setReadOnly(True)
    browser.setAcceptRichText(True)
    browser.setOpenExternalLinks(False)
    browser.setFrameShape(QFrame.NoFrame)
    browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    browser.document().setDocumentMargin(0)
    browser.setStyleSheet("background: transparent; background-color: transparent; border: none;")
    browser.viewport().setStyleSheet("background: transparent; background-color: transparent; border: none;")


def _set_browser_content(browser: QTextBrowser, text: str) -> None:
    normalized = str(text or "--").strip() or "--"
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json|html|markdown|md|text)?\s*|\s*```$", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip() or "--"
    if re.search(r"<[a-zA-Z][^>]*>", normalized):
        browser.setHtml(normalized)
        return
    browser.setPlainText(normalized)


class MetricTile(QFrame):
    def __init__(
        self,
        label: str,
        *,
        compact: bool = False,
        wrap_value: bool = False,
        min_height: int = 120,
    ):
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.setMinimumHeight(min_height)
        self._compact = compact
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

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
    def __init__(self, title: str, *, min_height: int = 224, rich_summary: bool = False):
        super().__init__()
        self.setObjectName("PositionCard")
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self._rich_summary = rich_summary
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("fieldLabel")
        layout.addWidget(title_label)

        if rich_summary:
            self.summary = QTextBrowser()
            self.summary.setObjectName("metricValue")
            self.summary.setProperty("sizeVariant", "body")
            _configure_readonly_browser(self.summary)
            _set_browser_content(self.summary, "--")
        else:
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
        if self._rich_summary:
            _set_browser_content(self.summary, text)
            return
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

        raw_header = QLabel("Log técnico")
        raw_header.setObjectName("fieldLabel")
        root.addWidget(raw_header)

        raw_box = QPlainTextEdit()
        raw_box.setObjectName("technicalLog")
        raw_box.setReadOnly(True)
        raw_box.setPlainText(self.summary.message_raw or "Sem log técnico disponível.")
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
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.dot_label = QLabel("●")
        self.dot_label.setObjectName("severityDot")
        top.addWidget(self.dot_label, 0, Qt.AlignLeft)

        self.severity_badge = QLabel("INFO")
        self.severity_badge.setObjectName("badge")
        top.addWidget(self.severity_badge, 0, Qt.AlignLeft)

        # Espaçamento flexível para empurrar o tempo para a direita
        top.addStretch(1)

        self.time_label = QLabel("--:--:--")
        self.time_label.setObjectName("metaLabel")
        top.addWidget(self.time_label, 0, Qt.AlignRight)

        self.count_label = QLabel("")
        self.count_label.setObjectName("metaLabel")
        self.count_label.hide()
        top.addWidget(self.count_label, 0, Qt.AlignRight)

        layout.addLayout(top)

        self.message_label = QLabel("")
        self.message_label.setObjectName("notificationMessage")
        _configure_dynamic_label(self.message_label)
        layout.addWidget(self.message_label)
        self.setToolTip("Ver log tecnico")

    def apply_summary(self, summary: HumanizedEvent) -> None:
        self.summary = summary
        self.dot_label.setText("●")
        self.severity_badge.setText(summary.severity.upper())
        self.message_label.setText(summary.message_human)
        self.time_label.setText(summary.timestamp.strftime("%H:%M:%S"))
        
        _set_widget_property(self, "severity", summary.severity)
        _set_widget_property(self.severity_badge, "severity", summary.severity)
        _set_widget_property(self.dot_label, "severity", summary.severity)
        
        self._refresh_count()
        
        # Ajusta a altura da mensagem para caber o texto
        _fit_label_height(self.message_label)
        
        # Força o card a recalcular seu tamanho total baseado no conteúdo novo
        self.updateGeometry()

    def bump(self, summary: HumanizedEvent) -> None:
        self.count += 1
        self.apply_summary(summary)

    def _refresh_count(self) -> None:
        if self.count <= 1:
            self.count_label.hide()
            return
        self.count_label.setText(f"x{self.count}")
        self.count_label.show()
        self.count_label.adjustSize()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        dialog = EventDetailsDialog(self.summary, parent=self.window())
        dialog.exec()

class SettingsDialog(QDialog):
    RESTART_REQUESTED = 1001

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
            "smoke_side": "buy",
            "smoke_wait_seconds": 3.0,
        }
        self.setModal(True)
        self.setWindowTitle("Configurações")
        self.resize(700, 620)
        self.setMinimumSize(620, 520)
        self._build_ui(smoke_side=smoke_side, smoke_wait_seconds=smoke_wait_seconds)
        self._initial_state = self._current_state()

    def _build_ui(self, smoke_side: str, smoke_wait_seconds: float) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        title = QLabel("Configurações rápidas")
        self.setObjectName("dialogSurface")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        hint = QLabel("Ajustes de filtros, risco e comportamento do bot. Todos os controles rápidos do launcher ficam disponíveis aqui.")
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

        scroll = QScrollArea()
        scroll.setObjectName("SettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("SettingsScrollContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

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
        form.addWidget(self._field_label("Smoke side"), 5, 0)
        form.addWidget(self.smoke_side_button, 5, 1)
        form.addWidget(self._field_label("Wait smoke"), 6, 0)
        form.addWidget(self.smoke_wait_input, 6, 1)

        content_layout.addLayout(form)
        content_layout.addWidget(self.api_key_hint)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.setSpacing(12)

        left_actions = QHBoxLayout()
        left_actions.setSpacing(10)

        self.reset_button = QPushButton("Restaurar padrões")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        left_actions.addWidget(self.reset_button)

        self.restart_button = QPushButton("Reiniciar")
        self.restart_button.setObjectName("SecondaryButton")
        self.restart_button.clicked.connect(self._request_restart)
        left_actions.addWidget(self.restart_button)

        footer.addLayout(left_actions)
        footer.addStretch(1)

        right_actions = QHBoxLayout()
        right_actions.setSpacing(10)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        right_actions.addWidget(self.cancel_button)

        self.save_button = QPushButton("Salvar")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.accept)
        right_actions.addWidget(self.save_button)

        footer.addLayout(right_actions)
        root.addLayout(footer)

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _toggle_smoke_side(self) -> None:
        self.smoke_side_button.setText("SELL" if self.smoke_side_button.isChecked() else "BUY")

    def _reset_to_defaults(self) -> None:
        self.risk_input.setValue(float(self._dialog_defaults["risk_pct"]))
        self.hold_threshold_input.setValue(float(self._dialog_defaults["hold_threshold"]))
        self.regime_abs_dist_input.setValue(float(self._dialog_defaults["regime_min_abs_dist_ema_240"]))
        self.regime_vol_input.setValue(float(self._dialog_defaults["regime_min_vol_regime_z"]))
        self.autocheck_input.setValue(float(self._dialog_defaults["auto_check_interval"]))
        self.smoke_side_button.setChecked(str(self._dialog_defaults["smoke_side"]).lower() == "sell")
        self._toggle_smoke_side()
        self.smoke_wait_input.setValue(float(self._dialog_defaults["smoke_wait_seconds"]))

    def _current_state(self) -> dict[str, float | str]:
        return {
            "risk_pct": round(float(self.risk_input.value()), 4),
            "hold_threshold": round(float(self.hold_threshold_input.value()), 4),
            "regime_min_abs_dist_ema_240": round(float(self.regime_abs_dist_input.value()), 4),
            "regime_min_vol_regime_z": round(float(self.regime_vol_input.value()), 4),
            "auto_check_interval": round(float(self.autocheck_input.value()), 4),
            "smoke_side": self.smoke_side(),
            "smoke_wait_seconds": round(float(self.smoke_wait_input.value()), 4),
        }

    def has_unsaved_changes(self) -> bool:
        return self._current_state() != self._initial_state

    def _request_restart(self) -> None:
        if self.has_unsaved_changes():
            answer = QMessageBox.warning(
                self,
                "Reiniciar launcher",
                "Existem alterações não salvas nesta janela. Reiniciar agora e descartar essas mudanças?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.done(self.RESTART_REQUESTED)

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
        self._kill_switch_in_progress = False
        self._close_after_kill_requested = False
        self._force_close_armed = False
        self._restart_requested = False
        self._restart_helper_started = False
        self._restart_launch_program: str | None = None
        self._restart_launch_args: list[str] = []
        self._active_task_context: dict[str, Any] = {"label": None, "silent": False}
        self._notification_widgets: dict[str, NotificationItemWidget] = {}
        self._notification_order: list[str] = []
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
        self._last_cycle_identity: str | None = None
        self.notifier = TelegramNotifier(logger=setup_logger("launcher-notifier", self.cfg.paths.logs_dir))
        

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

        self.setWindowTitle("Trader Launcher")
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

        self.runtime_state_timer = QTimer(self)
        self.runtime_state_timer.setInterval(2000)
        self.runtime_state_timer.timeout.connect(self._poll_runtime_state)
        self.runtime_state_timer.start()

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

        self.state_pill = QLabel("CARREGANDO...")
        self.state_pill.setObjectName("StatePill")
        hero_left.addWidget(self.state_pill, 0, Qt.AlignLeft)

        self.state_message = QLabel("Carregando Informações...")
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

        self.market_status_tile = MetricTile("Status do Mercado", compact=True, min_height=118)
        self.drawdown_tile = MetricTile("Drawdown Atual", compact=True, min_height=118)
        self.ai_bias_tile = MetricTile("Viés da IA", compact=True, min_height=118)
        self.ensemble_score_tile = MetricTile("Placar dos Modelos", compact=True, min_height=118)
        self.operations_tile = MetricTile("Operações hoje", compact=True, min_height=118)
        self.winrate_tile = MetricTile("Win Rate (Sessão)", compact=True, min_height=118)

        self.dashboard_metric_tiles = [
            self.market_status_tile,
            self.drawdown_tile,
            self.ai_bias_tile,
            self.ensemble_score_tile,
            self.operations_tile,
            self.winrate_tile,
        ]
        for tile in self.dashboard_metric_tiles:
            tile.setMinimumHeight(104)
        
        layout.addLayout(self.dashboard_metrics_grid)

        position = QFrame()
        position.setObjectName("PositionCard")
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(14, 14, 14, 14)
        position_layout.setSpacing(10)

        position_top = QHBoxLayout()
        position_top.setSpacing(10)

        position_meta = QVBoxLayout()
        position_meta.setSpacing(6)

        self.position_pill = QLabel("FLAT")
        self.position_pill.setObjectName("PositionPill")
        position_meta.addWidget(self.position_pill, 0, Qt.AlignLeft)

        self.position_status = QLabel("Nenhuma Ordem Aberta")
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

        details_host = QWidget()
        details_host.setObjectName("DetailsHost")
        details_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        content_layout = QVBoxLayout(details_host)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(18)

        self.details_metrics_grid = QGridLayout()
        self.details_metrics_grid.setContentsMargins(0, 0, 0, 0)
        self.details_metrics_grid.setHorizontalSpacing(14)
        self.details_metrics_grid.setVerticalSpacing(14)

        self.heartbeat_tile = MetricTile("Último check válido", compact=True, min_height=92)
        self.operations_today_tile = MetricTile("Operações hoje", compact=True, min_height=92)
        self.blocked_today_tile = MetricTile("Bloqueios hoje", compact=True, min_height=92)
        self.details_metric_tiles = [
            self.heartbeat_tile,
            self.operations_today_tile,
            self.blocked_today_tile,
        ]
        content_layout.addLayout(self.details_metrics_grid)

        decision_card = QFrame()
        decision_card.setObjectName("PositionCard")
        decision_layout = QVBoxLayout(decision_card)
        decision_layout.setContentsMargins(18, 16, 18, 16)
        decision_layout.setSpacing(8)

        decision_title = QLabel("Última Decisão do Bot")
        decision_title.setObjectName("fieldLabel")
        decision_layout.addWidget(decision_title)

        self.details_final_action = QLabel("Aguardando próximo ciclo...")
        self.details_final_action.setObjectName("metricValue")
        self.details_final_action.setProperty("sizeVariant", "body")
        _configure_dynamic_label(self.details_final_action)
        decision_layout.addWidget(self.details_final_action)

        self.details_decision_reason = QLabel("--")
        self.details_decision_reason.setObjectName("HeroHint")
        _configure_dynamic_label(self.details_decision_reason)
        decision_layout.addWidget(self.details_decision_reason)

        content_layout.addWidget(decision_card)

        self.details_filters_card = DetailFieldCard("Checks dos filtros", min_height=176, rich_summary=True)
        self.details_filters_card.set_summary("Aguardando proximo ciclo...")
        self.details_filters_card.set_note("Os filtros aparecerão aqui.")
        content_layout.addWidget(self.details_filters_card)

        votes_card = QFrame()
        votes_card.setObjectName("PositionCard")
        votes_layout = QVBoxLayout(votes_card)
        votes_layout.setContentsMargins(18, 16, 18, 16)
        votes_layout.setSpacing(12)

        votes_header = QHBoxLayout()
        votes_title = QLabel("Votos Individuais dos Modelos")
        votes_title.setObjectName("fieldLabel")
        votes_header.addWidget(votes_title)

        self.details_vote_timestamp = QLabel("--:--:--")
        self.details_vote_timestamp.setObjectName("subtleText")
        votes_header.addWidget(self.details_vote_timestamp, 0, Qt.AlignRight)
        votes_layout.addLayout(votes_header)

        self.details_votes_container = QVBoxLayout()
        self.details_votes_container.setSpacing(8)
        votes_layout.addLayout(self.details_votes_container)

        content_layout.addWidget(votes_card)
        content_layout.addStretch(1)

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

        self.clear_terminal_button = QPushButton("Limpar Terminal")
        self.clear_terminal_button.setObjectName("SecondaryButton")
        self.clear_terminal_button.clicked.connect(self._clear_terminal_output)
        header.addWidget(self.clear_terminal_button)

        self.close_other_instances_button = QPushButton("Fechar Instâncias")
        self.close_other_instances_button.setObjectName("SecondaryButton")
        self.close_other_instances_button.clicked.connect(self._close_other_instances)
        header.addWidget(self.close_other_instances_button)
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
        inner.setContentsMargins(12, 10, 12, 10)
        inner.setSpacing(4)

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

        detail_columns = 4 if dashboard_width >= 1280 else 2 if dashboard_width >= 760 else 1
        level_columns = 3 if dashboard_width >= 980 else 1

        self._relayout_grid_items(self.dashboard_metrics_grid, self.dashboard_metric_tiles, metrics_columns)

        if hasattr(self, "details_metrics_grid"):
            self._relayout_grid_items(self.details_metrics_grid, self.details_metric_tiles, detail_columns)

        self._relayout_grid_items(self.levels_grid, self.level_boxes, level_columns)
        if hasattr(self, "notifications_layout"):
            self._trim_notifications()

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
        action_text = "reiniciar" if self._restart_requested else "fechar"

        if self._force_close_armed:
            if self._finalize_launcher_shutdown():
                event.accept()
            else:
                event.ignore()
            return

        if self._kill_switch_in_progress:
            self._close_after_kill_requested = True
            event.ignore()
            return

        if self._close_after_kill_requested:
            event.ignore()
            return

        if bot_running or position_open:
            answer = QMessageBox.warning(
                self,
                "Reiniciar launcher" if self._restart_requested else "Fechar launcher",
                f"Tem certeza que deseja {action_text} o launcher? O bot vai ser parado.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self._clear_restart_request()
                event.ignore()
                return
            if self._finalize_launcher_shutdown():
                event.accept()
            else:
                event.ignore()
            return

        if self._finalize_launcher_shutdown():
            event.accept()
        else:
            event.ignore()

    def _finalize_launcher_shutdown(self) -> bool:
        if self._restart_requested and not self._ensure_restart_helper_started():
            return False

        self.health_timer.stop()
        self.runtime_state_timer.stop()

        self._thread_pool.clear()
        self._thread_pool.waitForDone(1000)

        if self.run_process.state() != QProcess.NotRunning:
            self.run_process.terminate()
            if not self.run_process.waitForFinished(2000):
                self.run_process.kill()
                self.run_process.waitForFinished(1000)

        if self.task_process.state() != QProcess.NotRunning:
            self.task_process.terminate()
            if not self.task_process.waitForFinished(2000):
                self.task_process.kill()
                self.task_process.waitForFinished(1000)
        return True

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
        self.kill_button.setEnabled(not task_busy and not self._kill_switch_in_progress)
        self.run_button.setEnabled(run_busy or not task_busy)
        if hasattr(self, "close_other_instances_button"):
            self.close_other_instances_button.setEnabled(not self._kill_switch_in_progress)
        if run_busy and self.runtime_stop_requested:
            run_label = "PARANDO..."
            run_state = "stopping"
        elif run_busy and self.state.state_label == "INICIANDO":
            run_label = "INICIANDO..."
            run_state = "starting"
        elif run_busy:
            run_label = "BOT ATIVO"
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

    def _runtime_state_path(self) -> Path:
        return runtime_state_path_for(
            network=self._current_mode().network,
            symbol=str(self.cfg.hyperliquid.symbol),
            logs_dir=self.cfg.paths.logs_dir,
        )

    def _runtime_state_stale_after_seconds(self) -> float:
        heartbeat_interval = max(
            5.0,
            float(getattr(self.cfg.execution, "runtime_guard_heartbeat_interval_seconds", 15)),
        )
        return max(20.0, heartbeat_interval * 4.0)

    @staticmethod
    def _cycle_identity(payload: dict[str, Any]) -> str:
        identity_payload = {
            "bar_timestamp": payload.get("bar_timestamp"),
            "event_code": payload.get("event_code"),
            "position_label": payload.get("position_label"),
            "position_is_open": payload.get("position_is_open"),
            "opened_trade": payload.get("opened_trade"),
            "closed_trade": payload.get("closed_trade"),
            "blocked_reason": payload.get("blocked_reason"),
            "manual_protection_required": payload.get("manual_protection_required"),
            "error": payload.get("error"),
            "final_action": payload.get("final_action"),
        }
        return json.dumps(identity_payload, sort_keys=True, ensure_ascii=True)

    def _poll_runtime_state(self) -> None:
        def _consume() -> None:
            payload = read_runtime_state(self._runtime_state_path())
            if payload is None:
                self.health_state.runtime_state_ok = False
                self.health_state.last_runtime_state_at = None
                if self.run_process.state() == QProcess.NotRunning:
                    self.health_state.bot_running = False
                    self.health_state.executor_alive = False
                self._refresh_connectivity_health(emit_event=False)
                return

            self._apply_runtime_state_payload(payload)

        self._guard_launcher_action("leitura do estado compartilhado da engine", _consume)

    def _apply_runtime_state_payload(self, payload: dict[str, Any]) -> None:
        heartbeat_at = _parse_iso_datetime(payload.get("last_heartbeat_at")) or _parse_iso_datetime(
            payload.get("started_at")
        )
        status = str(payload.get("status") or "").strip().lower()
        if heartbeat_at is not None and heartbeat_at.tzinfo is not None:
            now = datetime.now(heartbeat_at.tzinfo)
        else:
            now = datetime.now()
        heartbeat_age = (
            max(0.0, (now - heartbeat_at).total_seconds())
            if heartbeat_at is not None
            else float("inf")
        )
        runtime_alive = (
            status in {"starting", "running", "guard_closing_position", "guard_close_retry_pending"}
            and heartbeat_age <= self._runtime_state_stale_after_seconds()
        )

        self.health_state.last_runtime_state_at = heartbeat_at
        self.health_state.runtime_state_ok = runtime_alive
        self.health_state.bot_running = runtime_alive or self.run_process.state() != QProcess.NotRunning
        self.health_state.executor_alive = runtime_alive

        cycle_payload = payload.get("last_cycle_payload")
        if runtime_alive and isinstance(cycle_payload, dict):
            self._apply_cycle_payload(dict(cycle_payload), source="state")
        elif runtime_alive and self.run_process.state() == QProcess.NotRunning:
            state_label = "INICIANDO" if status == "starting" else "RODANDO"
            state_message = (
                "Engine headless ativa. Aguardando o primeiro ciclo."
                if status == "starting"
                else "Engine headless ativa em background."
            )
            if self.state.state_label != state_label or self.state.state_message != state_message:
                self.state.state_label = state_label
                self.state.state_message = state_message
                self.state.state_style = "wait"
                self._refresh_dashboard()

        self._refresh_connectivity_health(emit_event=False)

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

    def _report_launcher_internal_error(self, *, context: str, details: str, exc: Exception) -> None:
        summary = f"Falha interna do launcher durante {context}: {exc}"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_terminal_output(f"\n[{timestamp}] {summary}\n{details}\n")

        try:
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=details,
                message=summary,
                payload={
                    "context": context,
                    "exception_type": type(exc).__name__,
                },
                severity="error",
                event_code="system.launcher_internal_error",
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
                details="O launcher capturou a exceção e permaneceu aberto.",
                network=event.network,
                symbol=event.symbol,
                timeframe=event.timeframe,
                color="red",
                relevant=True,
                fingerprint=f"launcher_internal:{context}:{type(exc).__name__}",
                raw_detail=details,
                execution_summary="Exceção interna capturada no launcher.",
                simple_summary=summary,
            )
            self._push_event(event, prebuilt=prebuilt)
        except Exception:
            pass

    def _guard_launcher_action(self, context: str, callback):
        try:
            return callback()
        except Exception as exc:
            self._report_launcher_internal_error(context=context, details=traceback.format_exc(), exc=exc)
            return None

    def _sync_runtime_stopped_ui(self, *, exit_code: int, stop_requested: bool) -> None:
        self.health_state.bot_running = False
        self.health_state.executor_alive = False
        self.health_state.runtime_state_ok = False

        self._refresh_connectivity_health(emit_event=not stop_requested)
        
        if stop_requested and hasattr(self, 'notifier'):
            self.notifier.notify_stopped()

        if exit_code != 0 and not stop_requested:
            summary = "Bot parado após falha do runtime. Revise o erro."
            style = "error"
        else:
            summary = "Bot Parado"
            style = "blocked"

        # Atualiza o estado para a interface
        self.state.state_label = "PARADO"
        self.state.state_message = summary
        self.state.state_style = style
        self.state.humanized_simple_summary = summary
        self.state.last_decision = summary
        
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
            self._last_cycle_identity = None
            args = self._build_base_args(self._current_mode()) + ["run"]
            if self._start_process(self.run_process, args, label="Runtime", silent=False):
                self.health_state.bot_running = True
                self.health_state.executor_alive = True
                self.state.state_label = "INICIANDO"
                self.state.state_message = "Bot iniciando..."
                self.state.state_style = "wait"
                self.state.last_decision = "Bot iniciando..."
                self.state.humanized_simple_summary = ""
                self.state.humanized_execution_summary = ""
                self.state.last_cycle_fingerprint = ""
                self._update_controls()
                self._refresh_dashboard()
                self._push_event(
                    self._new_event(
                        source="launcher",
                        event_type="generic",
                        raw_line="Bot iniciando...",
                        message="Bot iniciando...",
                        severity="info",
                        event_code="system.runtime_started",
                    )
                )
            return

        if self.state.position_label in {"LONG", "SHORT"}:
            event = self._new_event(
                source="launcher",
                event_type="generic",
                raw_line="Parando bot com posição aberta.",
                message="Parando bot com posição aberta.",
                severity="warning",
                event_code="system.runtime_stop_requested_open_position",
            )
            self._push_event(event)

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
        self._last_cycle_identity = None
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
            if dialog.result() == SettingsDialog.RESTART_REQUESTED:
                self._restart_launcher()
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
        raw["launcher"]["auto_check_interval_seconds"] = int(auto_check_interval)

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)

        self.cfg = load_config(path)
        self.health_timer.setInterval(int(self.cfg.launcher.auto_check_interval_seconds) * 1000)
        self._hydrate_static_snapshot()
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        if self.state.decision_snapshot:
            final_action = float(self.state.decision_snapshot.get("final_action", 0.0) or 0.0)
            self.state.signal_label = self._signal_label(final_action)
        self._refresh_dashboard()
    
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
                    f"auto check {int(self.cfg.launcher.auto_check_interval_seconds)}s"
                ),
                event_code="system.settings_saved",
            )
        )

    def _clear_restart_request(self) -> None:
        self._restart_requested = False
        self._restart_helper_started = False
        self._restart_launch_program = None
        self._restart_launch_args = []

    def _launcher_restart_command(self) -> tuple[str, list[str]]:
        original_argv = [str(part) for part in (getattr(sys, "orig_argv", []) or []) if str(part)]
        if original_argv:
            return original_argv[0], original_argv[1:]
        return str(Path(sys.executable)), ["-m", "traderbot.launcher_entry"]

    def _ensure_restart_helper_started(self) -> bool:
        if self._restart_helper_started:
            return True

        program = self._restart_launch_program or str(Path(sys.executable))
        args = list(self._restart_launch_args or ["-m", "traderbot.launcher_entry"])
        helper_script = "\n".join(
            [
                "import os",
                "import socket",
                "import subprocess",
                "import time",
                f"program = {json.dumps(program)}",
                f"args = {json.dumps(args)}",
                f"repo = {json.dumps(str(REPO_ROOT))}",
                "deadline = time.time() + 15.0",
                "while time.time() < deadline:",
                "    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                "    try:",
                "        probe.bind(('127.0.0.1', 54321))",
                "    except OSError:",
                "        probe.close()",
                "        time.sleep(0.25)",
                "        continue",
                "    probe.close()",
                "    break",
                "else:",
                "    time.sleep(0.5)",
                "kwargs = {",
                "    'cwd': repo,",
                "    'stdin': subprocess.DEVNULL,",
                "    'stdout': subprocess.DEVNULL,",
                "    'stderr': subprocess.DEVNULL,",
                "}",
                "if os.name == 'nt':",
                "    kwargs['creationflags'] = (",
                "        getattr(subprocess, 'DETACHED_PROCESS', 0)",
                "        | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)",
                "        | getattr(subprocess, 'CREATE_NO_WINDOW', 0)",
                "    )",
                "else:",
                "    kwargs['start_new_session'] = True",
                "subprocess.Popen([program, *args], **kwargs)",
            ]
        )

        helper_kwargs: dict[str, Any] = {
            "cwd": str(REPO_ROOT),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            helper_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            helper_kwargs["start_new_session"] = True

        try:
            subprocess.Popen([str(Path(sys.executable)), "-c", helper_script], **helper_kwargs)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Reiniciar launcher",
                f"Não foi possível agendar o reinício automático do launcher.\n\nDetalhes: {exc}",
            )
            self._clear_restart_request()
            return False

        self._restart_helper_started = True
        return True

    def _restart_launcher(self) -> None:
        program, args = self._launcher_restart_command()
        self._restart_requested = True
        self._restart_helper_started = False
        self._restart_launch_program = program
        self._restart_launch_args = args
        self.close()

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
        self._queue_kill_switch(close_after=False)

    def _queue_kill_switch(self, *, close_after: bool) -> None:
        if close_after:
            self._close_after_kill_requested = True

        if self._kill_switch_in_progress:
            self._update_controls()
            return

        self.pending_emergency_close = True
        self._kill_switch_in_progress = True
        if self.run_process.state() != QProcess.NotRunning:
            self.runtime_stop_requested = True
            self._update_controls()
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
        if not self._start_process(self.task_process, args, label="Kill switch", silent=False):
            self._kill_switch_in_progress = False
            self._close_after_kill_requested = False
            self._update_controls()

    def _protected_process_ids(self) -> set[int]:
        protected = {int(os.getpid())}
        for process in (self.run_process, self.task_process):
            pid = int(process.processId() or 0)
            if pid > 0:
                protected.add(pid)
        return protected

    def _discover_traderbot_processes(self) -> list[dict[str, Any]]:
        repo_path = str(REPO_ROOT).replace("'", "''")
        script = (
            f"$repo = '{repo_path}'; "
            "$namePattern = '^(python|pythonw|py|pyw)\\.exe$'; "
            "$commandPattern = 'traderbot\\.(launcher_entry|launcher|main)'; "
            "$items = Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -match $namePattern -and $_.CommandLine -and ("
            "($_.CommandLine -like ('*' + $repo + '*')) -or "
            "($_.CommandLine -match $commandPattern)) "
            "} | ForEach-Object { "
            "$processInfo = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; "
            "[PSCustomObject]@{ "
            "ProcessId = $_.ProcessId; "
            "ParentProcessId = $_.ParentProcessId; "
            "Name = $_.Name; "
            "CommandLine = $_.CommandLine; "
            "MainWindowHandle = if ($processInfo) { [int64]$processInfo.MainWindowHandle } else { 0 }; "
            "MainWindowTitle = if ($processInfo) { $processInfo.MainWindowTitle } else { '' } "
            "} "
            "}; "
            "$items | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            fallback = "Falha ao listar processos do Traderbot. Fechar outras instâncias manualmente recomendado."
            raise RuntimeError(self._stderr_summary_line(result.stderr or result.stdout, fallback=fallback))

        payload = (result.stdout or "").strip()
        if not payload:
            return []

        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [dict(item) for item in parsed if isinstance(item, dict)]

    @staticmethod
    def _process_command_line(process: dict[str, Any]) -> str:
        return str(process.get("CommandLine") or "").strip().lower()

    def _is_launcher_process(self, process: dict[str, Any]) -> bool:
        command_line = self._process_command_line(process)
        return "traderbot.launcher" in command_line or "traderbot.launcher_entry" in command_line

    def _is_runtime_process(self, process: dict[str, Any]) -> bool:
        command_line = self._process_command_line(process)
        return "traderbot.main" in command_line and not self._is_launcher_process(process)

    @staticmethod
    def _has_visible_window(process: dict[str, Any]) -> bool:
        try:
            return int(process.get("MainWindowHandle") or 0) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _collect_process_tree(processes: list[dict[str, Any]], root_ids: set[int]) -> set[int]:
        children_by_parent: dict[int, set[int]] = {}
        for process in processes:
            pid = int(process.get("ProcessId") or 0)
            parent_pid = int(process.get("ParentProcessId") or 0)
            if pid <= 0 or parent_pid <= 0:
                continue
            children_by_parent.setdefault(parent_pid, set()).add(pid)

        preserved: set[int] = set()
        stack = [pid for pid in root_ids if pid > 0]
        while stack:
            current_pid = int(stack.pop())
            if current_pid <= 0 or current_pid in preserved:
                continue
            preserved.add(current_pid)
            stack.extend(children_by_parent.get(current_pid, ()))
        return preserved

    def _close_other_instances(self) -> None:
        if os.name != "nt":
            self._push_event(
                self._new_event(
                    source="launcher",
                    event_type="generic",
                    raw_line="Fechar outras instâncias não suportado fora do Windows.",
                    message="Fechar outras instâncias não suportado fora do Windows.",
                    severity="warning",
                    event_code="system.close_other_instances_unsupported",
                )
            )
            return

        answer = QMessageBox.warning(
            self,
            "Fechar outras instâncias do Traderbot",
            "Está janela será preservada, mas outras janelas do Traderbot serão fechadas. Deseja continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            protected = self._protected_process_ids()
            processes = self._discover_traderbot_processes()
        except Exception as exc:
            self._report_launcher_internal_error(
                context="listar outras instâncias",
                details=traceback.format_exc(),
                exc=exc,
            )
            return

        candidates: list[dict[str, Any]] = []
        preserved = self._collect_process_tree(processes, protected)
        launcher_roots = {
            int(process.get("ProcessId") or 0)
            for process in processes
            if self._is_launcher_process(process)
            and int(process.get("ProcessId") or 0) not in preserved
        }
        preserved.update(self._collect_process_tree(processes, launcher_roots))

        for process in processes:
            pid = int(process.get("ProcessId") or 0)
            if pid <= 0 or pid in preserved:
                continue
            if not self._is_runtime_process(process):
                continue
            if self._has_visible_window(process):
                continue
            candidates.append(process)

        if not candidates:
            self._push_event(
                self._new_event(
                    source="launcher",
                    event_type="generic",
                    raw_line="Nenhuma outra instância está aberta.",
                    message="Nenhuma outra instância do Traderbot encontrada.",
                    severity="info",
                    event_code="system.close_other_instances_empty",
                )
            )
            return

        candidate_ids = {int(item.get("ProcessId") or 0) for item in candidates}
        root_candidates = [
            item
            for item in candidates
            if int(item.get("ParentProcessId") or 0) not in candidate_ids
        ]
        terminated: list[int] = []
        failures: list[str] = []
        for process in root_candidates:
            pid = int(process.get("ProcessId") or 0)
            if pid <= 0:
                continue
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                terminated.append(pid)
                continue
            summary = self._stderr_summary_line(
                "\n".join(part for part in (result.stdout, result.stderr) if part),
                fallback=f"Falha ao encerrar PID {pid}.",
            )
            failures.append(summary)

        severity = "info" if failures else "info"
        message = (
            f"Instâncias encerradas: {len(terminated)}."
            if terminated
            else "Nenhuma instância foi encerrada."
        )
        if failures:
            message = f"{message} Falhas: {len(failures)}."

        self._append_terminal_output(f"{message}\n")
        self._push_event(
            self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=json.dumps(
                    {
                        "terminated_pids": terminated,
                        "failures": failures,
                    },
                    ensure_ascii=False,
                ),
                message=message,
                payload={
                    "terminated_pids": terminated,
                    "failures": failures,
                },
                severity=severity,
                event_code="system.close_other_instances",
            )
        )

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
        def _consume() -> None:
            while self.run_process.canReadLine():
                raw_payload = bytes(self.run_process.readLine()).decode("utf-8", errors="ignore")
                self._append_terminal_output(raw_payload)
                self._parse_line(raw_payload.rstrip("\r\n"), source="run")

        self._guard_launcher_action("leitura do stdout do runtime", _consume)

    def _handle_run_stderr(self) -> None:
        self._append_process_stderr(self.run_process)

    def _handle_task_output(self) -> None:
        def _consume() -> None:
            while self.task_process.canReadLine():
                raw_payload = bytes(self.task_process.readLine()).decode("utf-8", errors="ignore")
                self._append_terminal_output(raw_payload)
                self._parse_line(raw_payload.rstrip("\r\n"), source="task")

        self._guard_launcher_action("leitura do stdout do processo auxiliar", _consume)

    def _handle_task_stderr(self) -> None:
        self._append_process_stderr(self.task_process)

    def _handle_run_finished(self, exit_code: int, _status) -> None:
        def _consume() -> None:
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
            if self.pending_emergency_close:
                QTimer.singleShot(0, self._execute_pending_kill_switch)

        self._guard_launcher_action("finalizacao do runtime", _consume)

    def _handle_task_finished(self, exit_code: int, _status) -> None:
        def _consume() -> None:
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
                    show_dialog=not silent and not (label == "Kill switch" and self._close_after_kill_requested),
                )
        
            if self.pending_emergency_close:
                self._execute_pending_kill_switch()
            if label == "Kill switch":
                self._kill_switch_in_progress = False
                self._update_controls()
                if self._close_after_kill_requested:
                    self._finalize_close_after_kill(exit_code=exit_code, stderr_buffer=stderr_buffer)

        self._guard_launcher_action("finalizacao do processo auxiliar", _consume)

    def _finalize_close_after_kill(self, *, exit_code: int, stderr_buffer: str) -> None:
        position_open = self.state.position_label in {"LONG", "SHORT"}
        if not position_open:
            self._close_after_kill_requested = False
            self._force_close_armed = True
            QTimer.singleShot(0, self.close)
            return

        self._close_after_kill_requested = False
        fallback = "Kill switch executado, mas a posição segue aberta."
        summary = fallback
        if exit_code != 0 and stderr_buffer.strip():
            summary = self._stderr_summary_line(stderr_buffer, fallback=fallback)

        self._append_terminal_output(f"{summary}\n")
        self._push_event(
            self._new_event(
                source="launcher",
                event_type="generic",
                raw_line=stderr_buffer or summary,
                message=summary,
                severity="error",
                event_code="risk.close_after_kill_blocked",
            )
        )
        QMessageBox.critical(self, "Kill switch incompleto", summary)

    def _parse_line(self, raw_line: str, source: str) -> None:
        def _consume() -> None:
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
                self.state.state_message = "Aguardando proximo ciclo."
                self.state.state_style = "wait"
                self.state.last_decision = "Aguardando proximo ciclo."
                self._update_controls()
                self._refresh_dashboard()

        self._guard_launcher_action(f"parse de linha ({source})", _consume)

    def _ensure_daily_rollover(self) -> None:
        today = date.today()
        if self.stats_day == today:
            return
        self.stats_day = today
        self.state.operations_today_label = "0"
        self.state.blocked_today_label = "0"
        self.state.wins_today = 0
        self.state.losses_today = 0
        self.state.last_trade_reason_label = "Nenhuma operação hoje."
        self.state.last_skip_reason_label = "Nenhum bloqueio recente."

    def _increment_counter_label(self, current_value: str) -> str:
        try:
            return str(int(current_value) + 1)
        except (TypeError, ValueError):
            return "1"

    def _apply_status_payload(self, payload: dict[str, Any], *, source: str) -> None:
        check_at = datetime.now()
        
        # 1. Captura os dados de saúde e conta (suporta tanto o Check inicial quanto o Bot rodando)
        health = payload.get("health_summary", {})
        account = payload.get("account_summary", {})
        
        connected = bool(health.get("connected", payload.get("connected", False)))
        can_trade = bool(health.get("can_trade", payload.get("can_trade", False)))
        
        health_ok = connected and can_trade
        self.health_state.last_healthcheck_at = check_at
        self.health_state.last_check_execution_ok = True
        self.health_state.connection_ok = health_ok
        
        if health_ok:
            self.health_state.last_successful_healthcheck_at = check_at
            
        self.state.online = health_ok
        self.state.can_trade = can_trade
        self.state.network_label = str(payload.get("network", self._current_mode().label)).upper()
        
        # 2. Atualiza Saldo e Drawdown (Lógica unificada)
        balance_val = account.get("available_to_trade", payload.get("available_to_trade", 0.0))
        dd_val = account.get("drawdown_pct", payload.get("drawdown_pct", 0.0))
        
        self.state.balance_label = _currency(balance_val)
        self.state.drawdown_label = _pct(dd_val)
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        self.state.last_update_label = check_at.strftime("%H:%M:%S")
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)

        # 3. Atualiza o Status Visual para PARADO se o bot não estiver em execução
        runtime_running = self.run_process.state() != QProcess.NotRunning
        if connected and can_trade and not runtime_running:
            self.state.state_label = "PARADO"
            self.state.state_message = "Bot Parado"
            self.state.state_style = "blocked"
            self.state.humanized_simple_summary = "Bot Parado"

        # 4. ESSENCIAL: Força o Dashboard a mostrar os novos dados na tela
        self._refresh_dashboard()

        # 5. Notificação de evento (mantém seu log de eventos limpo)
        signature = (connected, can_trade, round(float(balance_val), 2))
        should_notify = signature != self.last_connection_signature
        self.last_connection_signature = signature
        
        if should_notify and not bool(self._active_task_context.get("silent")):
            msg = "Conectado com a Hyperliquid." if health_ok else "Falha de conexão ou permissão."
            event = self._new_event(
                source=source,
                event_type="status",
                raw_line=json.dumps(payload),
                message=msg,
                payload=payload,
                severity="info" if health_ok else "error",
                event_code="healthcheck.status",
            )
            self._push_event(event)

    def _apply_smoke_payload(self, payload: dict[str, Any], *, source: str) -> None:
        ok = payload.get("ok")
        msg = "Smoke test concluído com sucesso (abriu e fechou ordem)." if ok else "Falha no Smoke test."
        
        event = self._new_event(
            source=source,
            event_type="smoke",
            raw_line=json.dumps(payload, ensure_ascii=False),
            message=msg,
            payload=payload,
            severity="execution" if ok else "error",
            event_code="system.smoke_test",
        )
        self._push_event(event)

    def _apply_manual_close_payload(self, payload: dict[str, Any], *, source: str) -> None:
        if payload.get("ok"):
            self.state.position_label = "FLAT"
            self.state.position_status = "Nenhuma Ordem Aberta"
            self.state.position_size_label = "--"
            self.state.entry_label = "--"
            self.state.take_profit_label = "--"
            self.state.stop_loss_label = "--"
            self.state.pnl_label = "--"
            self.state.pnl_style = "muted"
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Posição encerrada manualmente."
            self.state.state_style = "wait"
            self._refresh_dashboard()

        ok = payload.get("ok")
        msg = "Posição encerrada manualmente com sucesso." if ok else "Falha ao tentar encerrar posição manualmente."
        
        event = self._new_event(
            source=source,
            event_type="manual_close",
            raw_line=json.dumps(payload, ensure_ascii=False),
            message=msg, # PASSE A MENSAGEM AQUI
            payload=payload,
            severity="risk" if ok else "warning",
            event_code="risk.manual_close",
        )
        self._push_event(event)

    def _push_event(self, event: LauncherEvent, *, prebuilt: HumanizedEvent | None = None) -> None:
        if prebuilt is None:
            summary = HumanizedEvent(
                source=event.source,
                event_type=event.event_type,
                raw_line=event.raw_line,
                message=event.message or "Evento do sistema",
                payload=dict(event.payload or {}),
                occurred_at=event.occurred_at,
                severity=event.severity,
                event_code=event.event_code,
                details=event.message,
                network=event.network,
                symbol=event.symbol,
                timeframe=event.timeframe,
                color="blue",
                relevant=True,
                fingerprint=f"{event.event_code}:{event.message}",
                raw_detail=event.raw_line,
                execution_summary=event.message,
                simple_summary=event.message,
            )
        else:
            summary = prebuilt

        self._apply_humanized_cycle_details(summary)
        
        if summary.relevant:
            widget, is_new = self._upsert_notification(summary)
            if is_new:
                if self.current_page != "events":
                    self.unread_events += 1
                    self._update_navigation_state()
                    self._show_event_toast(summary)
                
            self._apply_translated_notification(widget, summary)

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
            # O widget C++ já foi destruído pelo _trim_notifications antes da IA terminar a tradução.
            # Ignoramos silenciosamente para evitar crash do PySide6.
            pass

    def _apply_humanized_cycle_details(self, result: HumanizedEvent) -> None:
        if result.event_type != "cycle":
            return
        if not result.fingerprint or result.fingerprint != self.state.last_cycle_fingerprint:
            return
        if self.run_process.state() == QProcess.NotRunning and not self.health_state.runtime_state_ok:
            return

        updated = False
        for state_key, value in (
            ("humanized_market_state", result.market_state),
            ("humanized_model_interpretation", result.model_interpretation),
            ("humanized_filters_diagnostic", result.filters_diagnostic),
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
                        "runtime_state_ok": self.health_state.runtime_state_ok,
                        "last_healthcheck_at": self.health_state.last_healthcheck_at.isoformat() if self.health_state.last_healthcheck_at else None,
                        "last_successful_healthcheck_at": self.health_state.last_successful_healthcheck_at.isoformat() if self.health_state.last_successful_healthcheck_at else None,
                        "last_runtime_state_at": self.health_state.last_runtime_state_at.isoformat() if self.health_state.last_runtime_state_at else None,
                    },
                    message=f"health_status_change | status={status} | reason={reason}",
                    severity="info" if status == "online" else "warning" if status == "warning" else "error",
                    event_code=f"health.{reason}",
                )
            )

    def _derive_health_status(self, *, forced_reason: str | None = None) -> tuple[str, str]:
        state = self.health_state
        threshold = max(30, int(self.cfg.launcher.auto_check_interval_seconds) * 2)
        runtime_threshold = self._runtime_state_stale_after_seconds()
        runtime_age_seconds = None
        if state.last_runtime_state_at is not None:
            if state.last_runtime_state_at.tzinfo is not None:
                runtime_age_seconds = (
                    datetime.now(state.last_runtime_state_at.tzinfo) - state.last_runtime_state_at
                ).total_seconds()
            else:
                runtime_age_seconds = (datetime.now() - state.last_runtime_state_at).total_seconds()
        runtime_alive = (
            state.runtime_state_ok
            and runtime_age_seconds is not None
            and runtime_age_seconds <= runtime_threshold
        )

        if state.last_healthcheck_at is None:
            if runtime_alive:
                return "warning", forced_reason or "runtime_alive_healthcheck_pending"
            return "offline", forced_reason or "awaiting_first_healthcheck"

        age_seconds = (datetime.now() - state.last_healthcheck_at).total_seconds()

        if state.last_check_execution_ok is False and age_seconds <= threshold:
            if runtime_alive:
                return "warning", forced_reason or "runtime_alive_healthcheck_failed"
            return "offline", forced_reason or "health_check_command_failed"

        if age_seconds > threshold:
            if runtime_alive:
                return "warning", forced_reason or "runtime_alive_healthcheck_stale"
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
            "reference_currency": payload.get("reference_currency"),
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
            "model_votes": payload.get("model_votes"),
            "raw_actions": payload.get("raw_actions"),
            "reason": payload.get("decision_reason"),
            "final_action": final_action,
            "action_hold_threshold": payload.get("action_hold_threshold"),
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

    @staticmethod
    def _filter_status_text(passed: bool | None) -> str:
        if passed is None:
            return "SEM DADO"
        return "PASSOU" if passed else "FALHOU"

    def _build_filter_details_summary(self) -> tuple[str, str | None]:
        dec_snap = self.state.decision_snapshot if isinstance(self.state.decision_snapshot, dict) else {}
        filter_snap = self.state.filter_snapshot if isinstance(self.state.filter_snapshot, dict) else {}
        if not dec_snap and not filter_snap:
            return "Aguardando proximo ciclo...", "Nenhum dado recebido ainda."

        thresholds = filter_snap.get("regime_thresholds")
        if not isinstance(thresholds, dict):
            thresholds = {}

        hold_threshold = _safe_float(dec_snap.get("action_hold_threshold"))
        if hold_threshold is None:
            hold_threshold = float(self.cfg.environment.action_hold_threshold)
        final_action = _safe_float(dec_snap.get("final_action"))
        hold_strength = abs(final_action) if final_action is not None else None
        hold_passed = None if hold_strength is None else hold_strength >= hold_threshold

        regime_dist_raw = _safe_float(filter_snap.get("regime_dist_ema_240"))
        regime_dist_abs = abs(regime_dist_raw) if regime_dist_raw is not None else None
        min_abs_dist_ema_240 = _safe_float(thresholds.get("min_abs_dist_ema_240"))
        if min_abs_dist_ema_240 is None:
            min_abs_dist_ema_240 = float(self.cfg.environment.regime_min_abs_dist_ema_240)
        ema_passed = None if regime_dist_abs is None else regime_dist_abs >= min_abs_dist_ema_240

        regime_vol = _safe_float(filter_snap.get("regime_vol_regime_z"))
        min_vol_regime_z = _safe_float(thresholds.get("min_vol_regime_z"))
        if min_vol_regime_z is None:
            min_vol_regime_z = float(self.cfg.environment.regime_min_vol_regime_z)
        volume_passed = None if regime_vol is None else regime_vol > min_vol_regime_z

        def _get_status_html(passed: bool | None) -> str:
            if passed is None:
                return '<span style="color: #888;">SEM DADO</span>'
            return '<strong style="color: #4CAF50;">PASSOU</strong>' if passed else '<strong style="color: #F44336;">FALHOU</strong>'

        hold_val = f"{hold_strength * 100.0:.2f}" if hold_strength is not None else "--"
        hold_meta = f"{hold_threshold * 100.0:.2f}"
        
        vol_val = _optional_number(regime_vol, digits=2, signed=True)
        vol_meta = f"{min_vol_regime_z:.2f}"
        
        ema_val = f"{regime_dist_abs * 100.0:.2f}" if regime_dist_abs is not None else "--"
        ema_meta = f"{min_abs_dist_ema_240 * 100.0:.2f}"

        html_table = f"""
        <table width="100%" cellpadding="6" style="border-collapse: collapse; font-family: monospace;">
            <tr>
                <td width="30%"><b>HOLD</b></td>
                <td width="25%">{hold_meta}</td>
                <td width="25%">{hold_val}</td>
                <td width="20%">{_get_status_html(hold_passed)}</td>
            </tr>
            <tr>
                <td><b>Volume</b></td>
                <td>{vol_meta}</td>
                <td>{vol_val}</td>
                <td>{_get_status_html(volume_passed)}</td>
            </tr>
            <tr>
                <td><b>Dist EMA240</b></td>
                <td>{ema_meta}</td>
                <td>{ema_val}</td>
                <td>{_get_status_html(ema_passed)}</td>
            </tr>
        </table>
        """

        failed_filters: list[str] = []
        if hold_passed is False:
            failed_filters.append("HOLD")
        if volume_passed is False:
            failed_filters.append("Volume")
        if ema_passed is False:
            failed_filters.append("Distancia EMA240")

        blocked_reason_human = str(
            filter_snap.get("blocked_reason_human")
            or (self.state.blocker_label if self.state.blocker_label != "Sem bloqueio" else "")
            or ""
        ).strip()
        
        if blocked_reason_human:
            note = f"Bloqueio atual: {blocked_reason_human}."
            if failed_filters:
                note += f" Checks reprovados: {', '.join(failed_filters)}."
            return html_table, note

        if failed_filters:
            return html_table, f"Checks reprovados nesta barra: {', '.join(failed_filters)}."
        return html_table, "Todos os checks desta barra passaram."

    def _apply_cycle_payload(self, payload: dict[str, Any], *, source: str) -> None:
        cycle_identity = self._cycle_identity(payload)
        if cycle_identity == self._last_cycle_identity:
            return
        self._last_cycle_identity = cycle_identity

        self._ensure_daily_rollover()
        self.health_state.bot_running = True
        self.health_state.executor_alive = True
        self.health_state.runtime_state_ok = True
        self.state.last_update_label = datetime.now().strftime("%H:%M:%S")

        market_snapshot = self._market_snapshot_from_payload(payload)
        feature_snapshot = self._feature_snapshot_from_payload(payload)
        decision_snapshot = self._decision_snapshot_from_payload(payload)
        filter_snapshot = self._filter_snapshot_from_payload(payload)
        blocked_reason = filter_snapshot.get("blocked_reason", payload.get("blocked_reason"))
        silenced_cycle_issue = is_launcher_silenced_cycle_issue(payload, blocked_reason)
        if silenced_cycle_issue:
            filter_snapshot = dict(filter_snapshot)
            filter_snapshot["blocked_reason"] = None
            filter_snapshot["blocked_reason_human"] = None
            blocked_reason = None

        self.state.market_snapshot = market_snapshot
        self.state.feature_snapshot = feature_snapshot
        self.state.decision_snapshot = decision_snapshot
        self.state.filter_snapshot = filter_snapshot
        regime_valid = filter_snapshot.get("regime_valid_for_entry", False)
        # -- 1. Status do Mercado (Tendência + Filtro) --
        regime_valid = filter_snapshot.get("regime_valid_for_entry", False)
        dist_ema = _safe_float(feature_snapshot.get("dist_ema_240"))
        
        if not regime_valid:
            self.state.market_status_label = "Lateral"
        elif dist_ema is not None and dist_ema > 0:
            self.state.market_status_label = "Alta"
        elif dist_ema is not None and dist_ema < 0:
            self.state.market_status_label = "Baixa"
        else:
            self.state.market_status_label = "Favorável"

        final_action = decision_snapshot.get("final_action", payload.get("final_action", 0.0))
        vote_bucket = decision_snapshot.get("vote_bucket", payload.get("vote_bucket", "--"))
        manual_protection_required = bool(payload.get("manual_protection_required", False)) and not silenced_cycle_issue
        manual_protection_message = str(payload.get("manual_protection_message") or "").strip()
        payload_error = payload.get("error") if not silenced_cycle_issue else None

        self.state.reference_currency = _optional_price(
            market_snapshot.get("reference_currency", payload.get("reference_currency", 0.0))
        )
        self.state.risk_label = _pct(payload.get("dynamic_risk_pct", payload.get("risk_pct", 0.0)))
        self.state.strength_label = _strength(final_action)
        self.state.regime_label = "valido" if filter_snapshot.get("regime_valid_for_entry") else "bloqueado"
        self.state.signal_label = self._signal_label(float(final_action or 0.0))
        self.state.direction_label = str(vote_bucket or "--").upper()
        self.state.balance_label = _currency(payload.get("available_to_trade", 0.0))
        self.state.blocker_label = self._human_block_label(str(blocked_reason)) if blocked_reason else "Sem bloqueio"
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)
        self.state.humanized_execution_summary = ""
        self.state.humanized_simple_summary = ""
        self.state.last_cycle_fingerprint = ""

        position_label = str(payload.get("position_label", "FLAT")).upper()
        native_tp_sl = self._native_tp_sl_payload(payload)
        self.state.native_tp_sl_status = native_tp_sl
        self.state.position_label = position_label
        self.state.position_size_label = _currency(payload.get("adjusted_notional", payload.get("notional_value", 0.0)))
        self.state.entry_label = _currency(payload.get("position_avg_entry_price", 0.0))
        self.state.take_profit_label = _currency(payload.get("take_profit_price", 0.0))
        self.state.stop_loss_label = _currency(payload.get("stop_loss_price", 0.0))
        pnl_value = float(payload.get("position_unrealized_pnl", 0.0) or 0.0)
        self.state.pnl_label = _currency(pnl_value)
        self.state.pnl_style = "success" if pnl_value >= 0 else "error"

        if manual_protection_required:
            self.state.state_label = "AVISO"
            self.state.state_message = (
                manual_protection_message or "Proteção manual de TP/SL necessária."
            )
            self.state.state_style = "blocked"
        elif payload_error:
            self.state.state_label = "ERRO"
            self.state.state_message = (
                manual_protection_message or "Execução com erro. Revisar operação."
            )
            self.state.state_style = "error"
        elif blocked_reason:
            self.state.state_label = "BLOQUEADO"
            self.state.state_message = self._human_block_label(str(blocked_reason))
            self.state.state_style = "blocked"
        elif payload.get("position_is_open"):
            self.state.state_label = position_label
            self.state.state_message = f"Posição {position_label} em andamento."
            self.state.state_style = "long" if position_label == "LONG" else "short"
        elif self.state.signal_label == "HOLD":
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Sem entrada nesta barra."
            self.state.state_style = "wait"
        else:
            self.state.state_label = "FLAT"
            self.state.state_message = "Nenhuma Ordem Aberta"
            self.state.state_style = "flat"

        if payload.get("position_is_open"):
            self.state.position_status = self._position_status_text(
                self.state.position_label
            )
        elif blocked_reason:
            self.state.position_status = "Nenhuma Ordem Aberta"
        else:
            self.state.position_status = "Nenhuma Ordem Aberta"

        if payload.get("opened_trade"):
            summary_text = "Posição aberta."
            self.state.operations_today_label = self._increment_counter_label(self.state.operations_today_label)
            self.state.last_trade_reason_label = summary_text
        elif payload.get("closed_trade"):
            summary_text = "Posição fechada."
            self.state.last_trade_reason_label = summary_text
            exit_reason = str(payload.get("exit_reason") or "").lower()
            if "take_profit" in exit_reason or "take profit" in exit_reason:
                self.state.wins_today += 1
            elif "stop_loss" in exit_reason or "stop loss" in exit_reason:
                self.state.losses_today += 1
            elif float(payload.get("position_unrealized_pnl", 0.0) or 0.0) >= 0:
                self.state.wins_today += 1
            else:
                self.state.losses_today += 1
        elif blocked_reason:
            summary_text = self._human_block_label(str(blocked_reason))
            self.state.blocked_today_label = self._increment_counter_label(self.state.blocked_today_label)
            self.state.last_skip_reason_label = summary_text
        elif silenced_cycle_issue and payload.get("position_is_open"):
            summary_text = f"Posicao {position_label} em andamento."
        else:
            summary_text = "Ciclo concluído sem novas entradas."
            self.state.last_skip_reason_label = summary_text

        self.state.last_decision = summary_text
        self.state.state_message = summary_text
        self._refresh_dashboard()

        if silenced_cycle_issue:
            return

        event = self._new_event(
            source=source,
            event_type="cycle",
            raw_line=json.dumps(payload, ensure_ascii=False),
            message=summary_text,
            payload=payload,
            severity="execution",
            event_code="execution.cycle",
        )
        self.state.last_cycle_fingerprint = f"{event.event_code}:{event.message}"
        self._push_event(event)

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

        # 1. Status do Mercado
        if "Alta" in self.state.market_status_label:
            market_tone = "success" # Verde
        elif "Baixa" in self.state.market_status_label:
            market_tone = "error"   # Vermelho
        elif "Lateral" in self.state.market_status_label:
            market_tone = "default" # Cinza
        else:
            market_tone = "default"
            
        self.market_status_tile.update_tile(self.state.market_status_label, tone=market_tone)
        
        # 2. Drawdown
        dd_val = _safe_float(self.state.drawdown_label.replace('%', '')) or 0.0
        dd_tone = "error" if dd_val < -5.0 else "warning" if dd_val < -1.5 else "default"
        self.drawdown_tile.update_tile(self.state.drawdown_label, tone=dd_tone)

        # 3. Viés da IA
        bias_tone = "success" if "Compra" in self.state.ai_bias_label else "error" if "Venda" in self.state.ai_bias_label else "muted"
        self.ai_bias_tile.update_tile(self.state.ai_bias_label, tone=bias_tone)

        # 4. Placar dos Modelos
        score_tone = "success" if "Compra" in self.state.ensemble_score_label else "error" if "Venda" in self.state.ensemble_score_label else "muted"
        self.ensemble_score_tile.update_tile(self.state.ensemble_score_label, tone=score_tone)

        # 5. Operações Hoje
        self.operations_tile.update_tile(_display_value(self.state.operations_today_label, "0"), tone="default")
        
        # 6. Win Rate
        total_closed = self.state.wins_today + self.state.losses_today
        if total_closed > 0:
            win_rate = (self.state.wins_today / total_closed) * 100.0
            win_rate_str = f"{win_rate:.1f}% ({self.state.wins_today}V - {self.state.losses_today}D)"
            win_tone = "success" if win_rate >= 50 else "error"
        else:
            win_rate_str = "--"
            win_tone = "default"
        self.winrate_tile.update_tile(win_rate_str, tone=win_tone)

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

        # --- ATUALIZAÇÃO DA ABA DE DETALHES ---

        self.heartbeat_tile.update_tile(_display_value(self.state.last_valid_check_label, "--"), tone="default")
        self.operations_today_tile.update_tile(_display_value(self.state.operations_today_label, "0"), tone="default")
        self.blocked_today_tile.update_tile(_display_value(self.state.blocked_today_label, "0"), tone="default")

        dec_snap = self.state.decision_snapshot if isinstance(self.state.decision_snapshot, dict) else {}
        if dec_snap:
            final_action = dec_snap.get("final_action", 0.0)
            direction = str(dec_snap.get("vote_bucket", "--")).upper()
            reason = str(dec_snap.get("reason", self.state.last_decision))

            self.details_final_action.setText(
                f"Consenso: {direction} | Força do sinal: {_strength(final_action)}"
            )
            self.details_decision_reason.setText(reason)
        else:
            self.details_final_action.setText("Carregando...")
            self.details_decision_reason.setText("Nenhum dado recebido ainda.")

        _fit_label_height(self.details_final_action)
        _fit_label_height(self.details_decision_reason)

        filter_summary, filter_note = self._build_filter_details_summary()
        self.details_filters_card.set_summary(filter_summary)
        self.details_filters_card.set_note(filter_note)

        while self.details_votes_container.count():
            item = self.details_votes_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self._detach_layout_items(item.layout())

        votes_data = dec_snap.get("model_votes") or dec_snap.get("raw_actions") or {}
        fallback_votes = dec_snap.get("votes") or {}
        if (
            not votes_data
            and isinstance(fallback_votes, dict)
            and any(str(key).lower() not in {"buy", "hold", "sell"} for key in fallback_votes)
        ):
            votes_data = fallback_votes

        if isinstance(votes_data, str):
            try:
                votes_data = json.loads(votes_data)
            except Exception:
                votes_data = {}

        vote_items: list[tuple[str, Any]] = []
        if isinstance(votes_data, dict) and votes_data:
            vote_items = [(str(model_name), vote_value) for model_name, vote_value in votes_data.items()]
        elif isinstance(votes_data, list) and votes_data:
            configured_model_names = self.cfg.execution.selected_model_names or self.cfg.execution.ensemble_model_names
            vote_items = [
                (
                    str(configured_model_names[index]) if index < len(configured_model_names) else f"model_{index + 1}",
                    vote_value,
                )
                for index, vote_value in enumerate(votes_data)
            ]

        if dec_snap and vote_items:
            for model_name, vote_value in vote_items:
                row = QHBoxLayout()

                clean_name = str(model_name).replace("_", " ").upper()
                m_label = QLabel(clean_name)
                m_label.setObjectName("subtleText")

                val_float = _safe_float(vote_value)
                if val_float is not None:
                    tamanho_voto = f"{val_float * 100.0:+.1f}"
                    # Vai buscar o valor salvo nas definições (ex: 0.60)
                    hold_threshold = float(self.cfg.environment.action_hold_threshold)

                    if val_float >= hold_threshold:
                        v_text = f"LONG({tamanho_voto})"
                        tone = "success"
                    elif val_float <= -hold_threshold:
                        v_text = f"SHORT({tamanho_voto})"
                        tone = "error"
                    else:
                        v_text = f"HOLD({tamanho_voto})"
                        tone = "muted"
                else:
                    v_text = str(vote_value)
                    tone = "default"

                v_label = QLabel(v_text)
                v_label.setObjectName("metricValue")
                v_label.setProperty("sizeVariant", "body")
                _set_widget_property(v_label, "tone", tone)

                row.addWidget(m_label)
                row.addWidget(v_label, 0, Qt.AlignRight)
                self.details_votes_container.addLayout(row)
        else:
            lbl = QLabel("Aguardando registro numérico dos modelos no próximo ciclo...")
            lbl.setObjectName("HeroHint")
            self.details_votes_container.addWidget(lbl)

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

    @staticmethod
    def _native_tp_sl_payload(payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("native_tp_sl")
        return dict(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _position_label_from_side(side: Any) -> str:
        try:
            side_value = int(side or 0)
        except (TypeError, ValueError):
            side_value = 0
        if side_value > 0:
            return "LONG"
        if side_value < 0:
            return "SHORT"
        return "FLAT"

    def _position_status_text(
        self,
        position_label: str,
    ) ->str:
        if position_label not in {"LONG", "SHORT"}:
            return "Nenhuma Ordem Aberta"

        return "Ordem Aberta"

    def _human_block_label(self, reason: str) -> str:
        mapping = {
            "regime_filter": "Entrada bloqueada pelo filtro",
            "cooldown": "Entrada bloqueada pelo cooldown",
            "force_exit_only_by_tp_sl": "Sinal ignorado: Ordem aberta",
            "min_notional_risk": "Entrada pulada por ser abaixo do valor mínimo da corretora",
            "exchange_position_present": "Ordem ignorada por posição já presente na corretora",
            "duplicate_cycle_order": "Ordem ignorada para evitar ordem duplicada",
        }
        mapping.update(
            {
                "order_cooldown": "Entrada bloqueada pelo cooldown entre ordens",
                "open_position_locked": "Sinal ignorado: Ordem aberta.",
                "duplicate_cycle": "Entrada ignorada para evitar ordem duplicada na mesma vela",
            }
        )
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
        # Tenta travar a porta para garantir que é a única instância
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

        msg = "O TraderBot Launcher já esta rodando!\n\n"
        if old_pid:
            msg += f"Deseja forçar o encerramento da instância fantasma (PID: {old_pid}) para liberar o sistema?"
        else:
            msg += "Não encontrei o PID para matar automaticamente. Você terá que fechar o 'pythonw.exe' pelo Gerenciador de Tarefas do Windows."

        answer = QMessageBox.warning(
            None,
            "Instância Fantasma Detectada",
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
                    "Instância fantasma encerrada com sucesso!\nVoce já pode abrir o Launcher normalmente.",
                )
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "Erro",
                    f"Não foi possivel matar o processo fantasma.\nDetalhes: {e}\n\nAbra o terminal e execute 'taskkill /F /IM pythonw.exe /T {old_pid} e taskkill /F /IM python.exe /T {old_pid}' para encerrar manualmente."
                )
        return 1

    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    if LAUNCHER_ICON_PATH.exists():
        launcher_icon = QIcon(str(LAUNCHER_ICON_PATH))
        app.setWindowIcon(launcher_icon)
    theme_path = REPO_ROOT / "theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    window = TraderBotLauncher()
    if LAUNCHER_ICON_PATH.exists():
        window.setWindowIcon(launcher_icon)
    window.show()
    _apply_windows_taskbar_icon(window)
    QTimer.singleShot(0, lambda target=window: _apply_windows_taskbar_icon(target))
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
    app.setApplicationName("Trader Launcher")
    app.setApplicationDisplayName("Trader Launcher")
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName(WINDOWS_APP_ID)
    return _run_launcher_with_pid_lock(app)


if __name__ == "__main__":
    raise SystemExit(main())
