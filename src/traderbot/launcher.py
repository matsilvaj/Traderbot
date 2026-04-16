from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QObject, QProcess, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QStyle,
    QVBoxLayout,
    QWidget,
)

from traderbot.config import AppConfig, load_config
from traderbot.launcher_services import HumanizedEvent, LauncherEvent, OpenAILogTranslator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


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


def _json_marker(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


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
    last_decision: str = "Sem decisao no momento."
    last_update_label: str = "--"
    network_label: str = "TESTNET"
    position_label: str = "FLAT"
    position_status: str = "Sem sinal ativo"
    position_size_label: str = "--"
    entry_label: str = "--"
    take_profit_label: str = "--"
    stop_loss_label: str = "--"
    pnl_label: str = "--"
    pnl_style: str = "muted"
    last_raw_detail: str = ""


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
        self.signals.finished.emit(result)


class MetricTile(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        self.label = QLabel(label)
        self.label.setObjectName("metaLabel")
        layout.addWidget(self.label)

        self.value = QLabel("--")
        self.value.setObjectName("metricValue")
        self.value.setProperty("tone", "default")
        layout.addWidget(self.value)

        self.note = QLabel("")
        self.note.setObjectName("subtleText")
        _configure_dynamic_label(self.note)
        self.note.hide()
        layout.addWidget(self.note)

    def update_tile(self, value: str, *, note: str | None = None, tone: str = "default") -> None:
        self.value.setText(value)
        _set_widget_property(self.value, "tone", tone)
        if note:
            self.note.setText(note)
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

        title = QLabel(self.summary.message)
        title.setObjectName("dialogTitle")
        _configure_dynamic_label(title)
        root.addWidget(title)

        meta = QLabel(f"{self.summary.severity.upper()}  {self.summary.occurred_at.strftime('%H:%M:%S')}")
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
        raw_box.setPlainText(self.summary.raw_detail or "Sem log tecnico disponivel.")
        root.addWidget(raw_box, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_button = QPushButton("Fechar")
        close_button.setObjectName("primary")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        root.addLayout(footer)


class NotificationItemWidget(QFrame):
    def __init__(self, summary: HumanizedEvent):
        super().__init__()
        self.summary = summary
        self.count = 1
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("event")
        self._build_ui()
        self.apply_summary(summary)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

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
        layout.addWidget(self.message_label)
        self.setToolTip("Ver log tecnico")

    def apply_summary(self, summary: HumanizedEvent) -> None:
        self.summary = summary
        self.dot_label.setText("•")
        self.severity_badge.setText(summary.severity.upper())
        self.message_label.setText(summary.message)
        self.message_label.updateGeometry()
        self.updateGeometry()
        self.time_label.setText(summary.occurred_at.strftime("%H:%M:%S"))
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
        self.setModal(True)
        self.setWindowTitle("Configuracoes")
        self.resize(460, 420)
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

        self.openai_checkbox = QCheckBox("Usar OpenAI para resumir eventos")
        self.openai_checkbox.setChecked(bool(self.cfg.launcher.openai_enabled))

        self.api_key_hint = QLabel(
            f"API key via {self.cfg.launcher.openai_api_key_env}. "
            f"{'Detectada' if os.getenv(self.cfg.launcher.openai_api_key_env or 'OPENAI_API_KEY') else 'Nao detectada'}."
        )
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

        form = QGridLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(14)

        form.addWidget(self._field_label("Risco maximo por trade"), 0, 0)
        form.addWidget(self.risk_input, 0, 1)
        form.addWidget(self._field_label("Auto check"), 1, 0)
        form.addWidget(self.autocheck_input, 1, 1)

        root.addLayout(form)
        root.addWidget(self.openai_checkbox)
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
        advanced_layout.addWidget(self._field_label("Smoke side"), 0, 0)
        advanced_layout.addWidget(self.smoke_side_button, 0, 1)
        advanced_layout.addWidget(self._field_label("Wait smoke"), 1, 0)
        advanced_layout.addWidget(self.smoke_wait_input, 1, 1)
        self.advanced_frame.hide()
        root.addWidget(self.advanced_frame)

        footer = QHBoxLayout()
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

    def smoke_side(self) -> str:
        return "sell" if self.smoke_side_button.isChecked() else "buy"

    def smoke_wait_seconds(self) -> float:
        return float(self.smoke_wait_input.value())

    def risk_pct(self) -> float:
        return float(self.risk_input.value()) / 100.0

    def auto_check_interval(self) -> int:
        return int(self.autocheck_input.value())

    def openai_enabled(self) -> bool:
        return bool(self.openai_checkbox.isChecked())

    def openai_model(self) -> str:
        return "gpt-4o-mini"



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
        self.last_check_at: datetime | None = None
        self.last_runtime_seen_at: datetime | None = None
        self.last_connection_signature: tuple[Any, ...] | None = None

        self.log_translator = OpenAILogTranslator(self.cfg)

        self.run_process = self._build_process(self._handle_run_output, self._handle_run_finished)
        self.task_process = self._build_process(self._handle_task_output, self._handle_task_finished)

        self.setWindowTitle("TraderBot Launcher")
        self.resize(1400, 900)
        self.setMinimumSize(1180, 760)

        self._build_ui()
        self._hydrate_static_snapshot()
        self._set_online(False)
        self._refresh_dashboard()
        self._update_controls()

        self.health_timer = QTimer(self)
        self.health_timer.setInterval(int(self.cfg.launcher.auto_check_interval_seconds) * 1000)
        self.health_timer.timeout.connect(self._auto_check)
        self.health_timer.start()

        QTimer.singleShot(350, lambda: self._run_check(silent=True))

    def _build_process(self, output_handler, finished_handler) -> QProcess:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(output_handler)
        process.finished.connect(finished_handler)
        return process

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(26, 24, 26, 24)
        root.setSpacing(20)

        root.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setSpacing(22)
        body.addWidget(self._build_dashboard(), 2)
        body.addWidget(self._build_notifications_panel(), 1)
        root.addLayout(body, 1)

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

        self.last_check_chip = QLabel("sem check")
        self.last_check_chip.setObjectName("SubtleChip")
        left.addWidget(self.last_check_chip, 0, Qt.AlignLeft)
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
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("HeroCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(22, 22, 22, 22)
        hero_layout.setSpacing(18)

        hero_top = QHBoxLayout()
        hero_top.setSpacing(12)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(10)

        self.state_pill = QLabel("AGUARDANDO")
        self.state_pill.setObjectName("StatePill")
        hero_left.addWidget(self.state_pill, 0, Qt.AlignLeft)

        self.state_message = QLabel("Aguardando dados do bot")
        self.state_message.setObjectName("HeroTitle")
        _configure_dynamic_label(self.state_message)
        hero_left.addWidget(self.state_message)

        self.last_decision_label = QLabel("Sem decisao no momento.")
        self.last_decision_label.setObjectName("HeroHint")
        _configure_dynamic_label(self.last_decision_label)
        hero_left.addWidget(self.last_decision_label)

        self.block_reason_label = QLabel("Sem bloqueio")
        self.block_reason_label.setObjectName("BlockChip")
        hero_left.addWidget(self.block_reason_label, 0, Qt.AlignLeft)

        hero_top.addLayout(hero_left, 1)

        actions = QVBoxLayout()
        actions.setSpacing(10)

        self.run_button = QPushButton("INICIAR BOT")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._toggle_run)
        actions.addWidget(self.run_button)

        self.smoke_button = QPushButton("Smoke test")
        self.smoke_button.setObjectName("SecondaryButton")
        self.smoke_button.clicked.connect(self._run_smoke)
        actions.addWidget(self.smoke_button)

        self.kill_button = QPushButton("Kill switch")
        self.kill_button.setObjectName("DangerGhostButton")
        self.kill_button.clicked.connect(self._trigger_kill_switch)
        actions.addWidget(self.kill_button)
        actions.addStretch(1)

        hero_top.addLayout(actions)
        hero_layout.addLayout(hero_top)

        hero_bottom = QHBoxLayout()
        hero_bottom.setSpacing(18)

        balance_stack = QVBoxLayout()
        balance_stack.setSpacing(6)
        self.balance_caption = QLabel("Saldo operacional")
        self.balance_caption.setObjectName("MetricLabel")
        balance_stack.addWidget(self.balance_caption)
        self.balance_inline = QLabel("--")
        self.balance_inline.setObjectName("BalanceValue")
        balance_stack.addWidget(self.balance_inline)
        hero_bottom.addLayout(balance_stack)
        hero_bottom.addStretch(1)
        hero_layout.addLayout(hero_bottom)

        layout.addWidget(hero)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        self.regime_tile = MetricTile("Regime")
        self.signal_tile = MetricTile("Sinal")
        self.direction_tile = MetricTile("Ensemble")
        self.strength_tile = MetricTile("Forca")
        self.price_tile = MetricTile("Preco de referencia")
        self.risk_tile = MetricTile("Risco efetivo")

        grid.addWidget(self.regime_tile, 0, 0)
        grid.addWidget(self.signal_tile, 0, 1)
        grid.addWidget(self.direction_tile, 0, 2)
        grid.addWidget(self.strength_tile, 1, 0)
        grid.addWidget(self.price_tile, 1, 1)
        grid.addWidget(self.risk_tile, 1, 2)
        layout.addLayout(grid)

        position = QFrame()
        position.setObjectName("PositionCard")
        position_layout = QVBoxLayout(position)
        position_layout.setContentsMargins(22, 22, 22, 22)
        position_layout.setSpacing(16)

        position_top = QHBoxLayout()
        position_top.setSpacing(14)

        position_meta = QVBoxLayout()
        position_meta.setSpacing(8)

        self.position_pill = QLabel("FLAT")
        self.position_pill.setObjectName("PositionPill")
        position_meta.addWidget(self.position_pill, 0, Qt.AlignLeft)

        self.position_status = QLabel("Sem sinal ativo")
        self.position_status.setObjectName("PositionStatus")
        _configure_dynamic_label(self.position_status)
        position_meta.addWidget(self.position_status)

        position_top.addLayout(position_meta, 1)

        pnl_box = QVBoxLayout()
        pnl_box.setSpacing(6)
        pnl_label = QLabel("PnL aberto")
        pnl_label.setObjectName("MetricLabel")
        pnl_box.addWidget(pnl_label)
        self.pnl_value = QLabel("--")
        self.pnl_value.setObjectName("PnlValue")
        pnl_box.addWidget(self.pnl_value)
        position_top.addLayout(pnl_box)

        size_box = QVBoxLayout()
        size_box.setSpacing(6)
        size_label = QLabel("Tamanho")
        size_label.setObjectName("MetricLabel")
        size_box.addWidget(size_label)
        self.position_size_value = QLabel("--")
        self.position_size_value.setObjectName("PositionNotional")
        size_box.addWidget(self.position_size_value)
        position_top.addLayout(size_box)

        position_layout.addLayout(position_top)

        levels = QGridLayout()
        levels.setHorizontalSpacing(16)
        levels.setVerticalSpacing(14)

        self.entry_value = self._build_level_box(levels, 0, "Entrada")
        self.tp_value = self._build_level_box(levels, 1, "Take profit")
        self.sl_value = self._build_level_box(levels, 2, "Stop loss")
        position_layout.addLayout(levels)

        layout.addWidget(position, 1)

        return container

    def _build_level_box(self, layout: QGridLayout, column: int, title: str) -> QLabel:
        wrapper = QFrame()
        wrapper.setObjectName("LevelBox")
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(16, 14, 16, 14)
        inner.setSpacing(8)

        label = QLabel(title)
        label.setObjectName("MetricLabel")
        inner.addWidget(label)

        value = QLabel("--")
        value.setObjectName("LevelValue")
        inner.addWidget(value)
        layout.addWidget(wrapper, 0, column)
        return value

    def _build_notifications_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("NotificationPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        head = QHBoxLayout()
        head.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title = QLabel("Eventos")
        title.setObjectName("PanelTitle")
        title_box.addWidget(title)
        self.notification_hint = QLabel("Somente o que realmente importa para operar.")
        self.notification_hint.setObjectName("PanelHint")
        _configure_dynamic_label(self.notification_hint)
        title_box.addWidget(self.notification_hint)
        head.addLayout(title_box, 1)

        self.notification_status = QLabel("OpenAI: local")
        self.notification_status.setObjectName("SubtleChip")
        head.addWidget(self.notification_status, 0, Qt.AlignTop)
        layout.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setObjectName("NotificationScroll")

        self.notification_host = QWidget()
        self.notifications_layout = QVBoxLayout(self.notification_host)
        self.notifications_layout.setContentsMargins(0, 0, 0, 0)
        self.notifications_layout.setSpacing(12)
        self.notifications_layout.addStretch(1)
        scroll.setWidget(self.notification_host)

        layout.addWidget(scroll, 1)
        return panel

    def _hydrate_static_snapshot(self) -> None:
        snapshot = self.log_translator.status_snapshot()
        if snapshot["client_ready"]:
            self.notification_status.setText(f"OpenAI: {snapshot['model']}")
        elif snapshot["enabled"] and not snapshot["api_key_present"]:
            self.notification_status.setText("OpenAI: sem chave")
        else:
            self.notification_status.setText("OpenAI: local")
        self._sync_mode_buttons()

    def _sync_mode_buttons(self) -> None:
        mode = self._current_mode()
        self.mode_badge.setText(mode.label.upper())
        for key, button in self.mode_buttons.items():
            button.setChecked(key == self.current_mode)

    def _update_controls(self) -> None:
        run_busy = self.run_process.state() != QProcess.NotRunning
        task_busy = self.task_process.state() != QProcess.NotRunning
        for button in self.mode_buttons.values():
            button.setEnabled(not run_busy and not task_busy)
        self.refresh_button.setEnabled(not task_busy and not run_busy)
        self.smoke_button.setEnabled(not task_busy and not run_busy)
        self.settings_button.setEnabled(not task_busy)
        self.kill_button.setEnabled(not task_busy)
        self.run_button.setText("PARAR BOT" if run_busy else "INICIAR BOT")

    def _current_mode(self) -> LauncherMode:
        return MODES[self.current_mode]

    def _build_base_args(self, mode: LauncherMode) -> list[str]:
        args = ["-m", "traderbot.main", "--config", str(self.config_path)]
        args.extend(["--network-override", mode.network])
        args.extend(["--execution-mode-override", mode.execution_mode])
        if mode.allow_live_trading:
            args.append("--allow-live-trading")
        return args

    def _start_process(self, process: QProcess, args: list[str], *, label: str, silent: bool = False) -> bool:
        process.setWorkingDirectory(str(REPO_ROOT))
        self._active_task_context = {"label": label, "silent": silent}
        process.start(sys.executable, args)
        if not process.waitForStarted(3000):
            event = LauncherEvent(
                source="launcher",
                event_type="generic",
                raw_line=f"Nao foi possivel iniciar {label}.",
                message=f"Nao foi possivel iniciar {label}.",
            )
            self._push_event(event)
            return False
        if not silent:
            event = LauncherEvent(
                source="launcher",
                event_type="generic",
                raw_line=f"{label} iniciado.",
                message=f"{label} iniciado.",
            )
            self._push_event(event)
        self._update_controls()
        return True

    def _run_check(self, *, silent: bool = False) -> None:
        if self.task_process.state() != QProcess.NotRunning:
            return
        args = self._build_base_args(self._current_mode()) + ["check-hyperliquid"]
        self._start_process(self.task_process, args, label="Check Hyperliquid", silent=silent)

    def _auto_check(self) -> None:
        if self.run_process.state() != QProcess.NotRunning:
            self._refresh_connectivity_health()
            return
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
            args = self._build_base_args(self._current_mode()) + ["run"]
            self._start_process(self.run_process, args, label="Runtime", silent=False)
            return

        if self.state.position_label in {"LONG", "SHORT"}:
            event = LauncherEvent(
                source="launcher",
                event_type="generic",
                raw_line="Existe posicao aberta. Use o kill switch antes de parar o bot.",
                message="Existe posicao aberta. Use o kill switch antes de parar o bot.",
            )
            self._push_event(event)
            return

        self.run_process.kill()

    def _switch_mode(self, mode_key: str) -> None:
        if mode_key == self.current_mode:
            return
        if self.run_process.state() != QProcess.NotRunning or self.task_process.state() != QProcess.NotRunning:
            return
        self.current_mode = mode_key
        self.state.network_label = self._current_mode().label.upper()
        self._sync_mode_buttons()
        self._set_online(False)
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
            openai_enabled=dialog.openai_enabled(),
            openai_model=dialog.openai_model(),
            auto_check_interval=dialog.auto_check_interval(),
        )

    def _save_settings(
        self,
        *,
        max_risk_per_trade: float,
        openai_enabled: bool,
        openai_model: str,
        auto_check_interval: int,
    ) -> None:
        path = Path(self.config_path)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        raw.setdefault("environment", {})
        raw["environment"]["max_risk_per_trade"] = float(max_risk_per_trade)

        raw.setdefault("launcher", {})
        raw["launcher"]["openai_enabled"] = bool(openai_enabled)
        raw["launcher"]["openai_model"] = str(openai_model)
        raw["launcher"]["auto_check_interval_seconds"] = int(auto_check_interval)

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)

        self.cfg = load_config(path)
        self.log_translator = OpenAILogTranslator(self.cfg)
        self.health_timer.setInterval(int(self.cfg.launcher.auto_check_interval_seconds) * 1000)
        self._hydrate_static_snapshot()
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        self._refresh_dashboard()
        self._push_event(
            LauncherEvent(
                source="launcher",
                event_type="generic",
                raw_line="Configuracoes salvas.",
                message="Configuracoes salvas. Os proximos checks e execucoes ja usarao os novos valores.",
            )
        )

    def _trigger_kill_switch(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Confirmar kill switch",
            "Encerrar a posicao atual e parar o bot?",
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
        if not self.pending_emergency_close or self.task_process.state() != QProcess.NotRunning:
            return
        self.pending_emergency_close = False
        args = self._build_base_args(self._current_mode()) + ["close-hyperliquid-position"]
        self._start_process(self.task_process, args, label="Kill switch", silent=False)

    def _handle_run_output(self) -> None:
        payload = bytes(self.run_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for raw_line in payload.splitlines():
            self._parse_line(raw_line, source="run")

    def _handle_task_output(self) -> None:
        payload = bytes(self.task_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for raw_line in payload.splitlines():
            self._parse_line(raw_line, source="task")

    def _handle_run_finished(self, exit_code: int, _status) -> None:
        self._update_controls()
        self._refresh_connectivity_health()
        event = LauncherEvent(
            source="launcher",
            event_type="generic",
            raw_line=f"Runtime finalizado (exit={exit_code}).",
            message=f"Runtime finalizado (exit={exit_code}).",
        )
        self._push_event(event)

    def _handle_task_finished(self, exit_code: int, _status) -> None:
        silent = bool(self._active_task_context.get("silent"))
        self._update_controls()
        if exit_code != 0 and not silent:
            self._push_event(
                LauncherEvent(
                    source="launcher",
                    event_type="generic",
                    raw_line=f"Processo auxiliar finalizado com codigo {exit_code}.",
                    message=f"Processo auxiliar finalizado com codigo {exit_code}.",
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
            self.state.last_decision = "Runtime iniciado. Aguardando proximo ciclo."
            self._refresh_dashboard()

        event = LauncherEvent(source=source, event_type="generic", raw_line=line, message=plain)
        self._push_event(event)

    def _apply_status_payload(self, payload: dict[str, Any], *, source: str) -> None:
        self.last_check_at = datetime.now()
        self.state.online = bool(payload.get("connected"))
        self.state.can_trade = bool(payload.get("can_trade"))
        self.state.network_label = str(payload.get("network", self._current_mode().label)).upper()
        self.state.balance_label = _currency(payload.get("available_to_trade", 0.0))
        self.state.risk_label = _pct(self.cfg.environment.max_risk_per_trade)
        self.state.last_update_label = self.last_check_at.strftime("%H:%M:%S")
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)
        if self.state.online and self.state.can_trade:
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Sem sinal ativo"
            self.state.state_style = "wait"
        self._set_online(bool(payload.get("connected")) and bool(payload.get("can_trade")))
        self._refresh_dashboard()

        signature = (
            payload.get("connected"),
            payload.get("can_trade"),
            round(float(payload.get("available_to_trade", 0.0) or 0.0), 2),
            payload.get("network"),
        )
        should_notify = not bool(self._active_task_context.get("silent")) or signature != self.last_connection_signature
        self.last_connection_signature = signature
        if should_notify:
            event = LauncherEvent(
                source=source,
                event_type="status",
                raw_line=json.dumps(payload, ensure_ascii=False),
                payload=payload,
            )
            self._push_event(event)

    def _apply_cycle_payload(self, payload: dict[str, Any], *, source: str) -> None:
        self.last_runtime_seen_at = datetime.now()
        self.state.last_update_label = self.last_runtime_seen_at.strftime("%H:%M:%S")
        self.state.reference_price = _price(payload.get("reference_price", 0.0))
        self.state.risk_label = _pct(payload.get("dynamic_risk_pct", payload.get("risk_pct", 0.0)))
        self.state.strength_label = _strength(payload.get("final_action", 0.0))
        self.state.regime_label = "valido" if payload.get("regime_valid") else "bloqueado"
        self.state.signal_label = self._signal_label(float(payload.get("final_action", 0.0) or 0.0))
        self.state.direction_label = str(payload.get("vote_bucket", "--")).upper()
        self.state.balance_label = _currency(payload.get("available_to_trade", 0.0))
        self.state.blocker_label = (
            self._human_block_label(str(payload.get("blocked_reason")))
            if payload.get("blocked_reason")
            else "Sem bloqueio"
        )
        self.state.last_raw_detail = json.dumps(payload, ensure_ascii=False, indent=2)

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
        elif payload.get("blocked_reason"):
            self.state.state_label = "BLOQUEADO"
            self.state.state_message = self._human_block_label(str(payload.get("blocked_reason")))
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
        elif payload.get("blocked_reason"):
            self.state.position_status = "Sem posicao; entrada bloqueada"
        else:
            self.state.position_status = "Sem sinal ativo"

        self._set_online(True)
        self._refresh_dashboard()

        event = LauncherEvent(
            source=source,
            event_type="cycle",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
        )
        summary = self.log_translator.local.summarize(event)
        self.state.last_decision = summary.message
        self._refresh_dashboard()
        self._push_event(event, prebuilt=summary)

    def _apply_smoke_payload(self, payload: dict[str, Any], *, source: str) -> None:
        event = LauncherEvent(
            source=source,
            event_type="smoke",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
        )
        self._push_event(event)

    def _apply_manual_close_payload(self, payload: dict[str, Any], *, source: str) -> None:
        if payload.get("ok"):
            self.state.position_label = "FLAT"
            self.state.position_status = "Sem posicao aberta"
            self.state.position_size_label = "--"
            self.state.entry_label = "--"
            self.state.take_profit_label = "--"
            self.state.stop_loss_label = "--"
            self.state.pnl_label = "--"
            self.state.pnl_style = "muted"
            self.state.state_label = "AGUARDANDO"
            self.state.state_message = "Posicao encerrada manualmente."
            self.state.state_style = "wait"
            self.state.position_status = "Sem sinal ativo"
            self._refresh_dashboard()

        event = LauncherEvent(
            source=source,
            event_type="manual_close",
            raw_line=json.dumps(payload, ensure_ascii=False),
            payload=payload,
        )
        self._push_event(event)

    def _push_event(self, event: LauncherEvent, *, prebuilt: HumanizedEvent | None = None) -> None:
        summary = prebuilt or self.log_translator.local.summarize(event)
        if summary.relevant:
            widget = self._upsert_notification(summary)
            if self.log_translator.enabled:
                task = TranslationTask(self.log_translator, event, summary)
                task.signals.finished.connect(lambda result, target=widget: self._apply_translated_notification(target, result))
                self._thread_pool.start(task)

        if summary.severity == "error":
            self.state.state_label = "ERRO"
            self.state.state_message = summary.message
            self.state.state_style = "error"
            self._refresh_dashboard()

    def _upsert_notification(self, summary: HumanizedEvent) -> NotificationItemWidget:
        existing = self._notification_widgets.get(summary.fingerprint)
        if existing is not None:
            existing.bump(summary)
            self.notifications_layout.removeWidget(existing)
            self.notifications_layout.insertWidget(0, existing)
            return existing

        widget = NotificationItemWidget(summary)
        self._notification_widgets[summary.fingerprint] = widget
        self._notification_order.insert(0, summary.fingerprint)
        self.notifications_layout.insertWidget(0, widget)

        limit = int(self.cfg.launcher.notification_limit)
        while len(self._notification_order) > limit:
            stale_fingerprint = self._notification_order.pop()
            stale = self._notification_widgets.pop(stale_fingerprint, None)
            if stale is not None:
                self.notifications_layout.removeWidget(stale)
                stale.deleteLater()
        return widget

    def _apply_translated_notification(self, widget: NotificationItemWidget, result: HumanizedEvent) -> None:
        if widget is None:
            return
        widget.apply_summary(result)

    def _refresh_dashboard(self) -> None:
        self.mode_badge.setText(self.state.network_label)
        self.last_check_chip.setText(
            f"ultimo check {self.state.last_update_label}" if self.state.last_update_label != "--" else "sem check"
        )
        self.state_pill.setText(self.state.state_label)
        _set_widget_property(self.state_pill, "status", self.state.state_style)
        self.state_message.setText(self.state.state_message)
        self.last_decision_label.setText(self.state.last_decision)
        self.block_reason_label.setText(self.state.blocker_label)
        self.balance_inline.setText(self.state.balance_label)
        self.state_message.updateGeometry()
        self.last_decision_label.updateGeometry()

        self.regime_tile.update_tile(_display_value(self.state.regime_label, "indefinido"), tone="default")
        self.signal_tile.update_tile(_display_value(self.state.signal_label, "nenhum"), tone="default")
        self.direction_tile.update_tile(_display_value(self.state.direction_label, "nenhuma"), tone="default")
        self.strength_tile.update_tile(self.state.strength_label, tone="execution")
        self.price_tile.update_tile(self.state.reference_price, tone="default")
        self.risk_tile.update_tile(self.state.risk_label, tone="warning")

        self.position_pill.setText(self.state.position_label)
        position_style = "long" if self.state.position_label == "LONG" else "short" if self.state.position_label == "SHORT" else "flat"
        _set_widget_property(self.position_pill, "status", position_style)
        self.position_status.setText(self.state.position_status)
        self.position_status.updateGeometry()
        self.pnl_value.setText(self.state.pnl_label)
        _set_widget_property(self.pnl_value, "tone", self.state.pnl_style)
        self.position_size_value.setText(self.state.position_size_label)
        self.entry_value.setText(self.state.entry_label)
        self.tp_value.setText(self.state.take_profit_label)
        self.sl_value.setText(self.state.stop_loss_label)

    def _set_online(self, value: bool) -> None:
        self.state.online = value
        text = "online" if value else "offline"
        self.health_chip.setText(text)
        _set_widget_property(self.health_chip, "state", text)

    def _refresh_connectivity_health(self) -> None:
        now = datetime.now()
        threshold = int(self.cfg.launcher.auto_check_interval_seconds) * 2
        if self.last_runtime_seen_at is not None:
            alive = (now - self.last_runtime_seen_at).total_seconds() <= threshold
            self._set_online(alive)
            return
        if self.last_check_at is not None:
            alive = (now - self.last_check_at).total_seconds() <= threshold and self.state.can_trade
            self._set_online(alive)
            return
        self._set_online(False)

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
            "force_exit_only_by_tp_sl": "Sinal ignorado: posicao travada por TP/SL",
            "min_notional_risk": "Entrada pulada por risco minimo x notional",
            "exchange_position_present": "Entrada ignorada: posicao ja existe",
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


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    theme_path = REPO_ROOT / "theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    window = TraderBotLauncher()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
