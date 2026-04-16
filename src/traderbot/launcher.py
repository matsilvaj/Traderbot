from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from traderbot.config import AppConfig, load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+ \| [^|]+ \| ")


@dataclass(frozen=True)
class LauncherMode:
    key: str
    label: str
    network: str | None
    execution_mode: str
    allow_live_trading: bool
    pill_color: str
    subtitle: str


MODES: dict[str, LauncherMode] = {
    "paper_local": LauncherMode(
        key="paper_local",
        label="Paper Local",
        network=None,
        execution_mode="paper_local",
        allow_live_trading=False,
        pill_color="#7dd3fc",
        subtitle="Simulacao local com o core validado.",
    ),
    "testnet": LauncherMode(
        key="testnet",
        label="Testnet",
        network="testnet",
        execution_mode="exchange",
        allow_live_trading=True,
        pill_color="#4ade80",
        subtitle="Execucao real em ambiente de teste.",
    ),
    "mainnet": LauncherMode(
        key="mainnet",
        label="Mainnet",
        network="mainnet",
        execution_mode="exchange",
        allow_live_trading=True,
        pill_color="#fb7185",
        subtitle="Execucao real. Use com cuidado.",
    ),
}


def _currency(value: Any, prefix: str = "$") -> str:
    try:
        return f"{prefix} {float(value):,.2f}"
    except (TypeError, ValueError):
        return f"{prefix} 0.00"


def _price(value: Any) -> str:
    try:
        return f"$ {float(value):,.2f}"
    except (TypeError, ValueError):
        return "$ 0.00"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def _strip_log_prefix(line: str) -> str:
    return LOG_PREFIX_RE.sub("", line).strip()


def _extract_json_marker(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", accent: str = "#e5e7eb", subtitle: str = ""):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")
        layout.addWidget(self.title_label)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.value_label.setStyleSheet(f"color: {accent};")
        layout.addWidget(self.value_label)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("MetricSubtitle")
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)

    def update_card(self, value: str, subtitle: str | None = None, accent: str | None = None) -> None:
        self.value_label.setText(value)
        if subtitle is not None:
            self.subtitle_label.setText(subtitle)
        if accent is not None:
            self.value_label.setStyleSheet(f"color: {accent};")


class SettingsDialog(QDialog):
    emergency_requested = Signal()

    def __init__(self, cfg: AppConfig, mode: LauncherMode, smoke_side: str, smoke_wait_seconds: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuracoes")
        self.setModal(True)
        self.resize(520, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(18)

        intro = QLabel("Launcher operacional do TraderBot RL")
        intro.setObjectName("DialogTitle")
        root.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)

        self.mode_label = QLabel(mode.label)
        self.mode_label.setObjectName("DialogValue")
        form.addRow("Modo atual", self.mode_label)

        self.decision_mode_label = QLabel(cfg.execution.decision_mode)
        self.decision_mode_label.setObjectName("DialogValue")
        form.addRow("Modelo de execucao", self.decision_mode_label)

        self.risk_min_label = QLabel(_pct(cfg.environment.min_risk_per_trade))
        self.risk_min_label.setObjectName("DialogValue")
        form.addRow("Risco minimo", self.risk_min_label)

        self.risk_max_label = QLabel(_pct(cfg.environment.max_risk_per_trade))
        self.risk_max_label.setObjectName("DialogValue")
        form.addRow("Risco maximo", self.risk_max_label)

        self.smoke_side_input = QPushButton("BUY" if smoke_side.lower() == "buy" else "SELL")
        self.smoke_side_input.setCheckable(True)
        self.smoke_side_input.setChecked(smoke_side.lower() == "sell")
        self.smoke_side_input.clicked.connect(self._toggle_smoke_side)
        form.addRow("Smoke side", self.smoke_side_input)

        self.smoke_wait_input = QDoubleSpinBox()
        self.smoke_wait_input.setRange(1.0, 30.0)
        self.smoke_wait_input.setDecimals(1)
        self.smoke_wait_input.setSingleStep(1.0)
        self.smoke_wait_input.setValue(float(smoke_wait_seconds))
        form.addRow("Wait smoke (s)", self.smoke_wait_input)

        self.models_label = QLabel("\n".join(cfg.execution.selected_model_names or cfg.execution.ensemble_model_names))
        self.models_label.setObjectName("DialogValue")
        self.models_label.setWordWrap(True)
        form.addRow("Modelos ativos", self.models_label)

        root.addLayout(form)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("Separator")
        root.addWidget(separator)

        warning = QLabel("Zona de emergencia: encerra a posicao atual na Hyperliquid e interrompe o bot.")
        warning.setObjectName("DialogHint")
        warning.setWordWrap(True)
        root.addWidget(warning)

        self.kill_button = QPushButton("ENCERRAR POSICOES E PARAR BOT")
        self.kill_button.setObjectName("DangerButton")
        self.kill_button.clicked.connect(self._confirm_emergency)
        root.addWidget(self.kill_button)

    def _toggle_smoke_side(self) -> None:
        self.smoke_side_input.setText("SELL" if self.smoke_side_input.isChecked() else "BUY")

    def _confirm_emergency(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Confirmar kill switch",
            "Tem certeza que deseja encerrar a posicao atual e parar o bot?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.emergency_requested.emit()

    def smoke_side(self) -> str:
        return "sell" if self.smoke_side_input.isChecked() else "buy"

    def smoke_wait_seconds(self) -> float:
        return float(self.smoke_wait_input.value())


class TraderBotLauncher(QMainWindow):
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        super().__init__()
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.current_mode = "testnet"
        self.smoke_side = "buy"
        self.smoke_wait_seconds = 3.0
        self.pending_emergency_close = False
        self.run_active = False
        self.current_position_open = False

        self.run_process = self._build_process(self._handle_run_output, self._handle_run_finished)
        self.task_process = self._build_process(self._handle_task_output, self._handle_task_finished)

        self.setWindowTitle("TraderBot RL Launcher")
        self.resize(1280, 860)
        self.setMinimumSize(1120, 760)
        self._build_ui()
        self._apply_styles()
        self._refresh_static_info()
        self._update_mode_badges()
        self._update_buttons()
        self._append_log(
            "Launcher pronto. Use Testar Conexao ou Smoke Test antes de iniciar o bot.",
            level="info",
            detail="Modo padrao: testnet exchange com ensemble_majority_vote.",
        )

    def _build_process(self, output_handler, finished_handler) -> QProcess:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(output_handler)
        process.finished.connect(finished_handler)
        return process

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        root.addWidget(self._build_header())
        root.addWidget(self._build_dashboard(), 1)
        root.addWidget(self._build_terminal(), 1)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("HeaderFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(6)

        title = QLabel("TraderBot RL")
        title.setObjectName("AppTitle")
        left.addWidget(title)

        self.mode_hint_label = QLabel("Launcher operacional simples para H1 + Hyperliquid.")
        self.mode_hint_label.setObjectName("SubtleLabel")
        left.addWidget(self.mode_hint_label)
        layout.addLayout(left, 1)

        mode_box = QHBoxLayout()
        mode_box.setSpacing(10)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QPushButton] = {}
        for key in ("paper_local", "testnet", "mainnet"):
            button = QPushButton(MODES[key].label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, mode_key=key: self._switch_mode(mode_key))
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            mode_box.addWidget(button)
        layout.addLayout(mode_box)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.check_button = QPushButton("Testar Conexao")
        self.check_button.clicked.connect(self._run_check)
        actions.addWidget(self.check_button)

        self.smoke_button = QPushButton("Smoke Test")
        self.smoke_button.clicked.connect(self._run_smoke)
        actions.addWidget(self.smoke_button)

        self.settings_button = QPushButton("Configuracoes")
        self.settings_button.clicked.connect(self._open_settings)
        actions.addWidget(self.settings_button)

        self.run_button = QPushButton("INICIAR BOT")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._toggle_run)
        actions.addWidget(self.run_button)

        layout.addLayout(actions)
        return frame

    def _build_dashboard(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        top = QGridLayout()
        top.setHorizontalSpacing(16)
        top.setVerticalSpacing(16)

        self.mode_card = MetricCard("Modo", MODES[self.current_mode].label, subtitle=MODES[self.current_mode].subtitle)
        self.balance_card = MetricCard("Saldo disponivel", "$ 0.00", accent="#4ade80", subtitle="Aguardando check.")
        self.models_card = MetricCard(
            "Ensemble",
            str(len(self.cfg.execution.selected_model_names or self.cfg.execution.ensemble_model_names)),
            accent="#7dd3fc",
            subtitle="Modelos selecionados ativos.",
        )
        self.status_card = MetricCard("Status", "FLAT", accent="#cbd5e1", subtitle="Aguardando proximo sinal.")

        top.addWidget(self.mode_card, 0, 0)
        top.addWidget(self.balance_card, 0, 1)
        top.addWidget(self.models_card, 0, 2)
        top.addWidget(self.status_card, 0, 3)
        layout.addLayout(top)

        heart = QFrame()
        heart.setObjectName("HeroCard")
        heart_layout = QVBoxLayout(heart)
        heart_layout.setContentsMargins(22, 22, 22, 22)
        heart_layout.setSpacing(18)

        hero_top = QHBoxLayout()
        hero_top.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(10)
        self.position_badge = QLabel("FLAT")
        self.position_badge.setObjectName("PositionBadge")
        left.addWidget(self.position_badge, 0, Qt.AlignLeft)

        self.position_value_label = QLabel("$ 0.00")
        self.position_value_label.setObjectName("HeroValue")
        left.addWidget(self.position_value_label)

        self.position_note_label = QLabel("FLAT - aguardando proximo sinal.")
        self.position_note_label.setObjectName("HeroNote")
        left.addWidget(self.position_note_label)
        hero_top.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(QLabel("PnL aberto"))
        self.pnl_label = QLabel("$ 0.00")
        self.pnl_label.setObjectName("PnlValue")
        right.addWidget(self.pnl_label)
        self.runtime_state_label = QLabel("Bot parado")
        self.runtime_state_label.setObjectName("SubtleLabel")
        right.addWidget(self.runtime_state_label)
        hero_top.addLayout(right, 1)
        heart_layout.addLayout(hero_top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)

        self.entry_value = QLabel("--")
        self.tp_value = QLabel("--")
        self.sl_value = QLabel("--")
        self.price_value = QLabel("--")
        self.vote_value = QLabel("--")
        self.regime_value = QLabel("--")

        metrics = [
            ("Preco de entrada", self.entry_value),
            ("Take Profit", self.tp_value),
            ("Stop Loss", self.sl_value),
            ("Preco de referencia", self.price_value),
            ("Votacao ensemble", self.vote_value),
            ("Regime", self.regime_value),
        ]

        for idx, (label_text, value_label) in enumerate(metrics):
            label = QLabel(label_text)
            label.setObjectName("GridLabel")
            value_label.setObjectName("GridValue")
            row = idx // 3
            col = (idx % 3) * 2
            grid.addWidget(label, row, col)
            grid.addWidget(value_label, row, col + 1)

        heart_layout.addLayout(grid)
        layout.addWidget(heart)

        return wrapper

    def _build_terminal(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("TerminalFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel("Terminal inteligente")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.log_list = QListWidget()
        self.log_list.setAlternatingRowColors(False)
        self.log_list.itemDoubleClicked.connect(self._show_log_detail)
        layout.addWidget(self.log_list, 1)
        return frame

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #0f1720;
                color: #e5e7eb;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 13px;
            }
            QMainWindow {
                background: #0b1118;
            }
            QFrame#HeaderFrame, QFrame#HeroCard, QFrame#TerminalFrame, QFrame#MetricCard {
                background: #131c27;
                border: 1px solid #243244;
                border-radius: 18px;
            }
            QFrame#HeroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #131c27, stop:1 #10202c);
            }
            QFrame#TerminalFrame {
                background: #0d141d;
            }
            QLabel#AppTitle {
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#SubtleLabel, QLabel#MetricSubtitle, QLabel#HeroNote, QLabel#DialogHint {
                color: #94a3b8;
            }
            QLabel#MetricTitle, QLabel#GridLabel {
                color: #9fb0c7;
                font-size: 12px;
                letter-spacing: 0.4px;
            }
            QLabel#MetricValue {
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#HeroValue, QLabel#PnlValue {
                font-family: "Consolas", "Roboto Mono", monospace;
                font-size: 34px;
                font-weight: 700;
            }
            QLabel#GridValue, QLabel#DialogValue {
                font-family: "Consolas", "Roboto Mono", monospace;
                font-size: 15px;
                color: #f8fafc;
            }
            QLabel#SectionTitle, QLabel#DialogTitle {
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#PositionBadge {
                background: #22303f;
                color: #cbd5e1;
                border-radius: 13px;
                padding: 6px 12px;
                font-weight: 700;
                min-width: 74px;
            }
            QPushButton {
                background: #182533;
                border: 1px solid #2a3b4f;
                border-radius: 12px;
                padding: 10px 14px;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #46617e;
            }
            QPushButton:checked {
                border-color: #7dd3fc;
                background: #0f2534;
            }
            QPushButton#PrimaryButton {
                background: #15803d;
                border-color: #16a34a;
                min-width: 150px;
            }
            QPushButton#PrimaryButton:hover {
                background: #16a34a;
            }
            QPushButton#DangerButton {
                background: #7f1d1d;
                border-color: #ef4444;
                padding: 14px 16px;
            }
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
                padding: 4px;
            }
            QListWidget::item {
                background: #131c27;
                border: 1px solid #223040;
                border-radius: 12px;
                padding: 10px 12px;
                margin: 4px 0;
            }
            QListWidget::item:selected {
                background: #1a2734;
                border-color: #4f6b88;
            }
            QFrame#Separator {
                color: #223040;
            }
            """
        )

    def _refresh_static_info(self) -> None:
        model_names = self.cfg.execution.selected_model_names or self.cfg.execution.ensemble_model_names
        self.models_card.update_card(
            str(len(model_names)),
            subtitle=", ".join(model_names),
            accent="#7dd3fc",
        )

    def _switch_mode(self, mode_key: str) -> None:
        self.current_mode = mode_key
        self._update_mode_badges()
        self._append_log(
            f"Modo alterado para {MODES[mode_key].label}.",
            level="info",
            detail=f"Rede: {MODES[mode_key].network or 'local'} | execucao: {MODES[mode_key].execution_mode}",
        )

    def _update_mode_badges(self) -> None:
        mode = MODES[self.current_mode]
        for key, button in self.mode_buttons.items():
            button.setChecked(key == self.current_mode)
        self.mode_card.update_card(mode.label, subtitle=mode.subtitle, accent=mode.pill_color)
        self.mode_hint_label.setText(mode.subtitle)

    def _update_buttons(self) -> None:
        run_busy = self.run_process.state() != QProcess.NotRunning
        task_busy = self.task_process.state() != QProcess.NotRunning
        self.run_active = run_busy
        self.run_button.setText("PARAR BOT" if run_busy else "INICIAR BOT")
        for button in self.mode_buttons.values():
            button.setEnabled(not run_busy and not task_busy)
        self.check_button.setEnabled(not task_busy and not run_busy)
        self.smoke_button.setEnabled(not task_busy and not run_busy)
        self.settings_button.setEnabled(not task_busy)
        if run_busy:
            self.run_button.setStyleSheet("background: #7f1d1d; border-color: #ef4444; min-width: 150px;")
        else:
            self.run_button.setStyleSheet("")

    def _current_mode(self) -> LauncherMode:
        return MODES[self.current_mode]

    def _build_base_args(self, mode: LauncherMode) -> list[str]:
        args = ["-m", "traderbot.main", "--config", str(self.config_path)]
        if mode.network is not None:
            args.extend(["--network-override", mode.network])
        args.extend(["--execution-mode-override", mode.execution_mode])
        if mode.allow_live_trading:
            args.append("--allow-live-trading")
        return args

    def _start_process(self, process: QProcess, args: list[str], label: str) -> None:
        process.setWorkingDirectory(str(REPO_ROOT))
        process.start(sys.executable, args)
        if not process.waitForStarted(3000):
            self._append_log(
                f"Nao foi possivel iniciar: {label}.",
                level="error",
                detail=f"Comando: {sys.executable} {' '.join(args)}",
            )
            return
        self._append_log(
            f"{label} iniciado.",
            level="info",
            detail=f"Comando: {sys.executable} {' '.join(args)}",
        )
        self._update_buttons()

    def _run_check(self) -> None:
        mode = self._current_mode()
        args = self._build_base_args(mode) + ["check-hyperliquid"]
        self._start_process(self.task_process, args, "Check Hyperliquid")

    def _run_smoke(self) -> None:
        mode = self._current_mode()
        args = self._build_base_args(mode) + [
            "smoke-hyperliquid",
            "--side",
            self.smoke_side,
            "--wait-seconds",
            str(self.smoke_wait_seconds),
        ]
        if mode.network in {"mainnet", "testnet"}:
            args.extend(["--network", mode.network])
        if mode.network == "mainnet":
            args.append("--allow-mainnet")
        self._start_process(self.task_process, args, f"Smoke Test {mode.label}")

    def _toggle_run(self) -> None:
        if self.run_process.state() == QProcess.NotRunning:
            mode = self._current_mode()
            args = self._build_base_args(mode) + ["run"]
            self.runtime_state_label.setText("Bot inicializando...")
            self._start_process(self.run_process, args, f"Run {mode.label}")
            return

        if self.current_position_open and self._current_mode().execution_mode == "exchange":
            self._append_log(
                "Existe posicao aberta. Use o kill switch para encerrar a posicao antes de parar o bot.",
                level="warning",
                detail="O runtime atual gerencia TP/SL no proprio processo. Parar com posicao aberta pode interromper esse controle.",
            )
            return

        self.run_process.kill()
        self.runtime_state_label.setText("Bot parado")
        self._append_log("Bot interrompido manualmente.", level="warning")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.cfg,
            self._current_mode(),
            smoke_side=self.smoke_side,
            smoke_wait_seconds=self.smoke_wait_seconds,
            parent=self,
        )
        dialog.emergency_requested.connect(self._trigger_kill_switch)
        dialog.exec()
        self.smoke_side = dialog.smoke_side()
        self.smoke_wait_seconds = dialog.smoke_wait_seconds()

    def _trigger_kill_switch(self) -> None:
        mode = self._current_mode()
        if mode.execution_mode != "exchange":
            self._append_log(
                "Kill switch indisponivel em paper local.",
                level="warning",
                detail="No modo paper_local basta parar o processo.",
            )
            return

        self.pending_emergency_close = True
        self._append_log(
            "Kill switch acionado. Parando o bot e tentando encerrar a posicao atual.",
            level="error",
        )
        if self.run_process.state() != QProcess.NotRunning:
            self.run_process.kill()
            QTimer.singleShot(900, self._execute_pending_kill_switch)
        else:
            self._execute_pending_kill_switch()

    def _execute_pending_kill_switch(self) -> None:
        if not self.pending_emergency_close or self.task_process.state() != QProcess.NotRunning:
            return
        self.pending_emergency_close = False
        mode = self._current_mode()
        args = self._build_base_args(mode) + ["close-hyperliquid-position"]
        self._start_process(self.task_process, args, "Kill Switch")

    def _handle_run_output(self) -> None:
        payload = bytes(self.run_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for raw_line in payload.splitlines():
            self._parse_line(raw_line, source="run")

    def _handle_task_output(self) -> None:
        payload = bytes(self.task_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for raw_line in payload.splitlines():
            self._parse_line(raw_line, source="task")

    def _handle_run_finished(self, exit_code: int, _status) -> None:
        self.runtime_state_label.setText("Bot parado")
        self._append_log(
            f"Processo do bot finalizado (exit={exit_code}).",
            level="warning" if exit_code else "info",
        )
        self._update_buttons()

    def _handle_task_finished(self, exit_code: int, _status) -> None:
        if exit_code != 0:
            self._append_log(
                f"Processo auxiliar finalizado com codigo {exit_code}.",
                level="warning",
            )
        self._update_buttons()
        if self.pending_emergency_close:
            self._execute_pending_kill_switch()

    def _parse_line(self, raw_line: str, source: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        status_payload = _extract_json_marker(line, "Status Hyperliquid: ")
        if status_payload is not None:
            self._apply_status_payload(status_payload)
            return

        cycle_payload = _extract_json_marker(line, "Ciclo runtime HL | ")
        if cycle_payload is not None:
            self._apply_cycle_payload(cycle_payload)
            return

        smoke_payload = _extract_json_marker(line, "Smoke test Hyperliquid | ")
        if smoke_payload is not None:
            self._apply_smoke_payload(smoke_payload)
            return

        close_payload = _extract_json_marker(line, "Fechamento manual Hyperliquid | ")
        if close_payload is not None:
            self._apply_manual_close_payload(close_payload)
            return

        display_line = _strip_log_prefix(line)
        level = "error" if "erro" in display_line.lower() else "info"
        self._append_log(display_line, level=level, detail=line)
        if source == "run" and "Iniciando pipeline com comando: run" in display_line:
            self.runtime_state_label.setText("Bot rodando")

    def _apply_status_payload(self, payload: dict[str, Any]) -> None:
        self.balance_card.update_card(
            _currency(payload.get("available_to_trade", 0.0)),
            subtitle=f"Fonte: {payload.get('available_to_trade_source_field', '--')}",
            accent="#4ade80" if payload.get("connected") else "#fb7185",
        )
        mode_text = f"{payload.get('network', '--')} | {payload.get('execution_mode', '--')}"
        self.mode_card.update_card(
            self._current_mode().label,
            subtitle=mode_text,
            accent=self._current_mode().pill_color,
        )
        self.runtime_state_label.setText(
            "Conectado" if payload.get("connected") and payload.get("can_trade") else "Pronto para validacao"
        )
        self._append_log(
            f"Check concluido. Saldo disponivel: {_currency(payload.get('available_to_trade', 0.0))}.",
            level="success" if payload.get("connected") else "error",
            detail=json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _apply_cycle_payload(self, payload: dict[str, Any]) -> None:
        position_label = str(payload.get("position_label", "FLAT"))
        pnl_value = float(payload.get("position_unrealized_pnl", 0.0) or 0.0)
        accent = "#4ade80" if position_label == "LONG" else ("#fb7185" if position_label == "SHORT" else "#cbd5e1")

        self.position_badge.setText(position_label)
        self.position_badge.setStyleSheet(
            "background: %s; color: #f8fafc; border-radius: 13px; padding: 6px 12px; font-weight: 700; min-width: 74px;"
            % ("#166534" if position_label == "LONG" else "#9f1239" if position_label == "SHORT" else "#22303f")
        )
        self.position_value_label.setText(_currency(payload.get("adjusted_notional", payload.get("notional_value", 0.0))))
        self.position_value_label.setStyleSheet(f"color: {accent};")
        self.pnl_label.setText(_currency(pnl_value))
        self.pnl_label.setStyleSheet(f"color: {'#4ade80' if pnl_value >= 0 else '#fb7185'};")
        self.position_note_label.setText(
            "FLAT - aguardando proximo sinal."
            if not payload.get("position_is_open")
            else f"Posicao aberta ha {float(payload.get('position_time_in_bars', 0.0)):.0f} barras."
        )
        self.status_card.update_card(position_label, subtitle=f"Acao final: {float(payload.get('final_action', 0.0)):.2f}", accent=accent)
        self.entry_value.setText(_price(payload.get("position_avg_entry_price", 0.0)))
        self.tp_value.setText(_price(payload.get("take_profit_price", 0.0)))
        self.sl_value.setText(_price(payload.get("stop_loss_price", 0.0)))
        self.price_value.setText(_price(payload.get("reference_price", 0.0)))
        votes = payload.get("votes") or {}
        self.vote_value.setText(
            f"B:{int(votes.get('BUY', 0))} H:{int(votes.get('HOLD', 0))} S:{int(votes.get('SELL', 0))}"
        )
        self.regime_value.setText("Valido" if payload.get("regime_valid") else "Bloqueado")
        self.current_position_open = bool(payload.get("position_is_open", False))
        self.runtime_state_label.setText("Bot rodando")

        if payload.get("opened_trade"):
            message = (
                f"Abrindo {position_label} de {_currency(payload.get('adjusted_notional', 0.0))} | "
                f"TP: {_price(payload.get('take_profit_price', 0.0))} | "
                f"SL: {_price(payload.get('stop_loss_price', 0.0))}"
            )
            level = "success"
        elif payload.get("closed_trade"):
            message = f"Fechando posicao | motivo: {payload.get('exit_reason') or 'manual_close'}."
            level = "warning"
        elif payload.get("blocked_reason"):
            message = f"Entrada bloqueada | motivo: {payload.get('blocked_reason')}."
            level = "warning"
        elif payload.get("position_is_open"):
            message = f"Manutencao de posicao. Votacao Ensemble: {str(payload.get('vote_bucket', '--')).upper()}."
            level = "info"
        else:
            message = f"FLAT - aguardando proximo sinal. Votacao Ensemble: {str(payload.get('vote_bucket', '--')).upper()}."
            level = "info"

        self._append_log(message, level=level, detail=json.dumps(payload, ensure_ascii=False, indent=2))

    def _apply_smoke_payload(self, payload: dict[str, Any]) -> None:
        ok = bool(payload.get("ok"))
        opened = bool(payload.get("opened_position"))
        closed = bool(payload.get("closed_position"))
        message = (
            f"Smoke test {payload.get('network', '--')}: abriu={opened} | fechou={closed} | slippage={payload.get('smoke_slippage_used')}."
        )
        self._append_log(
            message,
            level="success" if ok else "error",
            detail=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        self.runtime_state_label.setText("Smoke validado" if ok else "Smoke com falha")

    def _apply_manual_close_payload(self, payload: dict[str, Any]) -> None:
        ok = bool(payload.get("ok"))
        self.current_position_open = False if ok else self.current_position_open
        self.status_card.update_card("FLAT", subtitle="Fechamento manual executado." if ok else "Nenhuma posicao fechada.", accent="#cbd5e1")
        self.position_badge.setText("FLAT")
        self.position_badge.setStyleSheet(
            "background: #22303f; color: #f8fafc; border-radius: 13px; padding: 6px 12px; font-weight: 700; min-width: 74px;"
        )
        self.position_note_label.setText("FLAT - aguardando proximo sinal.")
        self._append_log(
            "Kill switch executado com sucesso." if ok else "Kill switch nao encontrou posicao aberta.",
            level="success" if ok else "warning",
            detail=json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _append_log(self, message: str, level: str = "info", detail: str | None = None) -> None:
        colors = {
            "info": QColor("#dbe4f0"),
            "success": QColor("#86efac"),
            "warning": QColor("#facc15"),
            "error": QColor("#fda4af"),
        }
        prefixes = {
            "info": "[info]",
            "success": "[ok]",
            "warning": "[aviso]",
            "error": "[erro]",
        }
        item = QListWidgetItem(f"{prefixes.get(level, '[info]')} {message}")
        item.setForeground(colors.get(level, QColor("#dbe4f0")))
        if detail:
            item.setData(Qt.UserRole, detail)
        self.log_list.insertItem(0, item)
        while self.log_list.count() > 120:
            self.log_list.takeItem(self.log_list.count() - 1)

    def _show_log_detail(self, item: QListWidgetItem) -> None:
        detail = item.data(Qt.UserRole)
        if not detail:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Detalhes do log")
        box.setText(item.text())
        box.setDetailedText(str(detail))
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = TraderBotLauncher()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
