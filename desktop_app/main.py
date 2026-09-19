"""OserAI Trainer -- desktop contributor app.

A friendly front-end for agent/train_agent.py: fill in a coordinator URL
and a run ID, hit Start, and this machine becomes a DiLoCo training node
-- same BOINC/Folding@home-style "install once, contribute compute in the
background" shape. All the actual training logic lives in
agent/train_agent.py and training/*; desktop_app/training_worker.py just
drives it from a Qt thread.

Run directly:
    python -m desktop_app.main

Package into a single executable (see desktop_app/build.spec):
    pyinstaller desktop_app/build.spec
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.train_config import TrainAgentConfig  # noqa: E402
from desktop_app.training_worker import TrainingWorker, available_devices  # noqa: E402

SETTINGS_DIR = Path.home() / ".oserai_trainer"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_settings(settings: dict) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def make_tray_icon() -> QIcon:
    # build.spec bundles assets/logo_tray.png next to the frozen executable
    # (see its `datas` entry); sys._MEIPASS is where PyInstaller unpacks
    # that at runtime. Running from source, it's just next to this file.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return QIcon(str(base / "assets" / "logo_tray.png"))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OserAI Trainer")
        self.worker: TrainingWorker | None = None

        self._build_ui()
        self._build_tray()
        self._restore_settings()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        form = QFormLayout()
        self.coordinator_url_edit = QLineEdit("http://localhost:8000")
        self.node_name_edit = QLineEdit(socket.gethostname())
        self.run_id_edit = QLineEdit()
        self.run_id_edit.setPlaceholderText("paste a training run ID from the coordinator")
        self.shard_index_spin = QSpinBox()
        self.shard_index_spin.setRange(0, 999)
        self.num_shards_spin = QSpinBox()
        self.num_shards_spin.setRange(1, 999)
        self.num_shards_spin.setValue(1)
        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())

        form.addRow("Coordinator URL", self.coordinator_url_edit)
        form.addRow("Node name", self.node_name_edit)
        form.addRow("Run ID", self.run_id_edit)
        form.addRow("Shard index", self.shard_index_spin)
        form.addRow("Num shards", self.num_shards_spin)
        form.addRow("Device", self.device_combo)
        layout.addLayout(form)

        self.form_widgets = [
            self.coordinator_url_edit, self.node_name_edit, self.run_id_edit,
            self.shard_index_spin, self.num_shards_spin, self.device_combo,
        ]

        btn_row = QHBoxLayout()
        self.start_button = QPushButton("Start contributing")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Not running")
        self.status_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.status_label)

        layout.addWidget(QLabel("Activity log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view)

        self.setCentralWidget(central)
        self.resize(560, 520)

    def _build_tray(self) -> None:
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QMenu

        self.tray = QSystemTrayIcon(make_tray_icon(), self)
        self.tray.setToolTip("OserAI Trainer")
        menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self._show_and_raise)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(show_action)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._show_and_raise()

    def _show_and_raise(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _restore_settings(self) -> None:
        settings = load_settings()
        if settings.get("coordinator_url"):
            self.coordinator_url_edit.setText(settings["coordinator_url"])
        if settings.get("node_name"):
            self.node_name_edit.setText(settings["node_name"])
        if settings.get("run_id"):
            self.run_id_edit.setText(settings["run_id"])
        if "shard_index" in settings:
            self.shard_index_spin.setValue(settings["shard_index"])
        if "num_shards" in settings:
            self.num_shards_spin.setValue(settings["num_shards"])
        if settings.get("device") and settings["device"] in available_devices():
            self.device_combo.setCurrentText(settings["device"])

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in self.form_widgets:
            widget.setEnabled(enabled)

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def start_training(self) -> None:
        settings = {
            "coordinator_url": self.coordinator_url_edit.text().strip(),
            "node_name": self.node_name_edit.text().strip(),
            "run_id": self.run_id_edit.text().strip(),
            "shard_index": self.shard_index_spin.value(),
            "num_shards": self.num_shards_spin.value(),
            "device": self.device_combo.currentText(),
        }
        if not settings["coordinator_url"] or not settings["node_name"] or not settings["run_id"]:
            QMessageBox.warning(self, "Missing info", "Coordinator URL, node name, and run ID are all required.")
            return
        save_settings(settings)

        config = TrainAgentConfig(
            node_name=settings["node_name"], coordinator_url=settings["coordinator_url"],
            run_id=settings["run_id"], shard_index=settings["shard_index"],
            num_shards=settings["num_shards"], device=settings["device"],
            state_path=str(SETTINGS_DIR / f"{settings['node_name']}_node_state.json"),
        )

        self.worker = TrainingWorker(config)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.status_changed.connect(self._append_log)
        self.worker.log_message.connect(self._append_log)
        self.worker.round_completed.connect(self._on_round_completed)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_form_enabled(False)
        self._append_log(f"Starting as {settings['node_name']} against {settings['coordinator_url']}...")

    def stop_training(self) -> None:
        if self.worker is not None:
            self.status_label.setText("Stopping...")
            self.worker.stop()
        self.stop_button.setEnabled(False)

    def _on_worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_form_enabled(True)

    def _on_round_completed(self, round_num: int, loss: float) -> None:
        loss_text = f"{loss:.4f}" if loss == loss else "n/a"  # loss != loss for NaN
        self._append_log(f"Round {round_num} complete -- local loss {loss_text}")

    def _on_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}")
        QMessageBox.critical(self, "Training error", message)

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        # Minimize to tray instead of quitting -- a contributor closing the
        # window shouldn't interrupt a round in progress. Matches
        # BOINC/Folding@home-style background volunteer-compute apps.
        event.ignore()
        self.hide()
        if self.worker is not None and self.worker.isRunning():
            self.tray.showMessage(
                "OserAI Trainer", "Still contributing in the background.",
                QSystemTrayIcon.MessageIcon.Information, 3000,
            )

    def _quit(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        QApplication.instance().quit()


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
