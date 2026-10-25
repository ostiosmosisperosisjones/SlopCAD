"""
gui/sidebar.py

Sidebar — vertical splitter:
  top    : PartsPanel   — body list with visibility toggles
  middle : ParamsPanel  — document parameters
  bottom : HistoryPanel — flat op history with rollback bar
"""

from __future__ import annotations
from PyQt6.QtWidgets import QSplitter
from PyQt6.QtCore import Qt, pyqtSignal

from gui.parts_panel import PartsPanel
from gui.history_panel import HistoryPanel
from gui.params_panel import ParamsPanel
from cad.workspace import Workspace
from cad.history import History


class Sidebar(QSplitter):
    seek_requested            = pyqtSignal(int)
    replay_requested          = pyqtSignal(int)
    body_visibility_changed   = pyqtSignal(str, bool)
    plane_visibility_changed  = pyqtSignal(str, bool)
    sketch_on_plane_requested = pyqtSignal(str)

    def __init__(self, workspace: Workspace, history: History, parent=None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self.setHandleWidth(2)
        self.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        self.parts_panel   = PartsPanel(workspace)
        self.params_panel  = ParamsPanel(history)
        self.history_panel = HistoryPanel(workspace, history)

        self.addWidget(self.parts_panel)
        self.addWidget(self.params_panel)
        self.addWidget(self.history_panel)
        self.setStretchFactor(0, 0)
        self.setStretchFactor(1, 0)
        self.setStretchFactor(2, 1)
        self.setSizes([180, 110, 420])

        # Wire parts → history (body selection dims other entries)
        self.parts_panel.body_selected.connect(
            self.history_panel.set_selected_body)

        # Body visibility: update history dimming + forward to viewport
        self.parts_panel.body_visibility_changed.connect(
            self._on_visibility_changed)

        # World-plane signals — forward straight up to mainwindow
        self.parts_panel.plane_visibility_changed.connect(
            self.plane_visibility_changed)
        self.parts_panel.sketch_on_plane_requested.connect(
            self.sketch_on_plane_requested)

        # Forward history signals upward to mainwindow
        self.history_panel.seek_requested.connect(self.seek_requested)
        self.history_panel.replay_requested.connect(self.replay_requested)

    def _on_visibility_changed(self, body_id: str, visible: bool):
        self.history_panel.set_body_hidden(body_id, not visible)
        self.body_visibility_changed.emit(body_id, visible)

    def refresh(self):
        self.parts_panel.refresh()
        self.params_panel.refresh()
        self.history_panel.refresh()
