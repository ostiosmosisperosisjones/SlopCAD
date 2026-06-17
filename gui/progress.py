"""
gui/progress.py

ProgressController — one program-wide indicator for heavy OCCT ops + checks.

Heavy kernel work in this app runs off the UI thread (worker threads, and for
fillet a killable child process). Several of these take seconds — fillet build,
the multi-process "fit largest" search, tessellation/validation — and used to be
silent, so the app looked frozen ("is it stuck?"). This routes all of them
through a single status-bar indicator (spinner + phase text, plus a Cancel
button shown when the running op is cancelable).

API (call from anywhere; the controller marshals to the Qt thread itself):

    pc.begin("Fillet", modal=True, cancelable=True)
    pc.phase("validating mesh")          # or pc.step(3, 8) for k/N progress
    ...
    pc.end()

For work running on another thread/process, pass `pc.report` (a plain callable,
no Qt) down as a `progress` callback, and `pc.is_canceled` as a `should_cancel`
predicate. Both are thread-safe and Qt-free so kernel code can call them.
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QLabel, QPushButton


# Braille spinner frames — cheap, readable in a status bar.
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ProgressController(QObject):
    """Owned by the main window. Thread-safe entry points marshal onto the Qt
    thread via queued signals, so kernel/worker code can report freely."""

    # internal: (text, modal, cancelable, active)
    _state_changed = pyqtSignal(str, bool, bool, bool)

    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window
        self._lock = threading.Lock()
        self._canceled = False
        self._active = False
        self._name = ""
        self._sb_label: QLabel | None = None
        self._cancel_btn: QPushButton | None = None
        self._spin_i = 0

        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)

        self._state_changed.connect(
            self._apply_state, Qt.ConnectionType.QueuedConnection)

    # ---- public, thread-safe ----------------------------------------------

    def begin(self, name: str, modal: bool = True, cancelable: bool = False):
        with self._lock:
            self._canceled = False
            self._active = True
            self._name = name
        self._emit(name, modal, cancelable)

    def phase(self, text: str):
        """Replace the detail line, e.g. 'validating mesh'."""
        with self._lock:
            if not self._active:
                return
            name = self._name
        msg = f"{name}: {text}" if text else name
        self._emit(msg, _MODAL_KEEP, _CANCEL_KEEP)

    def step(self, i: int, n: int, text: str = ""):
        detail = f"{text} ({i}/{n})" if text else f"{i}/{n}"
        self.phase(detail)

    def end(self):
        with self._lock:
            self._active = False
        self._emit("", False, False, active=False)

    def cancel(self):
        with self._lock:
            self._canceled = True
        # reflect immediately so the user sees the click register
        self.phase("canceling…")

    # callbacks to hand to worker threads / kernel code (Qt-free) ------------

    def report(self, text: str = "", i: int | None = None, n: int | None = None):
        if i is not None and n is not None:
            self.step(i, n, text)
        else:
            self.phase(text)

    def is_canceled(self) -> bool:
        with self._lock:
            return self._canceled

    # ---- internal ---------------------------------------------------------

    def _emit(self, msg, modal, cancelable, active=True):
        # _MODAL_KEEP / _CANCEL_KEEP are sentinels meaning "leave as-is"; resolve
        # them on the Qt side where we hold the live widgets.
        self._state_changed.emit(
            msg,
            modal if modal is not _MODAL_KEEP else _last_modal[0],
            cancelable if cancelable is not _CANCEL_KEEP else _last_cancel[0],
            active,
        )
        if modal is not _MODAL_KEEP:
            _last_modal[0] = modal
        if cancelable is not _CANCEL_KEEP:
            _last_cancel[0] = cancelable

    def _apply_state(self, msg: str, modal: bool, cancelable: bool, active: bool):
        self._ensure_widgets()
        if not active:
            self._timer.stop()
            self._sb_label.setText("")
            self._cancel_btn.setVisible(False)
            return

        if not self._timer.isActive():
            self._spin_i = 0
            self._timer.start()
        self._cur_msg = msg
        self._refresh_text()
        self._cancel_btn.setVisible(bool(cancelable))

    def _tick(self):
        self._spin_i = (self._spin_i + 1) % len(_SPIN)
        self._refresh_text()

    def _refresh_text(self):
        frame = _SPIN[self._spin_i]
        msg = getattr(self, "_cur_msg", "")
        self._sb_label.setText(f"{frame}  {msg}")

    def _ensure_widgets(self):
        # Single indicator, in the status bar only (no on-screen overlay — it
        # duplicated this line). Cancel sits beside the text, shown on demand.
        if self._sb_label is not None:
            return
        self._sb_label = QLabel("")
        self._sb_label.setStyleSheet(
            "color: #ffca28; padding-left: 10px; font-weight: bold;")
        self._mw.statusBar().addWidget(self._sb_label)
        btn = QPushButton("Cancel")
        btn.setStyleSheet(
            "QPushButton { color:#fff; background:#3a3a3a; border:1px solid #555;"
            " border-radius:3px; padding:1px 8px; font-size:10px; }"
            "QPushButton:hover { background:#4a4a4a; }")
        btn.clicked.connect(self.cancel)
        btn.setVisible(False)
        self._mw.statusBar().addWidget(btn)
        self._cancel_btn = btn


# Sentinels for "leave this attribute unchanged" across phase() updates, plus
# the last-known values they resolve against on the Qt side.
_MODAL_KEEP = object()
_CANCEL_KEEP = object()
_last_modal = [True]
_last_cancel = [False]
