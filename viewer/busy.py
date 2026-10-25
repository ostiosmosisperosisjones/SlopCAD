"""
viewer/busy.py

Navigation-only input while heavy work runs (an async op, or a main-thread
replay waiting on kernel workers — see kernel_pool.ui_idle).

While active(): the viewport still gets mouse/wheel events (its handlers only
orbit / pan / zoom then), every other widget's input is dropped, and so are
shortcuts — an undo or a history click must not run mid-replay.
"""
from __future__ import annotations

from contextlib import contextmanager

from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtWidgets import QWidget

_depth = 0


def active() -> bool:
    return _depth > 0


def enter():
    global _depth
    _depth += 1


def leave():
    global _depth
    _depth = max(0, _depth - 1)


@contextmanager
def nav_only():
    enter()
    try:
        yield
    finally:
        leave()


_T = QEvent.Type
_VIEWPORT_OK = {_T.MouseButtonPress, _T.MouseButtonRelease, _T.MouseMove,
                _T.Wheel}
_INPUT = _VIEWPORT_OK | {_T.MouseButtonDblClick, _T.KeyPress, _T.KeyRelease,
                         _T.ContextMenu, _T.DragEnter, _T.DragMove, _T.Drop,
                         _T.TabletPress, _T.TabletMove, _T.TabletRelease,
                         _T.TouchBegin, _T.TouchUpdate, _T.TouchEnd}
_ALWAYS_BLOCK = {_T.Shortcut, _T.ShortcutOverride}


class NavOnlyFilter(QObject):
    """App-wide event filter; install once (main.py)."""

    def eventFilter(self, obj, ev):
        if not _depth:
            return False
        t = ev.type()
        if t in _ALWAYS_BLOCK:
            return True
        if t not in _INPUT or not isinstance(obj, QWidget):
            return False          # QWindow-level delivery: filtered at the widget
        from viewer.viewport import Viewport
        return not (t in _VIEWPORT_OK and isinstance(obj, Viewport))
