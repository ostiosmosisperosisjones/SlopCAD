"""
gui/params_panel.py

ParamsPanel — document-level named parameters (see cad/params.py).

Table:  Name | Expression | Value | Kind
  - edit Name / Expression in place; Kind via combo
  - "+" adds a parameter, right-click → Delete removes one
  - errors (bad syntax, cycles, unknown names) show in red under the table

All mutations go through the controller (the Viewport: param_set /
param_remove) so they replay the model and land on the undo stack.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QMenu, QAbstractItemView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from cad.history import History
from cad.params import Param, KINDS

_BG       = "#1e1e1e"
_TEXT     = "#d4d4d4"
_TEXT_DIM = "#777"
_ERR      = "#c05050"

_COL_NAME, _COL_EXPR, _COL_VALUE, _COL_KIND = range(4)


class ParamsPanel(QWidget):
    def __init__(self, history: History, parent=None):
        super().__init__(parent)
        self._history = history
        self.controller = None      # Viewport — set by mainwindow
        self._refreshing = False
        self._setup()
        self.refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup(self):
        self.setStyleSheet(f"background: {_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QWidget()
        head.setFixedHeight(prefs.scaled_px(28))
        head.setStyleSheet("background: #161616; border-bottom: 1px solid #2a2a2a;")
        hrow = QHBoxLayout(head)
        hrow.setContentsMargins(0, 0, 4, 0)
        title = QLabel("  Parameters")
        title.setStyleSheet(prefs.scale_stylesheet("""
            color: #555; font-size: 11px; font-weight: bold;
            letter-spacing: 1px; border: none;
        """))
        hrow.addWidget(title, 1)
        add = QPushButton("+")
        add.setToolTip("Add a parameter — use its name in any value field "
                       "or sketch dimension, e.g. width/2")
        add.setFixedSize(prefs.scaled_px(22), prefs.scaled_px(22))
        add.setStyleSheet(prefs.scale_stylesheet("""
            QPushButton { background: transparent; color: #777; border: none;
                          font-size: 15px; }
            QPushButton:hover { color: #d4d4d4; }
        """))
        add.clicked.connect(self._add)
        hrow.addWidget(add)
        root.addWidget(head)

        t = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(["Name", "Expression", "Value", "Kind"])
        t.verticalHeader().setVisible(False)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(_COL_NAME,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_EXPR,  QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(_COL_VALUE, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(_COL_KIND,  QHeaderView.ResizeMode.ResizeToContents)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                          | QAbstractItemView.EditTrigger.EditKeyPressed)
        t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        t.customContextMenuRequested.connect(self._context_menu)
        t.itemChanged.connect(self._on_item_changed)
        t.setStyleSheet(prefs.scale_stylesheet(f"""
            QTableWidget {{ background: {_BG}; color: {_TEXT}; border: none;
                           gridline-color: #2a2a2a; font-size: 11px; }}
            QHeaderView::section {{ background: #1a1a1a; color: #666;
                                   border: none; border-bottom: 1px solid #2a2a2a;
                                   padding: 2px 4px; font-size: 10px; }}
            QTableWidget::item:selected {{ background: #2f3f52; }}
            QLineEdit {{ background: #2a2a2a; color: {_TEXT};
                        font-family: monospace; }}
        """))
        self._table = t
        root.addWidget(t, 1)

        self._hint = QLabel(
            f"{prefs.key('new_parameter') or '+'} adds a parameter. Use its"
            " name in any value or dimension: w/2")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(prefs.scale_stylesheet(
            "color: #777; font-size: 11px; padding: 6px 8px;"))
        root.addWidget(self._hint)

        self._msg = QLabel("")
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet(prefs.scale_stylesheet(
            f"color: {_ERR}; font-size: 11px; padding: 3px 6px;"))
        self._msg.hide()
        root.addWidget(self._msg)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        from cad.units import format_value
        self._refreshing = True
        try:
            table = self._history.params
            values, errors = table.evaluate()
            t = self._table
            t.setRowCount(len(table.params))
            t.setVisible(bool(table.params))
            self._hint.setVisible(not table.params)
            for row, p in enumerate(table.params.values()):
                users = len(self._history.param_users({p.name}))
                name = QTableWidgetItem(p.name)
                name.setData(Qt.ItemDataRole.UserRole, p.name)
                tip = p.comment or ""
                tip += (f"\nUsed by {users} history entr{'y' if users == 1 else 'ies'}"
                        if users else "\nNot used yet")
                name.setToolTip(tip.strip())
                t.setItem(row, _COL_NAME, name)

                expr = QTableWidgetItem(p.expr)
                expr.setFont(_mono())
                t.setItem(row, _COL_EXPR, expr)

                if p.name in errors:
                    val = QTableWidgetItem("error")
                    val.setForeground(QColor(_ERR))
                    val.setToolTip(errors[p.name])
                    expr.setForeground(QColor(_ERR))
                    expr.setToolTip(errors[p.name])
                else:
                    v = values[p.name].value
                    if p.kind == "length":
                        txt = format_value(v, prefs.default_unit,
                                           prefs.display_decimals)
                    elif p.kind == "angle":
                        txt = f"{v:g}°"
                    else:
                        txt = f"{v:g}"
                    val = QTableWidgetItem(txt)
                    val.setForeground(QColor(_TEXT_DIM))
                val.setFlags(val.flags() & ~Qt.ItemFlag.ItemIsEditable)
                val.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                     | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(row, _COL_VALUE, val)

                kind = QComboBox()
                kind.addItems(KINDS)
                kind.setCurrentText(p.kind)
                kind.setStyleSheet(prefs.scale_stylesheet(
                    "QComboBox { background: transparent; color: #999; "
                    "border: none; font-size: 10px; }"))
                kind.currentTextChanged.connect(
                    lambda k, n=p.name: self._set_kind(n, k))
                t.setCellWidget(row, _COL_KIND, kind)
        finally:
            self._refreshing = False

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def _show_error(self, msg: str | None):
        self._msg.setText(msg or "")
        self._msg.setVisible(bool(msg))

    def _commit(self, param: Param, old_name: str | None) -> bool:
        if self.controller is None:
            return False
        err = self.controller.param_set(param, old_name)
        self._show_error(err)
        if err:
            self.refresh()      # revert the cell to the stored value
        return err is None

    def _add(self):
        names = self._history.params.params
        i = 1
        while f"p{i}" in names:
            i += 1
        self._commit(Param(f"p{i}", "10", unit=prefs.default_unit), None)
        row = self._table.rowCount() - 1
        if row >= 0:
            self._table.editItem(self._table.item(row, _COL_NAME))

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._refreshing:
            return
        row = item.row()
        name_item = self._table.item(row, _COL_NAME)
        old_name = name_item.data(Qt.ItemDataRole.UserRole)
        old = self._history.params.params.get(old_name)
        if old is None:
            return
        new_name = name_item.text().strip()
        expr_item = self._table.item(row, _COL_EXPR)
        expr = expr_item.text().strip() if expr_item else old.expr
        if new_name == old.name and expr == old.expr:
            return
        # Bare numbers in a newly typed expression mean the *current* unit.
        unit = prefs.default_unit if expr != old.expr else old.unit
        self._commit(Param(new_name, expr, old.kind, unit, old.comment), old_name)

    def _set_kind(self, name: str, kind: str):
        if self._refreshing:
            return
        old = self._history.params.params.get(name)
        if old is None or old.kind == kind:
            return
        self._commit(Param(old.name, old.expr, kind, old.unit, old.comment), name)

    def _context_menu(self, pos):
        item = self._table.itemAt(pos)
        if item is None or self.controller is None:
            return
        name = self._table.item(item.row(), _COL_NAME).data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        act = menu.addAction(f"Delete '{name}'")
        if menu.exec(self._table.viewport().mapToGlobal(pos)) is act:
            self._show_error(self.controller.param_remove(name))


def _mono():
    from PyQt6.QtGui import QFont
    f = QFont("monospace")
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f
