"""
The QC tab: what is waiting on the person using the app.

For now that is two lists, both drawn from the recorded calls and the
annotation registry: the experiments this user submitted (with what their
sync decode found) and the required annotations still empty on them. Picking
motion-correction candidates from movie previews joins this tab once the
cluster writer produces the previews (`agents/cluster-writer.md`); the pane
is shaped so that becomes a third list, not a new tab.
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Qt

from ..database import SyncFlag
from .context import Context
from .models import FrameModel, fit_columns

INGEST_METHODS = ("add_experiment", "add_grabs")


def my_experiments(db, user: None | str) -> pd.DataFrame:
    """
    One row per experiment `user` submitted, newest first, with its sync
    flags when it was decoded.
    """
    experiments = db.experiments
    calls = db.method_calls

    if experiments.empty or calls.empty:
        return pd.DataFrame(columns=["exp_path", "exp_type", "submitted_at", "sync"])

    mine = calls[
        calls["method_name"].map(lambda m: str(m).rsplit(".", 1)[-1] in INGEST_METHODS)
        & (calls["user"] == user)
    ]
    ids: dict[int, pd.Timestamp] = {}

    for _, call in mine.iterrows():
        output = call["call_output"]
        added = output.get("exp_ids", []) if isinstance(output, dict) else []

        for exp_id in added:
            if isinstance(exp_id, int) and exp_id not in ids:
                ids[exp_id] = call["called_at"]

    rows = experiments.loc[[i for i in ids if i in experiments.index]].copy()
    rows["submitted_at"] = [ids[i] for i in rows.index]

    decodes = db.latest_calls("decode_sync")
    flags: dict[int, str] = {}

    if not decodes.empty and "input_exp_id" in decodes:
        for _, decode in decodes.iterrows():
            exp_id = decode["input_exp_id"]

            if pd.isna(exp_id) or int(exp_id) in flags:
                continue

            flags[int(exp_id)] = sync_flags(int(decode["call_flag"]))

    rows["sync"] = [flags.get(int(i), "not decoded") for i in rows.index]

    return rows[
        ["exp_path", "exp_type", "objective", "submitted_at", "sync"]
    ].sort_values("submitted_at", ascending=False)


def sync_flags(call_flag: int) -> str:
    """The names of the bits set on a `decode_sync` call, or `clean`."""
    names = [flag.name for flag in SyncFlag if call_flag & int(flag) and flag.name]

    if call_flag & 1:
        names.insert(0, "RAISED")

    return ", ".join(names) or "clean"


def my_missing(db, mine: pd.DataFrame) -> pd.DataFrame:
    """The required annotations still empty on `mine`'s experiments."""
    missing = db.missing_annotations()

    if missing.empty or mine.empty:
        return missing

    on_experiments = (missing["target_type"] == "experiment") & missing[
        "target_id"
    ].isin(mine.index)

    return missing[on_experiments]


class QcPane(QWidget):
    """What is waiting on the current user."""

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.who = QLabel("")
        reload = QPushButton("Reload")
        reload.clicked.connect(self.reload)
        top.addWidget(self.who, stretch=1)
        top.addWidget(reload)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        self.experiments_view = self._table("Experiments I submitted", splitter)
        self.missing_view = self._table(
            "Required annotations still empty on them", splitter
        )
        splitter.addWidget(
            QLabel(
                "Motion-correction candidate picking (movie previews) lands here "
                "once the cluster writer produces them."
            )
        )

        context.changed.connect(self.reload)
        context.user_changed.connect(lambda _: self.reload())
        self.reload()

    def _table(self, title: str, splitter: QSplitter) -> QTableView:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(QLabel(title))
        view = QTableView()
        view.setModel(FrameModel())
        view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        view.setAlternatingRowColors(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        fit_columns(view)
        box_layout.addWidget(view)
        splitter.addWidget(box)

        return view

    def reload(self) -> None:
        user = self.context.user
        self.who.setText(
            f"Pending for {user}" if user else "Pick your name in the Users tab"
        )

        def load(db):
            mine = my_experiments(db, user)

            return mine, my_missing(db, mine)

        def loaded(result):
            mine, missing = result
            self._model(self.experiments_view).set_frame(mine)
            self._model(self.missing_view).set_frame(missing)

        self.context.run(load, loaded, "load QC")

    @staticmethod
    def _model(view: QTableView) -> FrameModel:
        model = view.model()
        assert isinstance(model, FrameModel)

        return model
