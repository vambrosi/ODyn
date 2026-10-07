"""
The QC tab: what is waiting on the person using the app.

For now that is two lists, both drawn from the recorded calls and the
annotation registry: the recordings (loops, grabs, z-stacks) this user
submitted, with what their sync decode found, and the required annotations
still empty on them. Picking motion-correction candidates from movie previews
joins this tab once the cluster writer produces the previews
(`agents/cluster-writer.md`); the pane is shaped so that becomes a third list,
 not a new tab.
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

# The calls that add recordings, and the annotation target each adds
INGEST_METHODS = {"add_loop": "loop", "add_grabs": "grab", "add_zstacks": "zstack"}

# Where each kind's folder (a loop) or file (the others) is
PATH_COLUMN = {"loop": "loop_path", "grab": "raw_path", "zstack": "raw_path"}
COLUMNS = ["kind", "path", "objective", "submitted_at", "sync"]


def my_recordings(db, user: None | str) -> pd.DataFrame:
    """
    One row per loop, grab or z-stack `user` submitted, newest first, with its
    sync flags when it was decoded. Indexed by `(kind, id)`, as annotations
    name their target.
    """
    calls = db.method_calls

    if calls.empty:
        return pd.DataFrame(columns=COLUMNS)

    method = calls["method_name"].map(lambda m: str(m).rsplit(".", 1)[-1])
    mine = calls[method.isin(list(INGEST_METHODS)) & (calls["user"] == user)]
    added: dict[tuple[str, int], pd.Timestamp] = {}

    for call_id, call in mine.iterrows():
        kind = INGEST_METHODS[method[call_id]]
        output = call["call_output"]
        ids = output.get(f"{kind}_ids", []) if isinstance(output, dict) else []

        for row_id in ids:
            if isinstance(row_id, int) and (kind, row_id) not in added:
                added[(kind, row_id)] = call["called_at"]

    tables = {"loop": db.loops, "grab": db.grabs, "zstack": db.zstacks}
    flags = _loop_sync_flags(db) | _grab_sync_flags(db)
    rows = []

    for (kind, row_id), submitted in added.items():
        table = tables[kind]

        if row_id not in table.index:
            continue

        decoded = flags.get((kind, row_id))
        rows.append(
            {
                "kind": kind,
                "id": row_id,
                "path": table.loc[row_id, PATH_COLUMN[kind]],
                "objective": table.loc[row_id, "objective"],
                "submitted_at": submitted,
                "sync": decoded or ("" if kind == "zstack" else "not decoded"),
            }
        )

    frame = pd.DataFrame(rows, columns=["kind", "id", *COLUMNS[1:]])
    frame = frame.set_index(["kind", "id"], drop=False).drop(columns="id")

    return frame.sort_values("submitted_at", ascending=False)


def _loop_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each loop's latest `decode_sync`, decoded or refused."""
    decodes = db.latest_calls("decode_sync")
    flags: dict[tuple[str, int], str] = {}

    if decodes.empty or "input_loop_id" not in decodes:
        return flags

    for _, decode in decodes.iterrows():  # newest first
        loop_id = decode["input_loop_id"]

        if not pd.isna(loop_id) and ("loop", int(loop_id)) not in flags:
            flags[("loop", int(loop_id))] = sync_flags(int(decode["call_flag"]))

    return flags


def _grab_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each matched grab's decode: the call that wrote its `sync_blocks` row."""
    rows = db.from_query(
        "SELECT b.grab_id, c.call_flag FROM sync_blocks AS b "
        "JOIN method_calls AS c ON c.method_call_id = b.method_call_id "
        "WHERE b.grab_id IS NOT NULL;"
    )

    return {
        ("grab", int(row["grab_id"])): sync_flags(int(row["call_flag"]))
        for _, row in rows.iterrows()
    }


def sync_flags(call_flag: int) -> str:
    """The names of the bits set on a `decode_sync` call, or `clean`."""
    names = [flag.name for flag in SyncFlag if call_flag & int(flag) and flag.name]

    if call_flag & 1:
        names.insert(0, "RAISED")

    return ", ".join(names) or "clean"


def my_missing(db, mine: pd.DataFrame) -> pd.DataFrame:
    """The required annotations still empty on `mine`'s recordings."""
    missing = db.missing_annotations()

    if missing.empty or mine.empty:
        return missing.iloc[0:0]

    targets = set(mine.index)
    on_mine = [
        (kind, int(row_id)) in targets
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]

    return missing[on_mine]


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

        self.recordings_view = self._table("Experiments I submitted", splitter)
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
            mine = my_recordings(db, user)

            return mine, my_missing(db, mine)

        def loaded(result):
            mine, missing = result
            self._model(self.recordings_view).set_frame(mine.reset_index(drop=True))
            self._model(self.missing_view).set_frame(missing)

        self.context.run(load, loaded, "load QC")

    @staticmethod
    def _model(view: QTableView) -> FrameModel:
        model = view.model()
        assert isinstance(model, FrameModel)

        return model
