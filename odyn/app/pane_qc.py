"""
The QC tab: what is waiting on the person using the app.

For now that is two lists, both drawn from the recorded calls and the
annotation registry: the recordings (loops, grabs, z-stacks) this user
submitted, with what processing their sync file found, and the required
annotations still empty on them. Picking motion-correction candidates from
movie previews joins this tab once the cluster writer produces the previews
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

from datetime import datetime

from ..cluster import ClusterConfig, heartbeat, pending_requests
from ..database import SyncFlag
from .context import Context
from .models import FrameModel, fit_columns
from .style import text_style

# The calls that add recordings, and the annotation target each adds
INGEST_METHODS = {"add_loop": "loop", "add_grabs": "grab", "add_zstacks": "zstack"}

# Where each kind's folder (a loop) or file (the others) is
PATH_COLUMN = {"loop": "loop_path", "grab": "raw_path", "zstack": "raw_path"}
COLUMNS = ["kind", "path", "objective", "submitted_at", "sync"]


def cluster_line(db, now: None | datetime = None) -> tuple[str, str]:
    """
    When the cluster last ran its tick, as `(text, kind)` for the QC page. The
    kind is `warning` when the chain of ticks looks stopped (no tick for twice
    its interval), else `computed`.
    """
    beat = heartbeat(db.main_folder, db.project)
    waiting = len(pending_requests(db.main_folder, db.project))
    plural = "" if waiting == 1 else "s"
    tail = "" if not waiting else f", {waiting} request{plural} waiting"

    if beat is None:
        return f"The cluster has not run yet{tail}", "computed"

    started = datetime.strptime(beat["started_at"], "%Y-%m-%d %H:%M:%S")
    minutes = ((now or datetime.now()) - started).total_seconds() / 60
    every = ClusterConfig.read(db.main_folder).every_minutes
    ago = f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.0f} h"
    text = f"The cluster last ran at {beat['started_at'][:16]} ({ago} ago){tail}"

    if minutes > 2 * every:
        return f"{text}, so its chain of ticks may have stopped", "warning"

    return text, "computed"


def my_recordings(db, user: None | str) -> pd.DataFrame:
    """
    One row per loop, grab or z-stack `user` submitted, newest first, with its
    sync flags once its sync file was processed. Indexed by `(kind, id)`, as
    annotations name their target.
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

        processed = flags.get((kind, row_id))
        rows.append(
            {
                "kind": kind,
                "id": row_id,
                "path": table.loc[row_id, PATH_COLUMN[kind]],
                "objective": table.loc[row_id, "objective"],
                "submitted_at": submitted,
                "sync": processed or ("" if kind == "zstack" else "not processed"),
            }
        )

    frame = pd.DataFrame(rows, columns=["kind", "id", *COLUMNS[1:]])
    frame = frame.set_index(["kind", "id"], drop=False).drop(columns="id")

    # Newest first: the rows are in the order of the calls (and of the ids
    # within one), so reversed they break ties of `submitted_at` the same way
    return frame.iloc[::-1].sort_values("submitted_at", ascending=False, kind="stable")


def _loop_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each loop's latest `process_sync`, decoded or refused."""
    calls = db.latest_calls("process_sync")
    flags: dict[tuple[str, int], str] = {}

    if calls.empty or "input_loop_id" not in calls:
        return flags

    for _, call in calls.iterrows():  # newest first
        loop_id = call["input_loop_id"]

        if not pd.isna(loop_id) and ("loop", int(loop_id)) not in flags:
            flags[("loop", int(loop_id))] = sync_flags(int(call["call_flag"]))

    return flags


def _grab_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each matched grab's processing: the call that wrote its `sync_blocks` row."""
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
    """The names of the bits set on a `process_sync` call, or `succeeded`."""
    names = [flag.name for flag in SyncFlag if call_flag & int(flag) and flag.name]

    if call_flag & 1:
        names.insert(0, "FAILED")

    return ", ".join(names) or "succeeded"


def my_missing(db, mine: pd.DataFrame) -> pd.DataFrame:
    """
    The required annotations still empty on `mine`'s recordings, in the order
    of `mine` (newest first).
    """
    missing = db.missing_annotations()

    if missing.empty or mine.empty:
        return missing.iloc[0:0]

    rank = {target: position for position, target in enumerate(mine.index)}
    order = [
        rank.get((kind, int(row_id)))
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]
    missing = missing.assign(_rank=order).dropna(subset="_rank")

    return missing.sort_values("_rank", kind="stable").drop(columns="_rank")


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
        self.cluster = QLabel("")
        layout.addWidget(self.cluster)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        self.recordings_view = self._table("Submissions", splitter)
        self.missing_view = self._table(
            "Required annotations that were not filled", splitter
        )
        splitter.addWidget(QLabel("Motion-correction (pick the best test run)"))

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
        if not user:
            self.who.setText("Pick a user to start")

        def load(db):
            mine = my_recordings(db, user)
            users = db.users
            named = users.loc[users["user_name"] == user, "display_name"]
            name = str(named.iloc[0]) if len(named) else user

            return name, mine, my_missing(db, mine), cluster_line(db)

        def loaded(result):
            name, mine, missing, (cluster, kind) = result

            if user:
                self.who.setText(f"To-do list for {name}")

            self.cluster.setText(cluster)
            self.cluster.setStyleSheet(text_style(kind))

            self._model(self.recordings_view).set_frame(mine.reset_index(drop=True))
            self._model(self.missing_view).set_frame(missing)

        self.context.run(load, loaded, "load QC")

    @staticmethod
    def _model(view: QTableView) -> FrameModel:
        model = view.model()
        assert isinstance(model, FrameModel)

        return model
