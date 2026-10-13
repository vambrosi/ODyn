"""
The QC tab: what is waiting on the person using the app.

Three lists. The drafts this user submitted that the cluster has not added
yet (`submissions.queued`), with how far their copy got or why they failed,
and what can be done about each: add it now, submit it again, or take it
back to edit. The recordings (experiments, grabs, z-stacks) the cluster added for
this user, with what processing their sync file found, and a way to process a
experiment's again. And the required annotations still empty on them. Picking
motion-correction candidates from movie previews joins this tab once the
cluster produces the previews; the pane is shaped so that becomes one more
list, not a new tab.
"""

from __future__ import annotations

from pathlib import Path

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

from PySide6.QtCore import Qt, Signal

from datetime import datetime

from ..cluster import ClusterConfig, heartbeat, pending_requests, send_request
from ..database import SyncFlag
from .context import Context
from .draft import Draft
from .models import FrameModel, fit_columns
from .style import text_style
from .submissions import add_now, queued, submit_again, take_back

# The calls that add recordings, the kind each adds, and the output key listing
# the ids it added
INGEST_METHODS = {
    "add_experiment": ("experiment", "exp_ids"),
    "add_grabs": ("grab", "grab_ids"),
    "add_zstacks": ("zstack", "zstack_ids"),
}

# Where each kind's folder (an experiment) or file (the others) is
PATH_COLUMN = {"experiment": "exp_path", "grab": "raw_path", "zstack": "raw_path"}
COLUMNS = ["kind", "path", "objective", "submitted_at", "sync"]
QUEUE_COLUMNS = ["session", "state", "progress", "submitted_at"]


def my_queue(main_folder: Path, project: None | str, user: str) -> pd.DataFrame:
    """
    The drafts `user` submitted that the cluster has not added yet, oldest
    first: the session, `waiting` or `failed`, how far the copy got (or why
    it failed), and when it was submitted. `path` is the queued file.
    """
    rows = []

    for path, state in queued(main_folder, project):
        try:
            draft = Draft.load(path)
        except (OSError, ValueError):
            continue

        if draft.data.get("submitted_by") != user:
            continue

        rows.append(
            {
                "session": draft.where,
                "state": state.state,
                "progress": state.message or "not looked at yet",
                "submitted_at": draft.data.get("submitted_at", ""),
                "path": str(path),
            }
        )

    return pd.DataFrame(rows, columns=[*QUEUE_COLUMNS, "path"])


def cluster_line(db, now: None | datetime = None) -> tuple[str, str]:
    """
    When the cluster last ran its tick, as `(text, kind)` for the QC page. The
    kind is `warning` when the cluster looks stopped (no tick for twice
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
        return f"{text}; it may have stopped", "warning"

    return text, "computed"


def my_recordings(db, user: None | str) -> pd.DataFrame:
    """
    One row per experiment, grab or z-stack `user` submitted, newest first, with its
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
        kind, key = INGEST_METHODS[method[call_id]]
        output = call["call_output"]
        ids = output.get(key, []) if isinstance(output, dict) else []

        for row_id in ids:
            if isinstance(row_id, int) and (kind, row_id) not in added:
                added[(kind, row_id)] = call["called_at"]

    tables = {"experiment": db.experiments, "grab": db.grabs, "zstack": db.zstacks}
    flags = _exp_sync_flags(db) | _grab_sync_flags(db)
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


def _exp_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each experiment's latest `process_sync`, decoded or refused."""
    calls = db.latest_calls("process_sync")
    flags: dict[tuple[str, int], str] = {}

    if calls.empty or "input_exp_id" not in calls:
        return flags

    for _, call in calls.iterrows():  # newest first
        exp_id = call["input_exp_id"]

        if not pd.isna(exp_id) and ("experiment", int(exp_id)) not in flags:
            flags[("experiment", int(exp_id))] = sync_flags(int(call["call_flag"]))

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
    """
    What is waiting on the current user. `restored` gives the path of a draft
    taken back from the cluster, now among this computer's drafts.
    """

    restored = Signal(object)

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.queue = pd.DataFrame(columns=[*QUEUE_COLUMNS, "path"])
        self.mine = pd.DataFrame(columns=COLUMNS)
        self.mine_ids: list[int] = []  # each row's experiment, grab or z-stack id

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

        self.add_now = QPushButton("Add now")
        self.add_now.setStatusTip(
            "Add it at the cluster's next run, without waiting for its TIFFs"
        )
        self.add_now.clicked.connect(self._add_now)
        self.submit_again = QPushButton("Submit again")
        self.submit_again.setStatusTip(
            "Look at it again, once what stopped it is fixed"
        )
        self.submit_again.clicked.connect(self._submit_again)
        self.edit_draft = QPushButton("Edit draft")
        self.edit_draft.setStatusTip("Take it back from the cluster to edit it here")
        self.edit_draft.clicked.connect(self._edit_draft)
        self.queue_view = self._table(
            "Waiting for the cluster",
            splitter,
            [self.add_now, self.submit_again, self.edit_draft],
        )

        self.rerun_sync = QPushButton("Rerun sync")
        self.rerun_sync.setStatusTip(
            "Process the experiment's sync file again at the cluster's next run"
        )
        self.rerun_sync.clicked.connect(self._rerun_sync)
        self.recordings_view = self._table(
            "Added to the database", splitter, [self.rerun_sync]
        )
        self.missing_view = self._table(
            "Required annotations that were not filled", splitter
        )
        splitter.addWidget(QLabel("Motion-correction (pick the best test run)"))

        for view in (self.queue_view, self.recordings_view):
            view.selectionModel().selectionChanged.connect(self._enable)

        context.changed.connect(self.reload)
        context.user_changed.connect(lambda _: self.reload())
        self._enable()
        self.reload()

    def _table(
        self, title: str, splitter: QSplitter, buttons: list[QPushButton] = ()
    ) -> QTableView:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.addWidget(QLabel(title))
        view = QTableView()
        view.setModel(FrameModel())
        view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        view.setAlternatingRowColors(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        fit_columns(view)
        box_layout.addWidget(view)

        if buttons:
            row = QHBoxLayout()
            row.addStretch(1)

            for button in buttons:
                row.addWidget(button)

            box_layout.addLayout(row)

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
            queue = my_queue(db.main_folder, db.project, user)

            return name, queue, mine, my_missing(db, mine), cluster_line(db)

        def loaded(result):
            name, queue, mine, missing, (cluster, kind) = result

            if user:
                self.who.setText(f"To-do list for {name}")

            self.cluster.setText(cluster)
            self.cluster.setStyleSheet(text_style(kind))

            self.queue = queue.reset_index(drop=True)
            self.mine = mine.reset_index(drop=True)
            self.mine_ids = [int(row_id) for _, row_id in mine.index]
            self._model(self.queue_view).set_frame(self.queue[QUEUE_COLUMNS])
            self._model(self.recordings_view).set_frame(self.mine)
            self._model(self.missing_view).set_frame(missing)
            self._enable()

        self.context.run(load, loaded, "load QC")

    # ----------------------------------------------------------------------- #
    # Actions
    # ----------------------------------------------------------------------- #

    def _selected(self, view: QTableView, frame: pd.DataFrame) -> None | pd.Series:
        rows = view.selectionModel().selectedRows()

        if not rows or rows[0].row() >= len(frame):
            return None

        return frame.iloc[rows[0].row()]

    def _enable(self, *_) -> None:
        queued_row = self._selected(self.queue_view, self.queue)
        state = None if queued_row is None else queued_row["state"]
        self.add_now.setEnabled(state is not None)
        self.submit_again.setEnabled(state == "failed")
        self.edit_draft.setEnabled(state is not None)

        added = self._selected(self.recordings_view, self.mine)
        self.rerun_sync.setEnabled(added is not None and added["kind"] == "experiment")

    def _add_now(self) -> None:
        row = self._selected(self.queue_view, self.queue)

        if row is not None:
            self._queue_action(
                lambda: add_now(Path(row["path"])),
                f"{row['session']} will be added at the cluster's next run",
            )

    def _submit_again(self) -> None:
        row = self._selected(self.queue_view, self.queue)

        if row is not None:
            self._queue_action(
                lambda: submit_again(Path(row["path"])),
                f"{row['session']} will be looked at again at the cluster's next run",
            )

    def _edit_draft(self) -> None:
        row = self._selected(self.queue_view, self.queue)

        if row is None or not self.context.confirm(
            "Edit draft",
            f"Take {row['session']} back from the cluster to edit it? Submit it "
            "again when it is ready",
            "Edit draft",
        ):
            return

        def done(local):
            self.context.status.emit(f"{row['session']} is back among the drafts")
            self.restored.emit(local)
            self.reload()

        drafts = self.context.drafts
        self.context.run(
            lambda db: take_back(Path(row["path"]), drafts), done, "edit draft"
        )

    def _rerun_sync(self) -> None:
        row = self._selected(self.recordings_view, self.mine)

        if row is None or row["kind"] != "experiment":
            return

        exp_id = self.mine_ids[int(row.name)]
        user, path = self.context.user, row["path"]

        def ask(db):
            send_request(
                db.main_folder,
                "process_sync",
                project=db.project,
                user=user,
                exp_id=exp_id,
            )

        self.context.run(
            ask,
            lambda _: self.context.status.emit(
                f"The sync file of {path} will be processed again at the cluster's "
                "next run"
            ),
            "rerun sync",
        )

    def _queue_action(self, action, message: str) -> None:
        def done(_):
            self.context.status.emit(message)
            self.reload()

        self.context.run(lambda db: action(), done, "submission")

    @staticmethod
    def _model(view: QTableView) -> FrameModel:
        model = view.model()
        assert isinstance(model, FrameModel)

        return model
