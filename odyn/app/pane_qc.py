"""
The QC page: what is waiting on the person using the app.

Four lists, a tab each, with how many rows each has in its name. The drafts
this user submitted that the cluster has not added yet
(`submissions.queued`), with how far their copy got or why they failed, and
what can be done about each: add it now, submit it again, or take it back to
edit. The recordings (experiments, z-stacks) the cluster added for
this user, with what processing their sync file found, and a way to process an
experiment's again. The required annotations still empty on them. And their
experiments' motion corrections, one row per experiment: its status (from
its test jobs: the cluster's own test and the runs of sweeps), a way to
download its test files to this computer (`cache`) and to open its folder and
its window (`mcor_window`) from there: played from the share, a video is
slow. Nothing is downloaded unless asked.

The page opens on the first tab with rows; once a tab is picked, reloads keep
it until the user changes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from datetime import datetime

from ..cluster import (
    ClusterConfig,
    heartbeat,
    latest_picks,
    mcor_sweep_runs,
    mcor_tests,
    pending_requests,
    pick_candidate,
    send_request,
)
from ..database import SyncFlag
from ..groups import FINAL_FOLDER, RAW_PREVIEW_FOLDER
from ..mcor_sweep import run_numbers
from . import cache
from .context import Context
from .draft import Draft
from .mcor_window import McorWindow, Run
from .models import FrameModel, fit_columns, sortable
from .style import text_style
from .submissions import add_now, queued, submit_again, take_back

# The calls that add recordings, the kind each adds, and the output key listing
# the ids it added
INGEST_METHODS = {
    "add_experiment": ("experiment", "exp_ids"),
    "add_zstacks": ("zstack", "zstack_ids"),
}

# Where each kind's folder (an experiment) or file (the others) is
PATH_COLUMN = {"experiment": "exp_path", "zstack": "raw_path"}
COLUMNS = ["submitted_at", "kind", "path", "objective", "sync"]
QUEUE_COLUMNS = ["submitted_at", "session", "state", "progress"]
TEST_COLUMNS = [
    "submitted_at",
    "path",
    "run",
    "state",
    "local",
    "corr_with_ref",
    "flow_px",
    "jitter_px",
    "at_limit",
    "finished_at",
]

TEST_HIDDEN = ["folder", "bytes", "group_id", "experiment_folder", "pick_call_id"]
EXPERIMENT_COLUMNS = ["submitted_at", "path", "status", "runs", "local copy", "size"]
EXPERIMENT_HIDDEN = ["group_id", "folder", "folders", "bytes", "finished"]

# An experiment's status, from its tests (later steps add their own)
WAITING = "waiting for the cluster"
STOPPED = "stopped"
READY = "ready to compare"
CORRECTING = "correcting with {run}"
CORRECTION_STOPPED = "correction with {run} stopped"
FINAL_CHECK = "ready for final check"

# An experiment some of whose files are downloaded (see `cache.state`)
PARTLY = "partly"
TO_DOWNLOAD = (cache.NOT_HERE, PARTLY, cache.NEWER)

# The numbers shown for a test, from `mcor_sweep.run_numbers`: medians over
# the acquisitions, or the whole test's
TEST_NUMBERS = {
    "corr_with_ref": "test_corr_with_ref_median",
    "flow_px": "test_flow_median_px_median",
    "jitter_px": "patch_jitter_median_px",
    "at_limit": "frames_at_limit",
}

# A job's state in words for the page ("unknown": no `squeue` here, so the
# job is either still in the cluster's queue or ended without finishing)
TEST_STATES = {"pending": "waiting", "unknown": "not finished"}


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
    One row per experiment or z-stack `user` submitted, newest first, with its
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

    tables = {"experiment": db.experiments, "zstack": db.zstacks}
    flags = _exp_sync_flags(db)
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

    frame = pd.DataFrame(rows, columns=["id", *COLUMNS])
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


def sync_flags(call_flag: int) -> str:
    """The names of the bits set on a `process_sync` call, or `succeeded`."""
    names = [flag.name for flag in SyncFlag if call_flag & int(flag) and flag.name]

    if call_flag & 1:
        names.insert(0, "FAILED")

    return ", ".join(names) or "succeeded"


def my_tests(
    db, user: None | str, mine: pd.DataFrame, base: None | Path = None
) -> pd.DataFrame:
    """
    The test motion corrections of `user`'s experiments, the newest submitted
    first (as in `mine`, from `my_recordings`): the cluster's test of each
    ("test") and the runs of each sweep (the baseline first), with the job's
    state and, once a report is there, a few of its numbers and whether it
    is downloaded (`local`, see `cache.state`; `base` is where the cache
    is). `folder` is where the report is, `bytes` what downloading it copies,
    `experiment_folder` where the experiment's tests are.
    """
    jobs = [
        {**test, "run": None}
        for test in mcor_tests(db.main_folder, db.project)
        if test["user"] == user
    ] + [
        run
        for run in mcor_sweep_runs(db.main_folder, db.project)
        if run["user"] == user
    ]
    jobs.sort(key=lambda job: job["submitted_at"], reverse=True)  # stable
    submitted = {
        int(row_id): when
        for (kind, row_id), when in mine["submitted_at"].items()
        if kind == "experiment"
    }
    links = db.group_experiments
    paths = db.experiments["exp_path"]
    rows = []

    for job in jobs:
        group_id = int(job["group_id"])

        if group_id not in links.index:
            continue

        exp_ids = links.loc[[group_id], "exp_id"].astype(int)
        folder = db.groups[group_id].mcor_test_folder(job["run"])
        times = [submitted[i] for i in exp_ids if i in submitted]
        row = {
            "submitted_at": max(times) if times else None,
            "path": ", ".join(str(paths.get(i, i)) for i in exp_ids),
            "run": job["run"]
            or job.get("label")
            or ("rigid (all)" if job.get("whole") else "test"),
            "state": TEST_STATES.get(job["state"], job["state"]),
            "local": "",
            **dict.fromkeys(TEST_NUMBERS),
            "finished_at": "",
            "folder": str(folder),
            "bytes": 0,
            "group_id": group_id,
            "experiment_folder": str(db.groups[group_id].mcor_test_folder()),
            "pick_call_id": job.get("pick_call_id"),
        }

        if job["state"] == "finished" and (folder / "metrics.json").is_file():
            try:
                numbers = run_numbers(folder)
            except (OSError, ValueError):
                numbers = {}

            row |= {key: numbers.get(name) for key, name in TEST_NUMBERS.items()}
            written = (folder / "metrics.json").stat().st_mtime
            row["finished_at"] = datetime.fromtimestamp(written).strftime(
                "%Y-%m-%d %H:%M"
            )
            row["local"] = cache.state(db.main_folder, folder, base)
            row["bytes"] = cache.remote_size(folder)

        rows.append(row)

    # By experiment, newest first, keeping the jobs' order within one (stable)
    rows.sort(key=lambda row: str(row["submitted_at"] or ""), reverse=True)

    return pd.DataFrame(rows, columns=[*TEST_COLUMNS, *TEST_HIDDEN])


def my_experiments(
    tests: pd.DataFrame,
    main_folder: Path,
    base: None | Path = None,
    picks: None | dict[int, dict] = None,
) -> pd.DataFrame:
    """
    One row per experiment of `tests` (from `my_tests`), in its order: its
    status (after a test was picked, from `picks`: see
    `cluster.latest_picks`), how many of its tests finished, whether its
    files are downloaded (`local copy`: "no", "yes", "partly" or "newer on
    server") and how much downloading them copies (`size`). Hidden:
    `folder`, where its tests are; `folders`, those with files to download
    (its own, `raw/`, `final/` and each finished run's); `bytes`, their size;
    `finished`, its finished runs (as `Run`s, with folders on the share), the
    cluster test first.
    """
    rows = []

    for group_id, runs in tests.groupby("group_id", sort=False):
        top = Path(runs["experiment_folder"].iloc[0])
        done = runs[runs["state"] == "finished"]
        states = set(runs["state"])

        if len(done):
            status = READY
        elif states <= {"stopped", "failed"}:
            status = STOPPED
        else:
            status = WAITING

        pick = (picks or {}).get(int(group_id))

        if pick is not None:
            status = _picked_status(pick, runs[runs["folder"] == str(top)])

        finished = [
            Run(
                name=str(run["run"]),
                folder=Path(run["folder"]),
                **{key: run[key] for key in ("corr_with_ref", "flow_px", "jitter_px")},
                at_limit=run["at_limit"],
            )
            for _, run in done.iterrows()
        ]
        finished.sort(key=lambda run: run.folder != top)  # stable
        candidates = [top, top / RAW_PREVIEW_FOLDER, top / FINAL_FOLDER]
        candidates += [run.folder for run in finished if run.folder != top]
        folders = [folder for folder in candidates if cache.small_files(folder)]
        size = sum(cache.remote_size(folder) for folder in folders)
        here = {cache.state(main_folder, folder, base) for folder in folders}

        if not here:
            local = ""
        elif cache.NEWER in here:
            local = cache.NEWER
        elif len(here) > 1:
            local = PARTLY
        else:
            local = here.pop()

        rows.append(
            {
                "submitted_at": runs["submitted_at"].iloc[0],
                "path": runs["path"].iloc[0],
                "status": status,
                "runs": f"{len(done)} of {len(runs)} finished",
                "local copy": local,
                "size": cache.human_size(size) if folders else "",
                "group_id": int(group_id),
                "folder": str(top),
                "folders": [str(folder) for folder in folders],
                "bytes": size,
                "finished": finished,
            }
        )

    return pd.DataFrame(rows, columns=[*EXPERIMENT_COLUMNS, *EXPERIMENT_HIDDEN])


def _picked_status(pick: dict, whole: pd.DataFrame) -> str:
    """
    The experiment status once a test was picked: correcting with it (the
    whole correction job in `whole`, or not submitted yet), stopped, or
    ready for the final check.
    """
    if not pick["correct"]:
        return FINAL_CHECK  # the last whole correction used these settings

    ran = whole[whole["pick_call_id"] == pick["call_id"]]

    if ran.empty:
        return CORRECTING.format(run=pick["run"])  # at the next cluster run

    state = ran["state"].iloc[0]

    if state == "finished":
        return FINAL_CHECK

    if state in ("stopped", "failed"):
        return CORRECTION_STOPPED.format(run=pick["run"])

    return CORRECTING.format(run=pick["run"])


class Downloads(QThread):
    """
    Downloads experiments' files one after the other, off the database thread
    (a share can be slow). `progress` says which, `done` how many were
    downloaded and, if it stopped early, why.
    """

    progress = Signal(str)
    done = Signal(int, str)

    def __init__(
        self,
        main_folder: Path,
        experiments: list[tuple[str, list[Path]]],
        base: Path,
    ):
        super().__init__()
        self.main_folder = main_folder
        self.experiments = experiments  # (what to call it, its folders)
        self.base = base

    def run(self) -> None:
        count = len(self.experiments)

        def check(_name: str) -> None:
            if self.isInterruptionRequested():
                raise InterruptedError("the app was closed")

        for index, (label, folders) in enumerate(self.experiments):
            self.progress.emit(
                f"Downloading experiment {index + 1} of {count} ({label})"
            )

            try:
                for folder in folders:
                    cache.download(self.main_folder, folder, self.base, progress=check)
            except (OSError, InterruptedError) as error:
                self.done.emit(index, f"{label}: {error}")
                return

        self.done.emit(count, "")


def my_missing(db, mine: pd.DataFrame) -> pd.DataFrame:
    """
    The required annotations still empty on `mine`'s recordings, in the order
    of `mine` (newest first), each with `waiting_since`: when its recording
    was submitted.
    """
    missing = db.missing_annotations()

    if missing.empty or mine.empty:
        return missing.iloc[0:0].assign(waiting_since=None)[
            ["waiting_since", *missing.columns]
        ]

    rank = {target: position for position, target in enumerate(mine.index)}
    order = [
        rank.get((kind, int(row_id)))
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]
    missing = missing.assign(_rank=order).dropna(subset="_rank")
    missing = missing.sort_values("_rank", kind="stable").drop(columns="_rank")
    since = [
        mine["submitted_at"].get((kind, int(row_id)))
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]

    return missing.reset_index(drop=True).assign(waiting_since=since)[
        ["waiting_since", *missing.columns]
    ]


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
        self.mine_ids: list[int] = []  # each row's experiment or z-stack id
        self.experiments = pd.DataFrame(
            columns=[*EXPERIMENT_COLUMNS, *EXPERIMENT_HIDDEN]
        )
        self.main_folder = context.main_folder  # as the database has it
        self.downloads: None | Downloads = None
        self.windows: dict[str, McorWindow] = {}  # open, by experiment folder
        self.tab_names: list[str] = []
        self.tab_picked = False  # by the user, since the user last changed

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

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBarClicked.connect(self._tab_clicked)
        layout.addWidget(self.tabs)

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
            "Waiting",
            "Submitted, waiting for the cluster to add them",
            [self.add_now, self.submit_again, self.edit_draft],
        )

        self.rerun_sync = QPushButton("Rerun sync")
        self.rerun_sync.setStatusTip(
            "Process the experiment's sync file again at the cluster's next run"
        )
        self.rerun_sync.clicked.connect(self._rerun_sync)
        self.recordings_view = self._table(
            "Added", "Added to the database by the cluster", [self.rerun_sync]
        )
        self.missing_view = self._table(
            "Annotations", "Required annotations that were not filled"
        )

        self.download = QPushButton("Download")
        self.download.setStatusTip(
            "Copy the selected experiments' test files to this computer"
        )
        self.download.clicked.connect(lambda: self._download(self._picked()))
        self.download_all = QPushButton("Download all")
        self.download_all.setStatusTip(
            "Copy the test files of every experiment of yours not on this computer"
        )
        self.download_all.clicked.connect(lambda: self._download(self._not_here()))
        self.delete_local = QPushButton("Delete local copy")
        self.delete_local.setStatusTip(
            "Delete the selected experiments' files from this computer "
            "(not the server's)"
        )
        self.delete_local.clicked.connect(self._delete_local)
        self.show_folder = QPushButton("Show folder")
        self.show_folder.clicked.connect(self._show_folder)
        self.compare = QPushButton("Compare tests")
        self.compare.clicked.connect(self._compare)
        self.tests_view = self._table(
            "Motion correction",
            "Your experiments' motion corrections: the cluster's tests and the "
            "runs of sweeps",
            [
                self.download,
                self.download_all,
                self.delete_local,
                self.show_folder,
                self.compare,
            ],
        )
        self.tests_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.tests_view.doubleClicked.connect(self._compare)

        for view in (self.queue_view, self.recordings_view, self.tests_view):
            view.selectionModel().selectionChanged.connect(self._enable)

        context.follow_changes(self, self.reload)
        context.user_changed.connect(self._user_changed)
        self._enable()
        self.reload()

    def _table(
        self, name: str, tip: str, buttons: list[QPushButton] = ()
    ) -> QTableView:
        """A tab named `name` with a table and `buttons` under it."""
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        view = QTableView()
        view.setModel(FrameModel())
        view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        view.setAlternatingRowColors(True)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        fit_columns(view)
        sortable(view)
        box_layout.addWidget(view)

        if buttons:
            row = QHBoxLayout()
            row.addStretch(1)

            for button in buttons:
                row.addWidget(button)

            box_layout.addLayout(row)

        index = self.tabs.addTab(box, name)
        self.tabs.setTabToolTip(index, tip)
        self.tab_names.append(name)

        return view

    def reload(self) -> None:
        user = self.context.user
        if not user:
            self.who.setText("Pick a user to start")

        base = self.context.cache_base

        def load(db):
            mine = my_recordings(db, user)
            users = db.users
            named = users.loc[users["user_name"] == user, "display_name"]
            name = str(named.iloc[0]) if len(named) else user
            queue = my_queue(db.main_folder, db.project, user)

            return (
                name,
                queue,
                mine,
                my_missing(db, mine),
                my_experiments(
                    my_tests(db, user, mine, base),
                    db.main_folder,
                    base,
                    latest_picks(db),
                ),
                cluster_line(db),
                db.main_folder,
            )

        def loaded(result):
            name, queue, mine, missing, experiments, (cluster, kind), main = result
            self.main_folder = main

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
            self.experiments = experiments.reset_index(drop=True)
            self._model(self.tests_view).set_frame(self.experiments[EXPERIMENT_COLUMNS])

            for _, row in self.experiments.iterrows():
                if str(row["folder"]) in self.windows:
                    self.windows[str(row["folder"])].set_status(row["status"])
            self._count_tabs()
            self._enable()

        self.context.run(load, loaded, "load QC")

    def _count_tabs(self) -> None:
        """Each tab's name with its row count; the first with rows shown."""
        views = (
            self.queue_view,
            self.recordings_view,
            self.missing_view,
            self.tests_view,
        )
        counts = [self._model(view).rowCount() for view in views]

        for index, (name, count) in enumerate(zip(self.tab_names, counts)):
            self.tabs.setTabText(index, f"{name} ({count})" if count else name)

        if not self.tab_picked:
            self.tabs.setCurrentIndex(
                next((i for i, count in enumerate(counts) if count), 0)
            )

    def _tab_clicked(self, index: int) -> None:
        if index >= 0:
            self.tab_picked = True

    def _user_changed(self, *_) -> None:
        self.tab_picked = False
        self.reload()

    # ----------------------------------------------------------------------- #
    # Actions
    # ----------------------------------------------------------------------- #

    def _selected(self, view: QTableView, frame: pd.DataFrame) -> None | pd.Series:
        rows = view.selectionModel().selectedRows()

        if not rows or rows[0].row() >= len(frame):
            return None

        # The view may be sorted: its row is not the frame's
        return frame.iloc[self._model(view).source_row(rows[0].row())]

    def _enable(self, *_) -> None:
        queued_row = self._selected(self.queue_view, self.queue)
        state = None if queued_row is None else queued_row["state"]
        self.add_now.setEnabled(state is not None)
        self.submit_again.setEnabled(state == "failed")
        self.edit_draft.setEnabled(state is not None)

        added = self._selected(self.recordings_view, self.mine)
        self.rerun_sync.setEnabled(added is not None and added["kind"] == "experiment")

        picked = self._picked()
        busy = self.downloads is not None
        here = [row["local copy"] not in (cache.NOT_HERE, "") for _, row in picked]
        self.download.setEnabled(
            not busy and any(row["local copy"] in TO_DOWNLOAD for _, row in picked)
        )
        self.delete_local.setEnabled(not busy and any(here))

        waiting = self._not_here()
        size = sum(int(row["bytes"]) for _, row in waiting)
        plural = "experiment" if len(waiting) == 1 else "experiments"
        self.download_all.setEnabled(not busy and bool(waiting))
        self.download_all.setText(
            f"Download all ({len(waiting)} {plural}, {cache.human_size(size)})"
            if waiting
            else "Download all"
        )

        # Opened from this computer only, so a slow share is never played from
        one = len(picked) == 1 and here[0]
        tip = "" if one else " (download it first)"
        self.show_folder.setEnabled(one)
        self.show_folder.setStatusTip("Open the experiment's test folder" + tip)
        ready = one and bool(len(picked[0][1]["finished"]))
        self.compare.setEnabled(ready)
        self.compare.setStatusTip(
            "Compare the raw video and each test, side by side"
            + (tip or ("" if ready else " (once a test finished)"))
        )

    def _local(self, row: pd.Series) -> Path:
        return cache.local_folder(
            self.main_folder, row["folder"], self.context.cache_base
        )

    def _show_folder(self) -> None:
        picked = self._picked()

        if len(picked) != 1:
            return

        folder = self._local(picked[0][1])

        if not folder.is_dir():
            self.context.status.emit(f"{picked[0][0]} is not downloaded yet")
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self.context.status.emit(f"Could not open {folder}")

    def _compare(self, *_) -> None:
        """Open the selected experiment's window, from its local copy."""
        picked = self._picked()

        if len(picked) != 1 or not self.compare.isEnabled():
            return

        label, row = picked[0]
        key = str(row["folder"])

        if key in self.windows:
            self.windows[key].raise_()
            self.windows[key].activateWindow()
            return

        base = self.context.cache_base
        runs = [
            replace(run, folder=cache.local_folder(self.main_folder, run.folder, base))
            for run in row["finished"]
        ]
        window = McorWindow(
            label,
            self._local(row),
            runs,
            self.context.status.emit,
            pick=lambda run: self._pick(key, run),
        )
        window.set_status(row["status"])
        window.closed.connect(lambda: self.windows.pop(key, None))
        self.windows[key] = window
        window.show()

    def _pick(self, key: str, run: str) -> None:
        """
        Once confirmed, pick test `run` to correct every acquisition of the
        experiment with tests in `key` (`cluster.pick_candidate`).
        """
        window = self.windows.get(key)
        rows = self.experiments[self.experiments["folder"] == key]

        if window is None or rows.empty:
            return

        row = rows.iloc[0]
        found = [r for r in row["finished"] if r.name == run]

        if not found or not self.context.confirm(
            "Pick candidate",
            f"Motion correct {row['path']} with {run}'s settings?\n\n"
            "Overwrite the last run's files?",
            "Motion correct",
            parent=window,
        ):
            return

        group_id, folder = int(row["group_id"]), found[0].folder

        def pick(db):
            try:
                return pick_candidate(db, group_id, run, folder)
            except ValueError as error:  # refused: said over the window
                return str(error)

        def picked(result):
            if isinstance(result, str):
                box = QMessageBox(
                    QMessageBox.Icon.Warning, "Pick candidate", result, parent=window
                )
                box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                box.open()
            elif result:
                self.context.status.emit(
                    f"{row['path']} will be corrected with {run} at the next "
                    "cluster run"
                )
            else:
                self.context.status.emit(
                    f"The last correction of {row['path']} already used {run}'s "
                    "settings"
                )

        self.context.run(pick, picked, "pick candidate", writes=True)

    def _picked(self) -> list[tuple[str, pd.Series]]:
        """The selected experiments, as `(what to call it, row)`, in view order."""
        model = self._model(self.tests_view)
        rows = sorted(i.row() for i in self.tests_view.selectionModel().selectedRows())

        return [
            (str(row["path"]), row)
            for row in (
                self.experiments.iloc[model.source_row(r)]
                for r in rows
                if r < len(self.experiments)
            )
        ]

    def _not_here(self) -> list[tuple[str, pd.Series]]:
        """The experiments with files not downloaded (or changed since)."""
        return [
            (str(row["path"]), row)
            for _, row in self.experiments.iterrows()
            if row["local copy"] in TO_DOWNLOAD
        ]

    def _download(self, picked: list[tuple[str, pd.Series]]) -> None:
        experiments = [
            (label, [Path(folder) for folder in row["folders"]])
            for label, row in picked
            if row["local copy"] in TO_DOWNLOAD
        ]

        if not experiments or self.downloads is not None:
            return

        self.downloads = Downloads(
            self.main_folder, experiments, self.context.cache_base
        )
        self.downloads.progress.connect(self.context.status.emit)
        self.downloads.done.connect(self._downloaded)
        self.downloads.start()
        self._enable()

    def _downloaded(self, count: int, problem: str) -> None:
        if self.downloads is not None:
            self.downloads.wait()
            self.downloads.deleteLater()
            self.downloads = None

        plural = "" if count == 1 else "s"
        self.context.status.emit(
            f"Downloaded {count} experiment{plural}"
            + (f"; stopped at {problem}" if problem else "")
        )
        self.context.cache_changed.emit()
        self.reload()

    def _delete_local(self) -> None:
        picked = self._picked()

        for _, row in picked:
            window = self.windows.get(str(row["folder"]))

            if window is not None:
                window.close()  # its videos are about to go

        freed = sum(
            cache.delete(self.main_folder, folder, self.context.cache_base)
            for _, row in picked
            for folder in row["folders"]
        )
        self.context.status.emit(
            f"Deleted the local copy ({cache.human_size(freed)} freed)"
        )
        self.context.cache_changed.emit()
        self.reload()

    def close_down(self) -> None:
        """
        Close the experiments' windows and stop a download (after the file it
        is on), before the app closes.
        """
        for window in list(self.windows.values()):
            window.close()

        if self.downloads is not None:
            self.downloads.requestInterruption()
            self.downloads.wait()

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
