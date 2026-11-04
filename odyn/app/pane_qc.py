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

import os

from dataclasses import replace
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import QElapsedTimer, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from ..claims import STEPS, ClaimTaken, claim, pending_claims
from ..cluster import ENDED, pick_repeats, send_request
from ..groups import FULL_FOLDER
from ..utils import get_user
from ..qc import (
    COLUMNS,
    EXPERIMENT_FOLDERS,
    MISSING_COLUMNS,
    TEST_COLUMNS,
    TEST_HIDDEN,
    TEST_NUMBERS,
    Listing,
    cluster_line,
    qc_path,
    qc_summary,
    read_qc_summary,
    remote_size,
)
from . import cache
from .context import Context
from ..drafts.draft import Draft
from .mcor_window import McorWindow, Run
from .models import FrameModel, fit_columns, sortable
from .style import text_style
from ..drafts.submissions import add_now, queued, submit_again, take_back

QUEUE_COLUMNS = ["submitted_at", "session", "state", "progress"]
EXPERIMENT_COLUMNS = [
    "submitted_at",
    "path",
    "status",
    "raw check",
    "runs",
    "local copy",
    "size",
]
EXPERIMENT_HIDDEN = [
    "group_id",
    "folder",
    "folders",
    "bytes",
    "finished",
    "excluded",
    "stopped",
    "correcting",
    "user",
]

# An experiment's status, from its tests (later steps add their own)
WAITING = "waiting for the cluster"
STOPPED = "stopped"

# A job's states once it ended without finishing (`squeue`'s and ours)
STOPPED_STATES = ("stopped", "failed", "timeout", "cancelled", "out_of_memory")
READY = "ready to compare"
CORRECTING = "correcting with {run}"
CORRECTION_STOPPED = "correction with {run} stopped"
FINAL_CHECK = "ready for final check"
APPROVED = "approved"

# An experiment some of whose files are downloaded (see `cache.state`)
PARTLY = "partly"
TO_DOWNLOAD = (cache.NOT_HERE, PARTLY, cache.NEWER)


def my_queue(
    main_folder: Path, project: None | str, user: str, everyone: bool = False
) -> pd.DataFrame:
    """
    The drafts `user` submitted (`everyone`: anyone, each with its `user`)
    that the cluster has not added yet, oldest first: the session, `waiting`
    or `failed`, how far the copy got (or why it failed), and when it was
    submitted. `path` is the queued file.
    """
    rows = []

    for path, state in queued(main_folder, project):
        try:
            draft = Draft.load(path)
        except (OSError, ValueError):
            continue

        if not everyone and draft.data.get("submitted_by") != user:
            continue

        rows.append(
            {
                "user": draft.data.get("submitted_by", ""),
                "session": draft.where,
                "state": state.state,
                "progress": state.message or "not looked at yet",
                "submitted_at": draft.data.get("submitted_at", ""),
                "path": str(path),
            }
        )

    return pd.DataFrame(rows, columns=[*QUEUE_COLUMNS, "path", "user"])


def my_experiments(
    tests: pd.DataFrame,
    main_folder: Path,
    base: None | Path = None,
    picks: None | dict[int, dict] = None,
    checks: None | dict[int, list[int]] = None,
    approvals: None | dict[int, dict] = None,
    listing: None | Listing = None,
) -> pd.DataFrame:
    """
    One row per experiment of `tests` (from `my_tests`, with the same
    `listing` to look at each folder once), in its order: its
    status (after a test was picked, from `picks`: see
    `cluster.latest_picks`), its raw check (from `checks`, see
    `latest_raw_checks`; the acquisitions excluded hidden in `excluded`;
    "approved" once approved after the latest pick, from `approvals`, see
    `latest_approvals`),
    how many of its tests finished, whether its
    files are downloaded (`local copy`: "no", "yes", "partly" or "newer on
    server") and how much downloading them copies (`size`). Hidden:
    `folder`, where its tests are; `folders`, those with files to download
    (its own, `raw/`, `full/` and each finished run's); `bytes`, their size;
    `finished`, its finished test runs (as `Run`s, with folders on the
    share), the cluster test first; not the whole corrections.
    """
    listing = listing or Listing()
    rows = []

    for group_id, runs in tests.groupby("group_id", sort=False):
        top = Path(runs["experiment_folder"].iloc[0])
        done = runs[runs["state"] == "finished"]
        stopped = runs[runs["state"].isin(STOPPED_STATES)]
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

        approved = (approvals or {}).get(int(group_id), {}).get("call_id")

        if approved is not None and (pick is None or approved > pick["call_id"]):
            status = APPROVED

        finished = [
            Run(
                name=str(run["run"]),
                folder=Path(run["folder"]),
                **{key: _value(run[key]) for key in TEST_NUMBERS},
            )
            for _, run in done.iterrows()
            # A whole correction repeats a test (rigid (all) is rigid): the
            # full run and the final check show it
            if not run["whole"]
        ]
        finished.sort(key=lambda run: run.folder != top)  # stable
        candidates = [top, *(top / name for name in EXPERIMENT_FOLDERS)]
        candidates += [run.folder for run in finished if run.folder != top]
        folders = [folder for folder in candidates if listing.files(folder)]
        size = sum(remote_size(folder, listing) for folder in folders)
        here = {cache.state(main_folder, folder, base, listing) for folder in folders}

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
                "user": runs["user"].iloc[0],
                "path": runs["path"].iloc[0],
                "status": status,
                "raw check": _raw_check_line((checks or {}).get(int(group_id))),
                "runs": f"{len(done)} of {len(runs)} finished"
                + (f", {len(stopped)} stopped" if len(stopped) else ""),
                "local copy": local,
                "size": cache.human_size(size) if folders else "",
                "group_id": int(group_id),
                "folder": str(top),
                "folders": [str(folder) for folder in folders],
                "bytes": size,
                "finished": finished,
                "excluded": list((checks or {}).get(int(group_id)) or []),
                "stopped": [
                    (str(run["job"]), str(run["run"])) for _, run in stopped.iterrows()
                ],
                # A whole correction not over: picking again waits for it
                "correcting": bool(
                    (runs["whole"].astype(bool) & ~runs["state"].isin(ENDED)).any()
                ),
            }
        )

    return pd.DataFrame(rows, columns=[*EXPERIMENT_COLUMNS, *EXPERIMENT_HIDDEN])


def _raw_check_line(excluded: None | list[int]) -> str:
    """`not yet`, `done`, or `done, 2 excluded`."""
    if excluded is None:
        return "not yet"

    return f"done, {len(excluded)} excluded" if excluded else "done"


def _value(value):
    """A number of a table row, `None` when missing (NaN)."""
    return None if value is None or pd.isna(value) else value


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


# How often a download says how far it is (ms), and what it says when stopped
REPORT_MS = 250
CANCELLED = "cancelled"


class Downloads(QThread):
    """
    Downloads experiments' files one after the other, off the database thread
    (a share can be slow). `progress(done, total, text)` tells the bytes
    copied of all to copy and, in words, which experiment, how fast and how
    long is left; `done` how many experiments were downloaded and, if it
    stopped early, why ("cancelled" when asked to stop).
    """

    progress = Signal(object, object, str)  # bytes: can pass 2 GB
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
        clock = QElapsedTimer()
        clock.start()
        last = -REPORT_MS

        try:
            total = sum(
                path.stat().st_size
                for _, folders in self.experiments
                for folder in folders
                for path in cache.pending(self.main_folder, folder, self.base)
            )
        except OSError as error:
            self.done.emit(0, str(error))
            return

        done = 0

        for index, (label, folders) in enumerate(self.experiments):
            where = f"{label} ({index + 1} of {count})" if count > 1 else label

            def copied(_name: str, size: int) -> None:
                nonlocal done, last

                if self.isInterruptionRequested():
                    raise InterruptedError(CANCELLED)

                done += size
                now = clock.elapsed()

                if now - last >= REPORT_MS:
                    last = now
                    self.progress.emit(
                        done, total, _download_line(where, done, total, now)
                    )

            try:
                copied("", 0)

                for folder in folders:
                    cache.download(self.main_folder, folder, self.base, progress=copied)
            except (OSError, InterruptedError) as error:
                self.done.emit(
                    index, CANCELLED if str(error) == CANCELLED else f"{label}: {error}"
                )
                return

        self.done.emit(count, "")


def _download_line(where: str, done: int, total: int, elapsed_ms: float) -> str:
    """`Downloading e1: 120 MB of 560 MB, 12 MB/s, about 40 s left`."""
    line = f"Downloading {where}: {cache.human_size(done)} of {cache.human_size(total)}"
    seconds = elapsed_ms / 1000

    if done and seconds >= 1:
        rate = done / seconds
        left = (total - done) / rate
        line += f", {cache.human_size(rate)}/s, about {_duration(left)} left"

    return line


def _duration(seconds: float) -> str:
    """`40 s`, `3 min`, `1 h 20 min`."""
    if seconds < 60:
        return f"{max(1, round(seconds))} s"

    if seconds < 3600:
        return f"{round(seconds / 60)} min"

    hours, minutes = divmod(round(seconds / 60), 60)

    return f"{hours} h {minutes} min"


def saved_folder(main_folder: Path, paths: str) -> Path:
    """
    Where an experiment's corrected files and reference are saved: the first
    experiment's `processed/mcor/` (`paths`, as the QC page lists them).
    """
    first = str(paths).split(", ")[0]

    return Path(main_folder) / first / "processed" / "mcor"


# How often the page looks for a newer view while on screen
POLL_MS = 30_000


def qc_view(
    summary: dict,
    main_folder: Path,
    user: None | str,
    everyone: bool = False,
    base: None | Path = None,
    claimed: None | dict[tuple[int, str], dict] = None,
):
    """
    One user's rows of the QC view `summary` (`odyn.qc.qc_summary`), or every
    user's: the display name, the recordings (indexed by `(kind, id)`), the
    missing annotations, the experiments (`my_experiments`, their downloads
    checked against the file lists in the view and the copies in `base`,
    with the decisions `claimed` and not recorded yet, see `with_claims`),
    and the cluster line. Reads only this computer's disk.
    """
    main = Path(main_folder)
    listing = Listing(
        seen={
            str(main / folder): {name: tuple(stat) for name, stat in files.items()}
            for folder, files in summary["files"].items()
        },
        scan=False,
    )

    mine = pd.DataFrame(summary["recordings"], columns=["id", *COLUMNS, "user"])
    mine = mine.set_index(["kind", "id"], drop=False).drop(columns="id")
    missing = pd.DataFrame(summary["missing"], columns=MISSING_COLUMNS)
    tests = pd.DataFrame(summary["tests"], columns=[*TEST_COLUMNS, *TEST_HIDDEN])

    for column in ("folder", "experiment_folder"):
        tests[column] = [str(main / folder) for folder in tests[column]]

    for column in TEST_NUMBERS:
        tests[column] = pd.to_numeric(tests[column], errors="coerce")

    if not everyone:
        mine = mine[mine["user"] == user]
        missing = missing[missing["user"] == user]
        tests = tests[tests["user"] == user]

    # Whether each finished test is downloaded here
    tests = tests.reset_index(drop=True)

    for index, row in tests.iterrows():
        if "metrics.json" in listing.files(row["folder"]):
            tests.at[index, "local"] = cache.state(main, row["folder"], base, listing)

    def by_group(name):
        return {int(group): value for group, value in summary[name].items()}

    experiments = my_experiments(
        tests,
        main,
        base,
        by_group("picks"),
        by_group("raw_checks"),
        by_group("approvals"),
        listing,
    )
    name = summary["users"].get(user) or user
    experiments = with_claims(experiments, claimed or {})

    return name, mine, missing, experiments, cluster_line(summary)


# What a row says of a decision the cluster has not recorded yet, the
# strongest first (an approval says more than the pick before it)
CLAIMED = {
    "approve": "approved by {user}, waiting for the cluster",
    "pick": "{run} picked by {user}, waiting for the cluster",
    "again": "run again by {user}, waiting for the cluster",
}
WAITING_NOTE = " (waiting for the cluster)"


def with_claims(
    experiments: pd.DataFrame, claimed: dict[tuple[int, str], dict]
) -> pd.DataFrame:
    """
    `experiments` with the decisions `claimed` (`odyn.claims.pending_claims`)
    in their status and raw check: decided, waiting for the cluster, or
    refused by it (and why).
    """
    status, raw_check = list(experiments["status"]), list(experiments["raw check"])
    excluded = list(experiments["excluded"])

    for index, group_id in enumerate(experiments["group_id"]):
        mine = {
            step: found for (group, step), found in claimed.items() if group == group_id
        }
        refused = [(step, found) for step, found in mine.items() if found.get("failed")]

        if refused:
            step, found = refused[-1]
            status[index] = (
                f"{STEPS[step]} by {found['user']} refused: {found['failed']}"
            )
            continue

        if "raw" in mine:
            excluded[index] = list(mine["raw"].get("excluded") or [])
            raw_check[index] = _raw_check_line(excluded[index]) + WAITING_NOTE

        for step, text in CLAIMED.items():
            if step in mine:
                status[index] = text.format(**mine[step])
                break

    return experiments.assign(
        status=status, **{"raw check": raw_check}, excluded=excluded
    )


class _Job(QThread):
    """Runs a function on a thread of its own and hands back its result."""

    result = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self) -> None:  # noqa: D102 (Qt's)
        try:
            outcome = self.function()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.result.emit(outcome)


class QcPane(QWidget):
    """
    What is waiting on the current user, or on every user with "Show all
    users" (to take over someone's QC; what is done is recorded as done by
    the current user). `restored` gives the path of a draft taken back from
    the cluster, now among this computer's drafts.
    """

    restored = Signal(object)

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.queue = pd.DataFrame(columns=[*QUEUE_COLUMNS, "path", "user"])
        self.mine = pd.DataFrame(columns=[*COLUMNS, "user"])
        self.mine_ids: list[int] = []  # each row's experiment or z-stack id
        self.experiments = pd.DataFrame(
            columns=[*EXPERIMENT_COLUMNS, *EXPERIMENT_HIDDEN]
        )
        self.main_folder = Path(context.main_folder).resolve()  # as the database
        self.downloads: None | Downloads = None
        self.windows: dict[str, McorWindow] = {}  # open, by experiment folder
        self.tab_names: list[str] = []
        self.tab_picked = False  # by the user, since the user last changed
        self.loading = 0  # the latest reload: an older one's rows are dropped
        self.summary: None | dict = None  # the QC view (`odyn.qc`) last read
        self.stamp: None | float = None  # its file time, then
        self.jobs: set[_Job] = set()  # reading it, off the database thread

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.who = QLabel("")
        reload = QPushButton("Reload")
        reload.clicked.connect(lambda: self.reload())
        # Every tab, every user's: to take over someone's QC
        self.everyone = QCheckBox("Show all users")
        self.everyone.setStatusTip(
            "Show every user's recordings, annotations and experiments in each tab"
        )
        self.everyone.toggled.connect(lambda _: self.reload())
        top.addWidget(self.who, stretch=1)
        top.addWidget(self.everyone)
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
            "Copy the test files of every experiment listed not on this computer"
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
        self.run_again = QPushButton("Run again")
        self.run_again.clicked.connect(self._run_again)
        self.tests_view = self._table(
            "Motion correction",
            "Your experiments' motion corrections: the cluster's tests and the "
            "runs of sweeps",
            [
                self.download,
                self.download_all,
                self.delete_local,
                self.show_folder,
                self.run_again,
                self.compare,
            ],
        )
        self.tests_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)

        # How far a download is, under the table, while there is one
        self.download_bar = QProgressBar()
        self.download_bar.setRange(0, 1000)
        self.download_bar.setTextVisible(False)
        self.download_text = QLabel("")
        self.cancel_download = QPushButton("Cancel")
        self.cancel_download.setStatusTip("Stop the download (what was copied stays)")
        self.cancel_download.clicked.connect(self._cancel_download)
        progress = QHBoxLayout()
        progress.addWidget(self.download_text, stretch=1)
        progress.addWidget(self.download_bar, stretch=1)
        progress.addWidget(self.cancel_download)
        self.tests_view.parentWidget().layout().insertLayout(1, progress)
        self._show_download(False)
        self.tests_view.doubleClicked.connect(self._compare)

        for view in (self.queue_view, self.recordings_view, self.tests_view):
            view.selectionModel().selectionChanged.connect(self._enable)

        context.follow_changes(self, self.reload)
        context.user_changed.connect(self._user_changed)
        self._enable()

        # Looks for a newer view while on screen (one round trip each time)
        self.poll = QTimer(self)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(lambda: self.isVisible() and self.reload())
        self.poll.start()
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
        """
        Show the view again: read the view file when it changed since the last
        look (one round trip when not), and the queue; then pick this user's
        rows and check the downloads here. Off the database thread, which the
        page needs only to build the view itself (see `_build_here`).
        """
        user = self.context.user
        self.loading += 1
        loading = self.loading

        if not user:
            self.who.setText("Pick a user to start")

        base = self.context.cache_base
        everyone = self.everyone.isChecked()
        main, project = self.main_folder, self.context.worker.project
        stamp, summary = self.stamp, self.summary

        def load():
            try:
                changed = os.stat(qc_path(main, project)).st_mtime != stamp
            except OSError:  # none yet
                changed = True

            read = read_qc_summary(main, project) if changed else summary
            queue = my_queue(main, project, user, everyone)
            claimed = pending_claims(main, project)

            if read is None:  # no view yet, or of another version
                return None, queue, None

            return read, queue, qc_view(read, main, user, everyone, base, claimed)

        def loaded(result):
            if loading != self.loading:  # a later reload is on its way
                return

            read, queue, view = result

            if read is None:
                self._build_here(queue, user, everyone, base)
                return

            if read is not self.summary:
                self.summary = read
                self.stamp = self._stamp()

            self._fill(view, queue, everyone, user)

        self._background(load, loaded)

    def _build_here(self, queue, user, everyone, base) -> None:
        """
        Build the view on this computer, for want of one from the cluster (none
        yet, or of another version): slow over a network, as it lists every
        test folder.
        """
        loading = self.loading
        main = self.main_folder
        self.cluster.setText("Looking at the tests on the share (this is slow)...")

        def build(db):
            summary = qc_summary(db)
            claimed = pending_claims(main, db.project)

            return summary, qc_view(summary, main, user, everyone, base, claimed)

        def built(result):
            if loading != self.loading:
                return

            self.summary, view = result
            self.stamp = None  # the cluster view is read again when it changes
            self._fill(view, queue, everyone, user)

        self.context.run(build, built, "build the QC view")

    def _stamp(self) -> None | float:
        try:
            return os.stat(
                qc_path(self.main_folder, self.context.worker.project)
            ).st_mtime
        except OSError:
            return None

    def _background(self, function, done) -> None:
        """Run `function()` on a thread of its own, then `done(result)` here."""
        job = _Job(function, self)
        self.jobs.add(job)
        self.context._busy(+1)  # counted with the database jobs (the cursor)

        def finished(result):
            try:
                done(result)
            finally:
                self.jobs.discard(job)
                self.context._busy(-1)

        def failed(message):
            self.jobs.discard(job)
            self.context._busy(-1)
            self.context.report("Load QC failed", message)

        job.result.connect(finished)
        job.failed.connect(failed)
        job.start()

    def _fill(self, view, queue, everyone, user) -> None:
        """Put one view (from `qc_view`) and the queue in the tables."""
        name, mine, missing, experiments, (cluster, kind) = view

        if everyone:
            self.who.setText("To-do list of all users")
        elif user:
            self.who.setText(f"To-do list for {name}")

        self.cluster.setText(cluster)
        self.cluster.setStyleSheet(text_style(kind))

        self.queue = queue.reset_index(drop=True)
        self.mine = mine.reset_index(drop=True)
        self.mine_ids = [int(row_id) for _, row_id in mine.index]

        def shown(frame, columns):
            """`columns`, and whose each row is when showing all users."""
            columns = list(columns)

            if everyone:
                columns.insert(1, "user")

            return frame[columns]

        self._model(self.queue_view).set_frame(shown(self.queue, QUEUE_COLUMNS))
        self._model(self.recordings_view).set_frame(shown(self.mine, COLUMNS))
        self._model(self.missing_view).set_frame(
            shown(missing, [c for c in missing.columns if c != "user"])
        )
        self.experiments = experiments.reset_index(drop=True)
        columns = list(EXPERIMENT_COLUMNS)

        if everyone:  # whose each is
            columns.insert(columns.index("path") + 1, "user")

        self._model(self.tests_view).set_frame(self.experiments[columns])

        for _, row in self.experiments.iterrows():
            if str(row["folder"]) in self.windows:
                self.windows[str(row["folder"])].set_status(
                    _window_status(row), row["status"] == FINAL_CHECK
                )
        self._count_tabs()
        self._enable()

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
        """
        Another user: the tables empty at once and fill with theirs, so
        nothing of the user before can be acted on in between.
        """
        self.tab_picked = False
        self._clear()
        self.reload()

    def _clear(self) -> None:
        """Every table empty, until a reload fills them."""
        self.queue = self.queue.iloc[0:0]
        self.mine = self.mine.iloc[0:0]
        self.mine_ids = []
        self.experiments = self.experiments.iloc[0:0]

        for view in (
            self.queue_view,
            self.recordings_view,
            self.missing_view,
            self.tests_view,
        ):
            self._model(view).set_frame(pd.DataFrame())

        if self.context.user:
            self.who.setText(f"Loading the to-do list of {self.context.user}")

        self._count_tabs()
        self._enable()

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
        # A test run to compare, or the full run to check (no sweep yet)
        ready = one and (
            bool(len(picked[0][1]["finished"]))
            or str(Path(picked[0][1]["folder"]) / FULL_FOLDER)
            in picked[0][1]["folders"]
        )
        stopped = len(picked) == 1 and bool(picked[0][1]["stopped"])
        self.run_again.setEnabled(stopped)
        self.run_again.setStatusTip(
            "Run the stopped jobs again at the cluster's next run"
            if stopped
            else "Run stopped jobs again (none stopped)"
        )
        self.compare.setEnabled(ready)
        self.compare.setStatusTip(
            "Check the full run, and compare the raw video and each test"
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
            pick=lambda run, aside: self._pick(key, run, aside),
            check_raw=lambda excluded: self._check_raw(key, excluded),
            excluded=row["excluded"],
            approve=lambda excluded: self._approve(key, excluded),
            final_ready=row["status"] == FINAL_CHECK,
            raw_checked=row["raw check"] != "not yet",
            saved_folder=saved_folder(self.main_folder, row["path"]),
        )
        window.set_status(_window_status(row))
        window.closed.connect(lambda: self.windows.pop(key, None))
        self.windows[key] = window
        window.show()

    def _approve(self, key: str, excluded: list[int]) -> None:
        """
        Once confirmed, claim the approval of the motion corrected files of
        the experiment with tests in `key`, leaving out `excluded` (the
        cluster records it as `approve_mcor_files`), then delete its local
        copy: its checking is done.
        """
        window = self.windows.get(key)
        rows = self.experiments[self.experiments["folder"] == key]

        if window is None or rows.empty:
            return

        row = rows.iloc[0]
        listed = ", ".join(str(acq_id) for acq_id in excluded)
        leaving = (
            f"leaving out acquisitions {listed}" if listed else "every acquisition"
        )

        if row["raw check"] == "not yet":
            leaving += " (the raw check is not finished)"

        if not self.context.confirm(
            "Approve",
            f"Approve the motion correction of {row['path']}, {leaving}?\n\n"
            "Its local copy is deleted from this computer.",
            "Approve",
            parent=window,
        ):
            return

        folders = list(row["folders"])

        def approved():
            window.close()  # its videos are about to go
            freed = sum(
                cache.delete(self.main_folder, folder, self.context.cache_base)
                for folder in folders
            )
            self.context.status.emit(
                f"{row['path']} approved; local copy deleted "
                f"({cache.human_size(freed)} freed)"
            )
            self.context.cache_changed.emit()

        self._claim(row, "approve", approved, window, excluded=list(excluded))

    def _run_again(self) -> None:
        """
        Once confirmed, claim running the selected experiment's stopped jobs
        again (the cluster records it as `Group.run_again`).
        """
        picked = self._picked()

        if len(picked) != 1 or not picked[0][1]["stopped"]:
            return

        label, row = picked[0]
        stopped = row["stopped"]
        listed = ", ".join(name for _, name in stopped)

        if not self.context.confirm(
            "Run again",
            f"Run again the stopped jobs of {label}: {listed}?",
            "Run again",
        ):
            return

        def asked():
            self.context.status.emit(
                f"{label}: {listed} will run again at the cluster's next run"
            )

        self._claim(row, "again", asked, runs=[job for job, _ in stopped])

    def _check_raw(self, key: str, excluded: list[int]) -> None:
        """
        Once confirmed, claim the raw check of the experiment with tests in
        `key`, leaving out the acquisitions `excluded` (the cluster records it
        as `Group.check_raw`).
        """
        window = self.windows.get(key)
        rows = self.experiments[self.experiments["folder"] == key]

        if window is None or rows.empty:
            return

        row = rows.iloc[0]
        listed = ", ".join(str(acq_id) for acq_id in excluded)
        leaving = (
            f"excluding acquisitions {listed}"
            if listed
            else "keeping every acquisition"
        )

        if not self.context.confirm(
            "Finished raw check",
            f"Finish the raw check of {row['path']}, {leaving}?",
            "Finish",
            parent=window,
        ):
            return

        def checked():
            window.raw_check_done()  # on to the tests
            self.context.status.emit(
                f"Raw check of {row['path']} done"
                + (f" ({len(excluded)} excluded)" if excluded else "")
            )

        self._claim(row, "raw", checked, window, excluded=sorted(set(excluded)))

    def _pick(self, key: str, run: str, aside: list[str] = ()) -> None:
        """
        Once confirmed, claim the pick of test `run` to correct every
        acquisition of the experiment with tests in `key`, with the runs set
        `aside`, first to last (the cluster records it with
        `cluster.pick_candidate`).
        """
        window = self.windows.get(key)
        rows = self.experiments[self.experiments["folder"] == key]

        if window is None or rows.empty:
            return

        row = rows.iloc[0]
        found = [r for r in row["finished"] if r.name == run]

        if not found:
            return

        folder = found[0].folder

        def warn(text: str) -> None:
            box = QMessageBox(
                QMessageBox.Icon.Warning, "Pick candidate", text, parent=window
            )
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            box.open()

        if row["correcting"]:
            warn(
                f"The last motion correction of every acquisition of {row['path']} "
                "has not finished yet; pick again once it has."
            )
            return

        main, base = self.main_folder, self.context.cache_base

        def ask():
            # The copies here when there are, else the share: two small files
            here = [cache.local_folder(main, where, base) for where in (folder, key)]
            there = [Path(folder), Path(key)]

            return pick_repeats(*(here if here[0].is_dir() else there))

        def asked(result):
            repeats, rigid = result
            then = (
                "This will proceed to the final check."
                if repeats
                else f"This will overwrite the {'rigid' if rigid else 'last'} "
                "run files."
            )

            if not self.context.confirm(
                "Pick candidate",
                f"Are you sure you want to pick {run}?\n\n{then}",
                "Pick",
                parent=window,
            ):
                return

            def picked():
                if repeats:  # already corrected so: on to the final check
                    self.context.status.emit(
                        f"The last correction of {row['path']} already used "
                        f"{run}'s settings"
                    )
                    window.final_check()
                    return

                self.context.status.emit(
                    f"{row['path']} will be corrected with {run} at the next "
                    "cluster run"
                )
                box = QMessageBox(
                    QMessageBox.Icon.Information,
                    "Pick candidate",
                    f"Sent to the cluster: {row['path']}, corrected with "
                    f"{run}'s settings.\n\nCheck back later for the final check.",
                    parent=window,
                )
                box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                box.open()

            self._claim(
                row,
                "pick",
                picked,
                window,
                run=run,
                folder=Path(folder).relative_to(main).as_posix(),
                set_aside=list(aside),
            )

        self._background(ask, asked)

    def _claim(self, row, step: str, done, window=None, **details) -> None:
        """
        Claim `step` of the experiment in `row` (`odyn.claims`) for the user,
        with `details` for the cluster to record it, then `done()`. Someone
        else claiming it first is said over `window`.
        """
        main, project = self.main_folder, self.context.worker.project
        user = self.context.user or get_user()
        group_id = int(row["group_id"])

        def make():
            try:
                claim(main, project, group_id, step, user, **details)
            except ClaimTaken as taken:
                return str(taken)

            return None

        def made(taken):
            if taken is not None:
                box = QMessageBox(
                    QMessageBox.Icon.Warning,
                    STEPS[step].capitalize(),
                    f"The {STEPS[step]} of {row['path']}: {taken}, and the "
                    "cluster has not recorded it yet.",
                    parent=window or self,
                )
                box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                box.open()
                return

            done()
            self.reload()

        self._background(make, made)

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
        self.downloads.progress.connect(self._download_progress)
        self.downloads.done.connect(self._downloaded)
        self.download_bar.setValue(0)
        self.download_text.setText("Looking at what to download...")
        self._show_download(True)
        self.downloads.start()
        self._enable()

    def _show_download(self, shown: bool) -> None:
        for widget in (self.download_bar, self.download_text, self.cancel_download):
            widget.setVisible(shown)

    def _download_progress(self, done: int, total: int, text: str) -> None:
        self.download_bar.setValue(int(1000 * done / total) if total else 1000)
        self.download_text.setText(text)

    def _cancel_download(self) -> None:
        if self.downloads is not None:
            self.downloads.requestInterruption()
            self.download_text.setText("Stopping the download...")

    def _downloaded(self, count: int, problem: str) -> None:
        if self.downloads is not None:
            self.downloads.wait()
            self.downloads.deleteLater()
            self.downloads = None

        self._show_download(False)
        plural = "" if count == 1 else "s"

        if problem == CANCELLED:
            self.context.status.emit(
                f"Download cancelled ({count} experiment{plural} downloaded)"
            )
        else:
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

        self.poll.stop()

        for job in list(self.jobs):
            job.wait()

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


def _window_status(row: pd.Series) -> str:
    """The experiment status for its window: its status and raw check."""
    return f"{row['status']}; raw check: {row['raw check']}"
