"""
The day's work: one draft per session, filled in as the experiment runs and
submitted at the end.

Two pages show the selected draft (`Context.draft`): the **session page**
(mouse, date, folder, the session's annotations, the odor panel, and the
session's actions) and the **experiment page** (one experiment folder's
settings, annotations and trial block calculator). `Sessions` owns the drafts
and the actions that both the pages and the window's right bar call; what
Check and Submit find goes out through `Sessions.reported`.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .context import Context
from .draft import EXPERIMENT_SETTINGS, Draft, Removals
from .fields import Field, form_fields
from .forms import Form, PanelPicker, TimingForm
from .pane_registry import current_vials
from .style import PAGE_MAX_PX, required_label, text_style
from .submit import check, submit

TIMING_KEYS = (
    "block_baseline_s",
    "block_odor_s",
    "block_post_odor_delay_s",
    "block_iti_s",
    "block_post_odor_acq_s",
    "block_n_odors",
    "block_n_trials",
)


def centered(inner: QWidget) -> QScrollArea:
    """`inner` in a scroll area, no wider than `PAGE_MAX_PX` and centered."""
    inner.setMaximumWidth(PAGE_MAX_PX)
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    # The page grows first; the stretches only take what is past its maximum
    row.addStretch(0)
    row.addWidget(inner, 1)
    row.addStretch(0)

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(holder)

    return area


def session_label(draft: Draft, position: int) -> str:
    """How the right bar names a session: its mouse, or `s<n>` before one."""
    return f"s{position + 1}" if draft.mouse_id is None else f"m{draft.mouse_id}"


class ExperimentForm(QWidget):
    """One experiment folder of a session: its settings and annotations."""

    def __init__(self, draft: Draft, name: str, fields: list[Field], parent=None):
        super().__init__(parent)
        self.draft = draft
        self.name = name

        entry = draft.experiment(name) or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # The objective is the form's first row; the block keys are drawn by
        # the calculator instead of as plain rows
        plain = [f for f in fields if f.key not in TIMING_KEYS]
        self.form = Form(plain)
        self.form.set_values(
            {k: v for k, v in entry.items() if k not in EXPERIMENT_SETTINGS}
            | {"objective": entry.get("objective")}
        )
        self.form.changed.connect(lambda key, value: self._set(**{key: value}))

        self.timing = TimingForm()
        self.timing.set_values(entry)
        self.timing.changed.connect(lambda: self._set(**self.timing.values()))

        row = QHBoxLayout()
        row.addWidget(self.form, 1, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.timing, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

        self.submitted = QLabel()
        self.submitted.setStyleSheet("color: #2a7;")
        layout.addWidget(self.submitted)
        self.refresh_status()

    def refresh_status(self) -> None:
        entry = self.draft.experiment(self.name) or {}
        exp_id = entry.get("submitted_exp_id")
        self.submitted.setText(
            "" if exp_id is None else f"In the database as exp_id {exp_id}."
        )

    def _set(self, **fields) -> None:
        self.draft.set_experiment(self.name, **fields)


class SessionPage(QWidget):
    """The selected draft's session: who, when, what was noted, which vials."""

    def __init__(self, sessions: Sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions
        self.context = sessions.context
        self.draft: None | Draft = None
        self.form: None | Form = None

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # The session's identity, as one row
        self.identity = QGroupBox("Session")
        row = QHBoxLayout(self.identity)
        self.mouse = QComboBox()
        self.mouse.setEditable(True)
        self.mouse.setMinimumWidth(120)
        self.mouse.currentTextChanged.connect(self._mouse_changed)
        self.date = QLineEdit()
        self.date.setPlaceholderText("YYYY-MM-DD")
        self.date.setMaximumWidth(130)
        self.date.editingFinished.connect(self._date_changed)
        self.folder = QLabel()
        self.folder.setStyleSheet(text_style("computed"))

        for label, widget in (
            (required_label("Mouse"), self.mouse),
            (required_label("Date"), self.date),
            (QLabel("Folder"), self.folder),
        ):
            row.addWidget(label)
            row.addWidget(widget)
            row.addSpacing(18)

        row.addStretch(1)
        layout.addWidget(self.identity)

        self.info = QGroupBox("Info")
        self.info_layout = QVBoxLayout(self.info)
        layout.addWidget(self.info)

        vials = QGroupBox("Odor panel")
        vials_layout = QVBoxLayout(vials)
        self.vial_list = PanelPicker()
        self.vial_list.changed.connect(self._vials_changed)
        vials_layout.addWidget(self.vial_list)
        layout.addWidget(vials)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(centered(inner))
        self.body = inner
        self.show_draft(None)

    def show_draft(self, draft: None | Draft) -> None:
        """Fill the page from `draft`; with None, an empty, disabled page."""
        self.draft = draft
        self.body.setEnabled(draft is not None)

        self.mouse.blockSignals(True)
        self.mouse.clear()
        self.mouse.addItems(["", *[f"m{m}" for m in self.sessions.mice]])
        self.mouse.setCurrentText(
            "" if draft is None or draft.mouse_id is None else f"m{draft.mouse_id}"
        )
        self.mouse.blockSignals(False)
        self.date.setText("" if draft is None else draft.date)
        self._refresh_identity()

        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()
            self.form = None

        if draft is not None:
            self.form = Form(self.sessions.session_fields)
            self.form.set_values(draft.session)
            self.form.changed.connect(
                lambda key, value: draft.update_session(**{key: value})
            )
            self.info_layout.addWidget(self.form)

        self.vial_list.blockSignals(True)
        self.vial_list.set_panels(self.sessions.panels, self.sessions.vials)
        self.vial_list.show_vials([] if draft is None else draft.vial_ids)
        self.vial_list.blockSignals(False)

    def _refresh_identity(self) -> None:
        draft = self.draft

        if draft is None:
            self.identity.setTitle("No session selected: pick or add one on the right")
            self.folder.setText("")
        elif draft.mouse_id is None:
            self.identity.setTitle(f"Session {draft.where}")
            self.folder.setText("pick a mouse")
        else:
            self.identity.setTitle(f"Session {draft.where}")
            self.folder.setText(draft.folder)

    def _mouse_changed(self, text: str) -> None:
        if self.draft is None:
            return

        text = text.strip().lstrip("mM")

        try:
            mouse_id = int(text) if text else None
        except ValueError:
            return

        if mouse_id is not None:
            self.draft.identify(mouse_id=mouse_id)
            self._refresh_identity()
            self.context.announce_draft()

    def _date_changed(self) -> None:
        if self.draft is None or self.date.text() == self.draft.date:
            return

        try:
            self.draft.identify(date=self.date.text())
        except ValueError as error:
            self.context.report("Date", str(error))
            return

        self._refresh_identity()
        self.context.announce_draft()

    def _vials_changed(self) -> None:
        if self.draft is not None:
            self.draft.set_vials(self.vial_list.checked())


class ExperimentPage(QWidget):
    """One experiment of the selected draft, and the button to remove it."""

    def __init__(self, sessions: Sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions
        self.draft: None | Draft = None
        self.name: None | str = None
        self.shown: None | ExperimentForm = None

        inner = QWidget()
        layout = QVBoxLayout(inner)
        self.box = QGroupBox("Experiment")
        self.box_layout = QVBoxLayout(self.box)
        layout.addWidget(self.box)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(centered(inner))

    def show_experiment(self, draft: None | Draft, name: None | str) -> None:
        """Fill the page from experiment `name` of `draft`."""
        if self.shown is not None:
            self.shown.setParent(None)
            self.shown.deleteLater()
            self.shown = None

        self.draft, self.name = draft, name

        if draft is None or name is None or draft.experiment(name) is None:
            self.box.setTitle("Experiment")
            return

        self.box.setTitle(f"Experiment {draft.where}/{name}")
        self.shown = ExperimentForm(draft, name, self.sessions.experiment_fields)
        self.box_layout.addWidget(self.shown)


class Sessions(QObject):
    """
    The drafts in progress and what can be done to them.

    `drafts` is in the order the sessions were started. The pages and the
    window's right bar call `new`, `remove`, `add_experiment`,
    `remove_experiment`, `undo`, `check` and `submit`; each acts on the
    selected draft (`Context.draft`). Removing only marks the draft or the
    experiment removed and pushes it on `removals`, the undo stack.

    - `listed` fires when drafts come or go (the right bar rebuilds);
    - `removals_changed` fires when the undo stack may have changed;
    - `reported(text)` carries what Check or Submit found;
    - `experiment_added(name)` fires once a new experiment is selected.
    """

    listed = Signal()
    removals_changed = Signal()
    reported = Signal(str)
    experiment_added = Signal(str)

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.drafts: list[Draft] = []
        self.removals = Removals(context.drafts)
        self.session_fields: list[Field] = []
        self.experiment_fields: list[Field] = []
        self.mice: list[int] = []
        self.panels: list[tuple[int, str]] = []
        self.vials = None

        self.session_page = SessionPage(self)
        self.experiment_page = ExperimentPage(self)

        context.changed.connect(self.reload_choices)
        context.selection_changed.connect(self._follow_selection)
        context.draft_changed.connect(self._follow_draft)
        self.reload_choices()
        self.reload_drafts()

    @property
    def current(self) -> None | Draft:
        """The selected draft, when it is one of these."""
        draft = self.context.draft

        return draft if any(draft is d for d in self.drafts) else None

    def position(self, draft: Draft) -> int:
        return next(i for i, d in enumerate(self.drafts) if d is draft)

    def label(self, draft: Draft) -> str:
        return session_label(draft, self.position(draft))

    # ----------------------------------------------------------------- #
    # Loading
    # ----------------------------------------------------------------- #

    def reload_choices(self) -> None:
        """What the forms offer: the registry's fields, the mice, the vials."""

        def load(db):
            vials = db.vials.join(db.panels["panel_name"], on="panel_id")
            vials = vials.join(db.odors["odor_name"], on="odor_id")
            panels = [(int(i), str(n)) for i, n in db.panels["panel_name"].items()]

            return (
                form_fields(db, "session"),
                form_fields(db, "experiment"),
                [int(m) for m in db.mice.index],
                (panels, current_vials(vials)),
            )

        self.context.run(load, self._choices_loaded, "session choices")

    def _choices_loaded(self, loaded) -> None:
        session_fields, experiment_fields, mice, (panels, vials) = loaded
        self.session_fields = [
            f for f in session_fields if f.key not in ("mouse_id", "session_date")
        ]
        self.experiment_fields = experiment_fields
        self.mice = mice
        self.panels, self.vials = panels, vials
        self.refresh_current()

    def reload_drafts(self) -> None:
        """Read the drafts from disk; select the first unless one still is."""
        self.drafts = sorted(
            Draft.pending(self.context.drafts), key=lambda d: d.path.name
        )
        kept = next(
            (
                d
                for d in self.drafts
                if self.context.draft is not None and d.path == self.context.draft.path
            ),
            None,
        )
        self.listed.emit()
        self.context.select(kept or (self.drafts[0] if self.drafts else None))
        self.refresh_current()

    def refresh_current(self) -> None:
        """Redraw both pages from the selected draft."""
        self.session_page.show_draft(self.current)
        self.experiment_page.show_experiment(self.current, self.context.experiment)

    def _follow_selection(self) -> None:
        draft = self.current

        if self.session_page.draft is not draft:
            self.session_page.show_draft(draft)

        page = self.experiment_page

        if page.draft is not draft or page.name != self.context.experiment:
            page.show_experiment(draft, self.context.experiment)

    def _follow_draft(self) -> None:
        """The selected draft changed (mouse, date, experiments)."""
        draft = self.current
        page = self.experiment_page

        if draft is not None and page.name != self.context.experiment:
            page.show_experiment(draft, self.context.experiment)
        elif draft is not None and page.shown is not None:
            page.box.setTitle(f"Experiment {draft.where}/{page.name}")

    # ----------------------------------------------------------------- #
    # Actions
    # ----------------------------------------------------------------- #

    def new(self) -> Draft:
        """Start a session, listed last, and select it."""
        draft = Draft.start(self.context.drafts, user=self.context.user)
        self.drafts.append(draft)
        self.listed.emit()
        self.context.select(draft)

        return draft

    def remove(self) -> None:
        """
        Remove the selected draft (can be undone). Nothing is selected
        afterwards, so a second click cannot remove another session.
        """
        draft = self.current

        if draft is None:
            return

        draft.remove()
        self.removals.push(draft)
        self.drafts.pop(self.position(draft))
        self.listed.emit()
        self.context.select(None)
        self.refresh_current()
        self.removals_changed.emit()

    def undo(self) -> None | tuple[Draft, None | str]:
        """
        Bring back the latest removed session or experiment and select it;
        returns `(draft, experiment or None)`, None if nothing was left.
        """
        restored = self.removals.undo(self.drafts)
        self.removals_changed.emit()

        if restored is None:
            return None

        draft, experiment = restored

        if experiment is None:
            self.drafts.append(draft)
            self.drafts.sort(key=lambda d: d.path.name)
            self.listed.emit()
            self.context.select(draft)
        else:
            self.context.select(draft, experiment)
            self.context.announce_draft()

        self.refresh_current()

        return restored

    def next_undo(self) -> None | str:
        """What undo would bring back, as the bottom bar says it; None if nothing."""
        found = self.removals.next()

        if found is None:
            return None

        draft, experiment = found

        if experiment is None:
            return f"Undo remove: session {draft.where}"

        return f"Undo remove: experiment {draft.where}/{experiment}"

    def add_experiment(self, name: None | str = None, exp_type: str = "loop") -> None:
        """
        Add an experiment folder to the selected draft, asking for its name
        (`e<n+1>` offered) unless given, and select it.
        """
        draft = self.current

        if draft is None:
            self.context.report("Experiment", "Add a session first.")
            return

        if name is None:
            # Removed experiments still hold their names
            numbers = [
                int(taken[1:])
                for taken in draft.names_taken()
                if taken.startswith("e") and taken[1:].isdigit()
            ]
            name, ok = QInputDialog.getText(
                self.context.window,
                "Folder name",
                "Experiment folder:",
                text=f"e{max(numbers, default=0) + 1}",
            )

            if not ok:
                return

        name = name.strip()

        if not name:
            return

        if draft.experiment(name) is not None:
            self.context.report("Experiment", f"'{name}' is already listed.")
            return

        if draft.is_removed(name):
            self.context.report(
                "Experiment",
                f"'{name}' was removed. Pick another name.",
            )
            return

        draft.set_experiment(name, exp_type=exp_type)
        self.context.announce_draft()
        self.context.select(draft, name)
        self.experiment_added.emit(name)

    def remove_experiment(self, name: None | str = None) -> None:
        """Remove experiment `name` (the selected one if None) from the draft."""
        draft = self.current
        name = name or self.context.experiment

        if draft is None or name is None or draft.experiment(name) is None:
            return

        draft.remove_experiment(name)
        self.removals.push(draft, name)
        self.context.announce_draft()
        self.removals_changed.emit()

    # ----------------------------------------------------------------- #
    # Check and submit
    # ----------------------------------------------------------------- #

    def check(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.reported.emit(f"Checking {draft.label}...")
        self.context.run(
            lambda db: check(db, draft),
            lambda problems: self.reported.emit(
                f"{draft.label}\n"
                + (
                    "\n".join(str(p) for p in problems)
                    or "Nothing wrong: ready to submit."
                )
            ),
            "check",
        )

    def submit(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.reported.emit(
            f"Submitting {draft.label}... (reading the recordings can take a while)"
        )
        self.context.run(
            lambda db: submit(db, draft),
            lambda result: self._submitted(draft, result),
            "submit",
            writes=True,
        )

    def _submitted(self, draft: Draft, result) -> None:
        lines = [str(p) for p in result.problems]

        if result.written is None:
            lines.insert(0, "Not submitted: fix what is listed, then try again.")

        else:
            written = result.written
            lines.insert(
                0,
                f"Session {written.session_id}: {len(written.experiments)} experiments, "
                f"{written.annotations} annotations, sync decoded for "
                f"{', '.join(written.decoded) or 'none'}.",
            )

            for name, message in written.decode_failed.items():
                lines.append(f"{name}: sync decode failed: {message}")

            if result.error:
                lines.append(
                    f"Stopped: {result.error}. The draft is kept; submit again after fixing it."
                )

        self.reported.emit(f"{draft.label}\n" + "\n".join(lines))

        if result.submitted:
            self.reload_drafts()
        elif self.experiment_page.shown is not None:
            self.experiment_page.shown.refresh_status()
