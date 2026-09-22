"""
The day's work: one draft per session, filled in as the experiment runs and
submitted at the end.

Left: the pending drafts, and a button for a new one. Right: the session's
mouse and date, its annotations, one tab per experiment folder (settings,
annotations, the trial block calculator), the vials loaded, and Check /
Submit with what they found.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .context import Context
from .draft import EXPERIMENT_SETTINGS, Draft
from .fields import Field, form_fields
from .forms import Form, PanelPicker, TimingForm
from .pane_registry import current_vials
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


class ExperimentTab(QWidget):
    """One experiment folder of the session: its settings and annotations."""

    def __init__(self, draft: Draft, name: str, fields: list[Field], parent=None):
        super().__init__(parent)
        self.draft = draft
        self.name = name

        entry = draft.experiment(name) or {}

        layout = QVBoxLayout(self)

        # The kind of recording is what the button that made the tab said; the
        # objective is the form's first row, the block keys are drawn by the
        # calculator instead of as plain rows
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
        row.addWidget(self.form, 1)
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


class SessionPane(QWidget):
    """The drafts and the one being filled in."""

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.drafts: list[Draft] = []
        self.current: None | Draft = None
        self.session_fields: list[Field] = []
        self.experiment_fields: list[Field] = []
        self.mice: list[int] = []
        self.panels: list[tuple[int, str]] = []
        self.vials = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # Left: the drafts
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Sessions in progress"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        left_layout.addWidget(self.list, 1)

        new = QPushButton("New session")
        new.clicked.connect(self._new)
        discard = QPushButton("Discard session")
        discard.clicked.connect(self._discard)
        left_layout.addWidget(new)
        left_layout.addWidget(discard)
        splitter.addWidget(left)

        # Right: the draft
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.right = QWidget()
        self.right_layout = QVBoxLayout(self.right)
        scroll.setWidget(self.right)
        splitter.addWidget(scroll)
        splitter.setSizes([250, 900])

        identity = QGroupBox("Session")
        identity_form = QFormLayout(identity)
        self.mouse = QComboBox()
        self.mouse.setEditable(True)
        self.mouse.currentTextChanged.connect(self._mouse_changed)
        self.date = QLineEdit()
        self.date.setPlaceholderText("YYYY-MM-DD")
        self.date.editingFinished.connect(self._date_changed)
        self.folder = QLabel()
        identity_form.addRow("Mouse *", self.mouse)
        identity_form.addRow("Date *", self.date)
        identity_form.addRow("Folder", self.folder)
        self.right_layout.addWidget(identity)

        self.session_box = QGroupBox("Info")
        self.session_box_layout = QVBoxLayout(self.session_box)
        self.session_form: None | Form = None
        self.right_layout.addWidget(self.session_box)

        experiments = QGroupBox("Experiments")
        experiments_layout = QVBoxLayout(experiments)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        experiments_layout.addWidget(self.tabs)
        buttons = QHBoxLayout()
        add_loop = QPushButton("Add loop")
        add_loop.clicked.connect(lambda: self._add_experiment("loop"))
        add_grabs = QPushButton("Add grabs folder")
        add_grabs.clicked.connect(lambda: self._add_experiment("grab"))
        add_stacks = QPushButton("Add z-stacks folder")
        add_stacks.clicked.connect(lambda: self._add_experiment("zstack"))
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_experiment)

        # What a grab or a z-stack needs is not settled yet
        for button in (add_grabs, add_stacks):
            button.setEnabled(False)
            button.setToolTip("Not available yet.")
        for button in (add_loop, add_grabs, add_stacks):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(remove)
        experiments_layout.addLayout(buttons)
        self.right_layout.addWidget(experiments)

        vials = QGroupBox("Odor panel")
        vials_layout = QVBoxLayout(vials)
        self.vial_list = PanelPicker()
        self.vial_list.changed.connect(self._vials_changed)
        vials_layout.addWidget(self.vial_list)
        self.right_layout.addWidget(vials)

        actions = QHBoxLayout()
        check_button = QPushButton("Check")
        check_button.clicked.connect(self._check)
        submit_button = QPushButton("Submit session")
        submit_button.clicked.connect(self._submit)
        actions.addWidget(check_button)
        actions.addWidget(submit_button)
        actions.addStretch(1)
        self.right_layout.addLayout(actions)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(140)
        self.right_layout.addWidget(self.report)
        self.right_layout.addStretch(1)

        self.right.setEnabled(False)

        context.changed.connect(self.reload_choices)
        context.selection_changed.connect(self._follow_selection)
        context.draft_changed.connect(self._follow_draft)
        self.reload_choices()
        self.reload_drafts()

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
        self.vial_list.blockSignals(True)
        self.vial_list.set_panels(panels, vials)
        self.vial_list.blockSignals(False)

        current = self.mouse.currentText()
        self.mouse.blockSignals(True)
        self.mouse.clear()
        self.mouse.addItems(["", *[f"m{m}" for m in mice]])
        self.mouse.setCurrentText(current)
        self.mouse.blockSignals(False)

        if self.current is not None:
            self._show(self.current)

    def reload_drafts(self) -> None:
        self.drafts = Draft.pending(self.context.drafts)
        self.list.blockSignals(True)
        self.list.clear()

        for draft in self.drafts:
            QListWidgetItem(draft.label, self.list)

        self.list.blockSignals(False)

        if self.drafts:
            self.list.setCurrentRow(0)
        else:
            self.current = None
            self.right.setEnabled(False)
            self.context.select(None)

    def _select(self, row: int) -> None:
        if row < 0 or row >= len(self.drafts):
            self.current = None
            self.right.setEnabled(False)
            self.context.select(None)
            return

        self._show(self.drafts[row])

    def refresh_current(self) -> None:
        """Redraw the shown draft after something else (the recorder) changed it."""
        if self.current is not None:
            self._show(self.current)

    def _follow_draft(self) -> None:
        """The draft changed elsewhere: redraw if its experiments differ."""
        draft = self.current

        if draft is None or draft is not self.context.draft:
            return

        shown = [
            self.tabs.widget(i).name
            for i in range(self.tabs.count())
            if isinstance(self.tabs.widget(i), ExperimentTab)
        ]

        if shown != [str(e["name"]) for e in draft.experiments]:
            self._show(draft)

    def _show(self, draft: Draft) -> None:
        self.current = draft
        self.right.setEnabled(True)

        self.mouse.blockSignals(True)
        self.mouse.setCurrentText(
            "" if draft.mouse_id is None else f"m{draft.mouse_id}"
        )
        self.mouse.blockSignals(False)
        self.date.setText(draft.date)
        self._refresh_folder()

        if self.session_form is not None:
            self.session_form.setParent(None)
            self.session_form.deleteLater()

        self.session_form = Form(self.session_fields)
        self.session_form.set_values(draft.session)
        self.session_form.changed.connect(
            lambda key, value: draft.update_session(**{key: value})
        )
        self.session_box_layout.addWidget(self.session_form)

        self.tabs.blockSignals(True)
        self.tabs.clear()
        for entry in draft.experiments:
            tab = ExperimentTab(draft, str(entry["name"]), self.experiment_fields)
            self.tabs.addTab(tab, str(entry["name"]))
        self.tabs.blockSignals(False)

        self.vial_list.blockSignals(True)
        self.vial_list.show_vials(draft.vial_ids)
        self.vial_list.blockSignals(False)
        self.report.clear()

        # This draft is now the one every tab works on; keep its experiment
        # if it is still there, else the last one
        wanted = self.context.experiment if self.context.draft is draft else None
        self.context.select(draft, wanted)
        self._follow_selection()

    def _refresh_folder(self) -> None:
        if self.current is None or self.current.mouse_id is None:
            self.folder.setText("pick a mouse")
            return

        self.folder.setText(self.current.folder)

    # ----------------------------------------------------------------- #
    # Editing
    # ----------------------------------------------------------------- #

    def _new(self) -> None:
        draft = Draft.start(self.context.drafts, user=self.context.user)
        self.drafts.insert(0, draft)
        self.list.insertItem(0, QListWidgetItem(draft.label))
        self.list.setCurrentRow(0)

    def _discard(self) -> None:
        if self.current is None:
            return

        self.current.discard()
        self.reload_drafts()

    def _mouse_changed(self, text: str) -> None:
        if self.current is None:
            return

        text = text.strip().lstrip("mM")

        try:
            mouse_id = int(text) if text else None
        except ValueError:
            return

        if mouse_id is not None:
            self.current.identify(mouse_id=mouse_id)
            self._refresh_folder()
            self.list.currentItem().setText(self.current.label)
            self.context.announce_draft()

    def _date_changed(self) -> None:
        if self.current is None:
            return

        try:
            self.current.identify(date=self.date.text())
        except ValueError as error:
            self.context.report("Date", str(error))
            return

        self._refresh_folder()
        self.list.currentItem().setText(self.current.label)
        self.context.announce_draft()

    def _add_experiment(self, exp_type: str) -> None:
        if self.current is None:
            return

        if exp_type == "loop":
            numbers = [
                int(e["name"][1:])
                for e in self.current.experiments
                if str(e["name"]).startswith("e") and str(e["name"])[1:].isdigit()
            ]
            default = f"e{max(numbers, default=0) + 1}"
        else:
            default = "grabs" if exp_type == "grab" else "zstacks"

        name, ok = QInputDialog.getText(
            self, "Folder name", "Experiment folder:", text=default
        )
        name = name.strip()

        if not ok or not name:
            return

        if self.current.experiment(name) is not None:
            self.context.report("Experiment", f"'{name}' is already listed.")
            return

        self.current.set_experiment(name, exp_type=exp_type)
        tab = ExperimentTab(self.current, name, self.experiment_fields)
        self.tabs.addTab(tab, name)
        self.context.announce_draft()
        self.tabs.setCurrentWidget(tab)  # selects it through `_tab_changed`

    def _remove_experiment(self) -> None:
        tab = self.tabs.currentWidget()

        if self.current is None or not isinstance(tab, ExperimentTab):
            return

        self.current.remove_experiment(tab.name)
        self.tabs.removeTab(self.tabs.currentIndex())
        self.context.announce_draft()
        self._tab_changed(self.tabs.currentIndex())

    def _tab_changed(self, index: int) -> None:
        """The experiment tab in front is the experiment being worked on."""
        tab = self.tabs.widget(index)

        if self.current is not None and self.current is self.context.draft:
            self.context.select_experiment(
                tab.name if isinstance(tab, ExperimentTab) else None
            )

    def _follow_selection(self) -> None:
        """Bring the experiment picked elsewhere (the recorder) to the front."""
        if self.context.draft is not self.current:
            return

        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)

            if isinstance(tab, ExperimentTab) and tab.name == self.context.experiment:
                if self.tabs.currentIndex() != index:
                    self.tabs.blockSignals(True)
                    self.tabs.setCurrentIndex(index)
                    self.tabs.blockSignals(False)

                return

    def _vials_changed(self) -> None:
        if self.current is not None:
            self.current.set_vials(self.vial_list.checked())

    # ----------------------------------------------------------------- #
    # Check and submit
    # ----------------------------------------------------------------- #

    def _check(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.report.setPlainText("Checking...")
        self.context.run(
            lambda db: check(db, draft),
            lambda problems: self.report.setPlainText(
                "\n".join(str(p) for p in problems) or "Nothing wrong: ready to submit."
            ),
            "check",
        )

    def _submit(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.report.setPlainText(
            "Submitting... (reading the recordings can take a while)"
        )
        self.context.run(
            lambda db: submit(db, draft), self._submitted, "submit", writes=True
        )

    def _submitted(self, result) -> None:
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

        self.report.setPlainText("\n".join(lines))

        if result.submitted:
            self.reload_drafts()

        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if isinstance(tab, ExperimentTab):
                tab.refresh_status()
