"""
Forms built from `Field`s, and the two calculator forms.

`Form` draws one row per field -- a line, a box, a dropdown or a yes/no --
and reads them back as typed values. `RecipeForm` and `TimingForm` are the
spreadsheets the lab kept: inputs on the left, what follows from them shown
as they change, through `odyn.recipes`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .style import ROW_SPACING_PX, required_label, text_style
from ..recipes import block_timing, camera_frames, solvent_volume_ml, vial_recipe
from .fields import BOOLEAN, ENUM, LONG_TEXT, Field


class WheelGuard(QObject):
    """
    The mouse wheel never changes number boxes and dropdowns; it scrolls the
    area they sit in.

    Installed once on the application (`QApplication.installEventFilter`).
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel or not isinstance(
            watched, (QAbstractSpinBox, QComboBox)
        ):
            return False

        area = watched.parentWidget()

        while area is not None and not isinstance(area, QAbstractScrollArea):
            area = area.parentWidget()

        if area is not None:
            QApplication.sendEvent(area.viewport(), event.clone())

        return True  # the box itself never sees it


class FieldRow(QWidget):
    """One field: its editor, and an inline note when what was typed is wrong."""

    changed = Signal(str, object)  # key, parsed value (None when blank)

    def __init__(self, field: Field, parent=None):
        super().__init__(parent)
        self.field = field
        self._value: Any = None
        self._wrong = False  # what was typed does not read as the field's type

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.editor: QWidget
        self._read: Callable[[], str]
        self._write: Callable[[str], None]

        if field.value_type == BOOLEAN:
            # A box, unchecked (False) until someone says otherwise
            box = QCheckBox()
            box.toggled.connect(
                lambda checked: self._edited("yes" if checked else "no")
            )
            self._value = False
            self.editor, self._read, self._write = (
                box,
                lambda: "yes" if box.isChecked() else "no",
                lambda text: box.setChecked(
                    text.strip().lower() in ("yes", "true", "1")
                ),
            )

        elif field.value_type == ENUM or field.options:
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(["", *field.options])
            combo.currentTextChanged.connect(self._edited)
            self.editor, self._read, self._write = (
                combo,
                combo.currentText,
                combo.setCurrentText,
            )

        elif field.value_type == LONG_TEXT:
            box = QPlainTextEdit()
            box.setPlaceholderText(field.description)
            box.setFixedHeight(90)
            box.textChanged.connect(lambda: self._edited(box.toPlainText()))
            self.editor, self._read, self._write = (
                box,
                box.toPlainText,
                box.setPlainText,
            )

        else:
            line = QLineEdit()
            line.setPlaceholderText(field.description)
            line.editingFinished.connect(lambda: self._edited(line.text()))
            self.editor, self._read, self._write = line, line.text, line.setText

        self.editor.setToolTip(field.description)
        self.editor.setEnabled(field.editable)

        if not field.editable:
            self.editor.setStyleSheet(text_style("computed"))

        self.note = QLabel()
        self.note.setStyleSheet(text_style("error"))
        self.note.setVisible(False)

        layout.addWidget(self.editor)
        layout.addWidget(self.note)

    def label(self) -> QLabel:
        """The row's label: in the error color when the field is required."""
        return required_label(self.field.prompt, self.field.required)

    def value(self) -> Any:
        """The typed value, or `None` when blank or unreadable."""
        return self._value

    def set_value(self, value: Any) -> None:
        """Show a stored value without announcing a change."""
        if self.field.value_type == BOOLEAN:
            value = bool(value)  # a box is never blank

        self._value, self._wrong = value, False
        self.editor.blockSignals(True)
        self._write(self.field.show(value))
        self.editor.blockSignals(False)
        self.note.setVisible(False)

    def show_note(self, text: str, kind: str = "error") -> None:
        """A line under the editor, in `kind`'s color (see `text_style`)."""
        self.note.setStyleSheet(text_style(kind))
        self.note.setText(text)
        self.note.setVisible(True)

    def problem(self) -> None | str:
        """What is wrong with what was typed, if anything."""
        return self.note.text() if self._wrong else None

    def commit(self) -> None:
        """
        Read what is typed now, without announcing it: a line is otherwise
        read when it loses focus, which clicking a button on macOS does not do.
        """
        self._read_text(self._read())

    def _edited(self, text: str) -> None:
        if self._read_text(text):
            self.changed.emit(self.field.key, self._value)

    def _read_text(self, text: str) -> bool:
        try:
            self._value = self.field.parse(text)

        except ValueError as error:
            self.show_note(str(error))
            self._value, self._wrong = None, True
            return False

        self.note.setVisible(False)
        self._wrong = False

        return True


class Form(QWidget):
    """
    A form with one `FieldRow` per field.

    `changed(key, value)` fires on every edit that reads as its type;
    `values()` is the whole form as typed values, blanks left out.
    """

    changed = Signal(str, object)

    def __init__(self, fields: list[Field], parent=None):
        super().__init__(parent)
        self.rows: dict[str, FieldRow] = {}

        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setVerticalSpacing(ROW_SPACING_PX)

        for field in fields:
            row = FieldRow(field)
            row.changed.connect(self.changed)
            self.rows[field.key] = row
            layout.addRow(row.label(), row)

    def values(self) -> dict[str, Any]:
        return {
            key: row.value()
            for key, row in self.rows.items()
            if row.value() is not None
        }

    def set_values(self, values: dict[str, Any]) -> None:
        for key, row in self.rows.items():
            row.set_value(values.get(key))

    def clear(self) -> None:
        self.set_values({})

    def commit(self) -> None:
        """Read every field as typed now (see `FieldRow.commit`)."""
        for row in self.rows.values():
            row.commit()

    def problems(self) -> list[str]:
        """What is wrong with what was typed, one line per field."""
        return [p for row in self.rows.values() if (p := row.problem()) is not None]


class FormRows:
    """
    A list whose entries are rows of a form (a mouse's lines, a mix's
    components, a loop's conditions).

    Each entry is a row of `form`, so it is spaced and its label placed like
    any other field's: `title` labels the first, the rest have an empty
    label. With no entries, a row saying `none_text` holds the list's place.
    A `header` (column names, say) goes above the entries, titled instead of
    the first, and shows only when there are entries.
    """

    def __init__(
        self,
        form: QFormLayout,
        title: str,
        none_text: str = "none",
        header: None | QWidget = None,
    ):
        self.form = form
        self.title = title
        self.header = header
        self.rows: list[QWidget] = []
        self.none = shown_value(kind=None)
        self.none.setPlaceholderText(none_text)
        form.addRow(title, self.none)

        if header is not None:
            form.addRow(QLabel(title), header)
            form.setRowVisible(header, False)

    def insert(self, row: QWidget) -> None:
        """A row below the list's last one (or its header, or "none" row)."""
        last = self.rows[-1] if self.rows else (self.header or self.none)
        at = self.form.getWidgetPosition(last)[0] + 1
        self.form.insertRow(at, QLabel(), row)  # (text "" would give no label)
        self.rows.append(row)
        self._relabel()

    def take(self, row: QWidget) -> None:
        """
        Take a row out. Taken, not removed: it may still be running its own
        remove button's click, so it is deleted later.
        """
        self.rows.remove(row)
        taken = self.form.takeRow(self.form.getWidgetPosition(row)[0])

        for item in (taken.labelItem, taken.fieldItem):
            item.widget().hide()
            item.widget().deleteLater()

        self._relabel()

    def _relabel(self) -> None:
        """The title on the first row; "none" shown only when there is none."""
        self.form.setRowVisible(self.none, not self.rows)

        if self.header is not None:
            self.form.setRowVisible(self.header, bool(self.rows))

        for i, row in enumerate(self.rows):
            first = i == 0 and self.header is None
            self.form.labelForField(row).setText(self.title if first else "")


def fit_row(widgets, height: int) -> None:
    """
    Widgets side by side in a form row, as tall as one field: buttons set to
    `height`, and every one laid out by its own size (macOS otherwise pads
    each by the room its native look would need around it).
    """
    for widget in widgets:
        if widget is None:
            continue

        if isinstance(widget, QPushButton):
            widget.setFixedHeight(height)

        widget.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)


# --------------------------------------------------------------------------- #
# Calculators
# --------------------------------------------------------------------------- #


def _spin(
    minimum: float, maximum: float, value: float, decimals: int = 2
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setValue(value)

    return box


class ComponentRow(QWidget):
    """
    One chemical in a vial: fixed by the odor, only its target is typed. The
    row is the target and the computed volume; `label` names the chemical,
    for the vial's form to show beside it.
    """

    changed = Signal()

    def __init__(
        self, odor_id: int, name: str, vapor_pressure: None | float, parent=None
    ):
        super().__init__(parent)
        self.odor_id = odor_id
        self.vapor_pressure = vapor_pressure

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        note = "" if vapor_pressure else " (no vapor pressure)"
        self.label = QLabel(f"{odor_id} - {name}{note}")
        self.label.setWordWrap(True)

        self.target = _spin(0, 1e6, 1.0, 3)
        self.target.setSuffix(" ppm")

        self.result = QLabel("")
        self.result.setStyleSheet(text_style("computed"))

        layout.addWidget(self.target, 1)
        layout.addWidget(self.result, 1)

        self.target.valueChanged.connect(lambda _: self.changed.emit())

    def compute(self, vial_sccm: float, total_sccm: float, total_volume_ml: float):
        """The recipe for this component, or `None` when it cannot be made."""
        pressure = self.vapor_pressure

        if not pressure or self.target.value() <= 0:
            self.result.setText("needs a vapor pressure and a target")
            return None

        recipe = vial_recipe(
            target_ppm=self.target.value(),
            vapor_pressure_mmhg=pressure,
            vial_sccm=vial_sccm,
            total_sccm=total_sccm,
            total_volume_ml=total_volume_ml,
        )
        self.result.setText(f"{recipe.liquid_ul:.0f} µl ({recipe.percent_vv:.3g} %)")

        return recipe


class RecipeForm(QGroupBox):
    """
    One vial: its position, what the olfactometer delivers, the flows and
    volume, and the target of each component -- with the volumes to pipette
    computed as the targets are typed.

    What goes in the vial follows the odor picked: a mix's components are
    its `odor_components`, a single chemical is its own one component, and a
    chemical with no vapor pressure (mineral oil) has nothing to pipette.

    A vial of a new panel can be set up freely and removed (`remove`). A vial
    of a stored panel is `fix`ed: its position and odor are the panel's, and
    its box's check says whether it is remade in this batch.

    `vial()` gives the dict `Database.add_vials` takes, computed values
    included.
    """

    def __init__(self, odors, chemicals, made_of, parent=None):
        """
        `odors` is `(odor_id, name)` for everything the olfactometer can
        deliver; `chemicals` maps a single chemical's id to
        `(name, vapor_pressure)`; `made_of` maps a mix's id to its
        components' ids.
        """
        super().__init__("Vial", parent)
        self.chemicals = chemicals
        self.made_of = made_of

        self.position = QSpinBox()
        self.position.setRange(1, 64)

        self.delivered = QComboBox()
        for odor_id, name in odors:
            self.delivered.addItem(f"{odor_id} - {name}", odor_id)

        self.vial_sccm = _spin(1, 10000, 200, 0)
        self.total_sccm = _spin(1, 100000, 1500, 0)
        self.total_volume = _spin(0.01, 1000, 5, 2)

        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form.setVerticalSpacing(ROW_SPACING_PX)
        self.form.addRow("Position", self.position)
        self.form.addRow("Delivers", self.delivered)
        self.form.addRow("Vial flow (sccm)", self.vial_sccm)
        self.form.addRow("Total flow (sccm)", self.total_sccm)
        self.form.addRow("Total volume (ml)", self.total_volume)

        # The components are rows of the same form, named on the left
        self.components: list[ComponentRow] = []

        self.oil = QLabel("")
        self.oil.setStyleSheet(text_style("computed"))
        self.form.addRow("Mineral oil", self.oil)

        self.remove = QPushButton("remove")
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(self.remove)

        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addLayout(foot)

        for box in (self.vial_sccm, self.total_sccm, self.total_volume):
            box.valueChanged.connect(lambda _: self.recompute())

        self.delivered.currentIndexChanged.connect(lambda _: self.set_components())
        self.set_components()

    @property
    def odor_id(self) -> None | int:
        data = self.delivered.currentData()

        return None if data is None else int(data)

    @property
    def included(self) -> bool:
        """Whether the vial is part of the batch (always, for a new panel)."""
        return not self.isCheckable() or self.isChecked()

    def fix(self, position: int, odor_id: int) -> None:
        """
        A vial of a stored panel: its position and odor are the panel's, it
        cannot be removed, and its check says whether it is remade now.
        """
        self.position.setValue(position)
        self.delivered.setCurrentIndex(max(self.delivered.findData(odor_id), 0))
        self.position.setEnabled(False)
        self.delivered.setEnabled(False)
        self.remove.hide()
        self.setTitle(f"Position {position}: {self.delivered.currentText()}")
        self.setCheckable(True)
        self.setChecked(True)
        self.setToolTip("Unchecked, this vial is not remade in this batch")

    def set_recipe(self, vial, targets: dict[int, float]) -> None:
        """Fill in a stored vial's flows, volume and targets (to remake it)."""
        self.vial_sccm.setValue(float(vial["vial_sccm"]))
        self.total_sccm.setValue(float(vial["total_sccm"]))
        self.total_volume.setValue(float(vial["total_volume_ml"]))
        self.set_targets(targets)

    def set_components(self) -> None:
        """Rebuild the component rows for the odor now picked."""
        for row in self.components:
            self.form.removeRow(row)  # (deletes it and its label)

        self.components = []
        odor_id = self.odor_id

        if odor_id is None:
            return

        ids = self.made_of.get(odor_id, [odor_id] if odor_id in self.chemicals else [])
        at = self.form.getWidgetPosition(self.oil)[0]

        for component_id in ids:
            name, pressure = self.chemicals.get(component_id, (str(component_id), None))

            if odor_id == component_id and not pressure:
                continue  # a solvent: nothing to pipette

            row = ComponentRow(component_id, name, pressure)
            row.changed.connect(self.recompute)
            self.components.append(row)
            self.form.insertRow(at, row.label, row)
            at += 1

        self.recompute()

    def set_targets(self, targets: dict[int, float]) -> None:
        """Type in the target of each component, by odor id."""
        for row in self.components:
            if row.odor_id in targets:
                row.target.setValue(float(targets[row.odor_id]))

    def recompute(self) -> None:
        volumes = []

        for row in self.components:
            recipe = row.compute(
                self.vial_sccm.value(),
                self.total_sccm.value(),
                self.total_volume.value(),
            )
            if recipe is not None:
                volumes.append(recipe.liquid_ul)

        try:
            oil = solvent_volume_ml(self.total_volume.value(), volumes)
            self.oil.setText(f"{oil:.3f} ml")

        except ValueError as error:
            self.oil.setText(str(error))

    def vial(self) -> dict[str, Any]:
        """What `Database.add_vials` takes for this vial."""
        components = []
        volumes = []

        for row in self.components:
            recipe = row.compute(
                self.vial_sccm.value(),
                self.total_sccm.value(),
                self.total_volume.value(),
            )
            component: dict[str, Any] = {
                "odor_id": row.odor_id,
                "target_ppm": row.target.value(),
            }

            if recipe is not None:
                component["liquid_ul"] = recipe.liquid_ul
                component["percent_vv"] = recipe.percent_vv
                volumes.append(recipe.liquid_ul)

            components.append(component)

        vial: dict[str, Any] = {
            "vial_position": self.position.value(),
            "odor_id": self.odor_id,
            "vial_sccm": self.vial_sccm.value(),
            "total_sccm": self.total_sccm.value(),
            "total_volume_ml": self.total_volume.value(),
            "components": components,
        }

        try:
            vial["solvent_volume_ml"] = solvent_volume_ml(
                self.total_volume.value(), volumes
            )
        except ValueError:
            pass

        return vial


def _heading(form: QFormLayout, text: str) -> None:
    """
    A section title in the label column, so it lines up with the labels
    below it (a row that spans both columns would sit at the left edge).
    """
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold; margin-top: 6px;")
    form.addRow(label, QWidget())


def shown_value(text: str = "", kind: None | str = "computed") -> QLineEdit:
    """
    A value that is shown, not typed: a read-only line without its frame, so
    it is exactly as tall as the editors around it and its text starts where
    theirs does. `kind` colors it (see `text_style`); it can still be copied.
    """
    line = QLineEdit(text)
    line.setObjectName("shown")
    line.setReadOnly(True)
    line.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    if kind is not None:
        line.setStyleSheet(text_style(kind))

    return line


class TimingForm(QGroupBox):
    """
    The trial layout, and what it means for each program on the rig.

    Five sections, by where a number goes: what the olfactometer program is
    set to (the only timing it knows), what the imaging wants after the odor,
    what ScanImage is told, what Micromanager is told, and figures that are
    only good to know. `values()` gives the `block_*` annotation values.
    """

    changed = Signal()

    # Everything in here has its natural size, so the box does not grow
    WIDTH_PX = 300

    # A gap this short between the end of imaging and the next trial is
    # worth a warning
    TIGHT_GAP_S = 1.0

    # The olfactometer program's timing, then its counts
    OLFACTOMETER = (
        ("block_baseline_s", "Pre-odor baseline (s)", 5.0),
        ("block_odor_s", "Odor presentation (s)", 4.0),
        ("block_post_odor_delay_s", "Post-odor delay (s)", 5.0),
        ("block_iti_s", "Inter-trial interval (s)", 16.0),
    )
    WANTS = (("block_post_odor_acq_s", "Post-odor acq time (s)", 11.0),)
    INPUTS = OLFACTOMETER + WANTS

    SCANIMAGE = (
        ("frames", "Frames per acquisition"),
        ("acquisitions", "Acquisition count"),
    )
    MICROMANAGER = (
        ("camera_rate", "Frame rate"),
        ("camera_frames", "Camera frame count"),
    )
    FYIS = (
        ("acquisition_s", "Acquisition duration"),
        ("trial_s", "Trial duration"),
        ("trials_per_odor", "Trials per odor"),
        ("program_min", "Program duration"),
        ("loop_min", "Loop duration"),
    )

    def __init__(self, parent=None):
        super().__init__("Quick maths", parent)
        form = QFormLayout(self)
        form.setVerticalSpacing(ROW_SPACING_PX)

        self.inputs: dict[str, QDoubleSpinBox] = {}
        self.outputs: dict[str, QLineEdit] = {}

        def spin_row(key: str, label: str, default: float) -> None:
            box = _spin(0, 3600, default, 2)
            box.valueChanged.connect(lambda _: self.recompute())
            self.inputs[key] = box
            form.addRow(label, box)

        # A box with arrows stands taller than a line: shown values match it
        row_px = _spin(0, 1, 0).sizeHint().height()

        def output_rows(items) -> None:
            for key, label in items:
                self.outputs[key] = shown_value()
                self.outputs[key].setMinimumHeight(row_px)
                form.addRow(label, self.outputs[key])

        _heading(form, "Olfactometer")

        for key, label, default in self.OLFACTOMETER:
            spin_row(key, label, default)

        self.n_odors = QSpinBox()
        self.n_odors.setRange(1, 999)
        self.n_odors.setValue(16)
        self.n_trials = QSpinBox()
        self.n_trials.setRange(1, 99999)
        self.n_trials.setValue(112)
        self.n_programs = QSpinBox()
        self.n_programs.setRange(1, 20)
        self.n_programs.setValue(2)

        for box in (self.n_odors, self.n_trials, self.n_programs):
            box.valueChanged.connect(lambda _: self.recompute())

        form.addRow("# odors", self.n_odors)
        form.addRow("# trials per program", self.n_trials)
        form.addRow("# programs", self.n_programs)

        _heading(form, "Wants")

        for key, label, default in self.WANTS:
            spin_row(key, label, default)

        # Shown when the imaging wants more than the program leaves it
        self.overrun = QLabel("")
        self.overrun.setStyleSheet(text_style("error"))
        self.overrun.setWordWrap(True)
        self.overrun.setVisible(False)
        form.addRow(self.overrun)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        _heading(form, "ScanImage")
        self.frame_rate = _spin(0.01, 1000, 12.86, 3)
        self.frame_rate.valueChanged.connect(lambda _: self.recompute())
        form.addRow("Frame rate (Hz)", self.frame_rate)
        output_rows(self.SCANIMAGE)

        # The camera runs at the imaging rate; shown again so it is typed right
        _heading(form, "Micromanager")
        output_rows(self.MICROMANAGER)

        _heading(form, "FYIs")
        output_rows(self.FYIS)

        self.setFixedWidth(self.WIDTH_PX)
        self.recompute()

    def recompute(self) -> None:
        try:
            timing = block_timing(
                baseline_s=self.inputs["block_baseline_s"].value(),
                odor_s=self.inputs["block_odor_s"].value(),
                post_odor_delay_s=self.inputs["block_post_odor_delay_s"].value(),
                iti_s=self.inputs["block_iti_s"].value(),
                post_odor_acq_s=self.inputs["block_post_odor_acq_s"].value(),
                n_odors=self.n_odors.value(),
                n_trials=self.n_trials.value(),
                frame_rate_hz=self.frame_rate.value(),
                n_programs=self.n_programs.value(),
            )

        except ValueError as error:
            for line in self.outputs.values():
                line.setText("")

            # Read from its start, and whole when hovered if it does not fit
            self.outputs["frames"].setText(str(error))
            self.outputs["frames"].setCursorPosition(0)
            self.outputs["frames"].setToolTip(str(error))
            return

        self.outputs["frames"].setToolTip("")

        shown = {
            "frames": f"{timing.frames}",
            "acquisitions": f"{timing.acquisitions}",
            "acquisition_s": f"{timing.acquisition_s:g} s",
            "trial_s": f"{timing.trial_s:g} s",
            "trials_per_odor": f"{timing.trials_per_odor:g}",
            "program_min": f"{timing.program_min:.0f} min",
            "loop_min": f"{timing.loop_min:.0f} min",
            "camera_rate": f"{self.frame_rate.value():g} Hz",
            "camera_frames": f"{camera_frames(timing.loop_min, self.frame_rate.value())}",
        }

        for key, text in shown.items():
            self.outputs[key].setText(text)

        # Red when the imaging runs past the trial, yellow when it only just fits
        if timing.overrun_s > 0:
            self.overrun.setStyleSheet(text_style("error"))
            self.overrun.setText(
                f"Delay + Interval is {timing.overrun_s:g} s too short.\n"
                "Change the olfactometer program."
            )
        elif timing.gap_s <= self.TIGHT_GAP_S:
            self.overrun.setStyleSheet(text_style("warning"))
            self.overrun.setText(f"Only {timing.gap_s:g} s between acquisitions.")

        self.overrun.setVisible(
            timing.overrun_s > 0 or timing.gap_s <= self.TIGHT_GAP_S
        )

        if self.overrun.isVisibleTo(self):
            # A wrapped label does not ask the form for more height on its own
            width = self.WIDTH_PX - 24
            self.overrun.setFixedHeight(self.overrun.heightForWidth(width))

        self.changed.emit()

    def values(self) -> dict[str, Any]:
        """The `block_*` annotation values for the loop."""
        return {
            **{key: box.value() for key, box in self.inputs.items()},
            "block_n_odors": self.n_odors.value(),
            "block_n_trials": self.n_trials.value(),
        }

    def set_values(self, values: dict[str, Any]) -> None:
        for key, box in self.inputs.items():
            if values.get(key) is not None:
                box.setValue(float(values[key]))

        if values.get("block_n_odors") is not None:
            self.n_odors.setValue(int(values["block_n_odors"]))

        if values.get("block_n_trials") is not None:
            self.n_trials.setValue(int(values["block_n_trials"]))


class PanelPicker(QWidget):
    """
    The odor panel for the day: pick a panel, and its latest vials are listed
    with a box per vial, checked unless that vial was left out of the
    olfactometer.

    `set_panels(panels, vials)` takes `(panel_id, name)` pairs and one row per
    vial with `panel_id`, `vial_position`, `odor_id`, `odor_name`, `made_on`,
    indexed by `vial_id` (the latest per position, from `current_vials`).
    """

    changed = Signal()

    COLUMNS = ("installed", "vial position", "odor id", "odor name", "made on")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vials = None
        self._rows: list[int] = []  # vial_id per table row

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.addWidget(QLabel("Panel"))
        self.panel = QComboBox()
        self.panel.currentIndexChanged.connect(self._panel_picked)
        top.addWidget(self.panel, 1)
        layout.addLayout(top)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._box_toggled)
        # The page scrolls, the table does not: it grows to show every row
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.table)
        self._fit_height()

    def set_panels(self, panels: list[tuple[int, str]], vials) -> None:
        """The choices; keeps the pick and the boxes when they still apply."""
        self.vials = vials
        chosen = self.panel_id()
        checked = self.checked()

        self.panel.blockSignals(True)
        self.panel.clear()
        self.panel.addItem("", None)

        for panel_id, name in panels:
            self.panel.addItem(name, panel_id)

        self.panel.blockSignals(False)
        self.show_vials(checked if chosen is not None else [], chosen)

    def panel_id(self) -> None | int:
        data = self.panel.currentData()

        return None if data is None else int(data)

    def show_vials(self, checked: list[int], panel_id: None | int = None) -> None:
        """Show the panel the checked vials belong to (or `panel_id`), boxes set."""
        if panel_id is None and checked and self.vials is not None:
            known = [v for v in checked if v in self.vials.index]

            if known:
                panel_id = int(self.vials.loc[known[0], "panel_id"])

        self.panel.blockSignals(True)
        self.panel.setCurrentIndex(max(self.panel.findData(panel_id), 0))
        self.panel.blockSignals(False)
        self._fill(set(checked), everything=not checked)

    def _panel_picked(self, _index: int) -> None:
        self._fill(set(), everything=True)
        self.changed.emit()

    def _fill(self, checked: set[int], everything: bool) -> None:
        """One row per vial of the picked panel; `everything` checks them all."""
        panel_id = self.panel_id()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._rows = []

        if panel_id is not None and self.vials is not None:
            mine = self.vials[self.vials["panel_id"] == panel_id].sort_values(
                "vial_position"
            )

            for vial_id, row in mine.iterrows():
                n = self.table.rowCount()
                self.table.insertRow(n)
                box = QTableWidgetItem()
                box.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                )
                box.setCheckState(
                    Qt.CheckState.Checked
                    if everything or vial_id in checked
                    else Qt.CheckState.Unchecked
                )
                self.table.setItem(n, 0, box)

                for column, text in enumerate(
                    (
                        str(int(row["vial_position"])),
                        str(int(row["odor_id"])),
                        str(row["odor_name"]),
                        "" if pd.isna(row["made_on"]) else str(row["made_on"])[:10],
                    ),
                    start=1,
                ):
                    self.table.setItem(n, column, QTableWidgetItem(text))

                self._rows.append(int(vial_id))

        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
        self._fit_height()

    def _fit_height(self) -> None:
        """Exactly tall enough for the header and every row."""
        height = self.table.horizontalHeader().height() + 2 * self.table.frameWidth()

        for row in range(self.table.rowCount()):
            height += self.table.rowHeight(row)

        if self.table.horizontalScrollBar().isVisible():
            height += self.table.horizontalScrollBar().height()

        self.table.setFixedHeight(max(height, 40))

    def _box_toggled(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.changed.emit()

    def checked(self) -> list[int]:
        """The vials installed: those with their box checked."""
        return [
            vial_id
            for n, vial_id in enumerate(self._rows)
            if self.table.item(n, 0).checkState() == Qt.CheckState.Checked
        ]
