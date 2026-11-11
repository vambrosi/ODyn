"""
The window the ROIs of a group are curated in, opened from the QC page for
an approved experiment: the Cellpose candidates of the group
(`odyn.segmentation`) over its mean image, as cells to keep, resize, add and
reject (`Curation`).

The cells start as Cellpose keeps them at its defaults. A click on a cell
selects it, and it stays selected, however small, rejected or outside the
area range, until a click on it again or Esc. A click anywhere else adds the
candidate there, or grows the cell of that basin until it takes the point
clicked (`Curation.add`). The panel at the right has:
- **All cells:** the threshold of every cell without a size of its own (to
  the right, lower: every cell larger), the flow check, the area range (um2:
  cells outside it are excluded), and whether excluded and rejected cells
  show;
- **Selected cell:** its size in pixels and um2, on a logarithmic slider (`[`
  and `]`: one pixel), Follow threshold (T) to give it the threshold of all
  again, and Reject or Restore (Delete);
- **Image:** brightness and gamma of the mean image, and the outlines (O).

Ctrl+Z undoes and Ctrl+Shift+Z redoes; the wheel zooms at the pointer, a drag
pans and 0 fits. The outlines run along the edges of the pixels, in the
colors of the survey viewer: the cells of Cellpose cyan, resized ones pink,
added ones green, rejected ones red and excluded ones gray (both dashed), the
selected one white.

The edits are kept on this computer as they are made (a draft, `draft`) and
picked up when the window opens again on the same candidates; nothing goes
to the share from here. The candidates and the mean image are read from the
share, off the main thread.
"""

from __future__ import annotations

import hashlib
import math

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
    QTransform,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..segmentation import FLOOR, THRESHOLD, Candidates, Curation, Edits
from ..segmentation import read_candidates
from ..shared import read_json, write_json
from . import style

# The threshold slider: from this down to the lowest threshold kept, in these
# steps, so that to the right every cell is larger
TOP_THRESHOLD = 3.0
THRESHOLD_STEP = 0.05

# The size slider, logarithmic from one pixel to the whole basin
SIZE_STEPS = 1000

# The mean image on the screen, as the survey viewer shows it: linear between
# these percentiles, then raised to the gamma and times the brightness, each
# from its slider (low, high, at first; in steps of `LEVEL_STEP`)
LOW_PERCENT, HIGH_PERCENT = 0.1, 99.95
BRIGHTNESS = (0.5, 4.0, 1.0)
GAMMA = (0.2, 1.5, 0.5)
LEVEL_STEP = 0.05

ZOOM_STEP = 1.25
MAX_ZOOM = 32.0
DRAG_PX = 4  # a press that moves less than this is a click
CROSS_PX = 8  # the seed of a selected cell with no pixels left

# The outlines, on the dark image in either theme (the colors of the survey
# viewer)
OUTLINE_PX = 1.4
COLORS = {
    "cellpose": "#3fd8ec",
    "resized": "#ff72dc",
    "added": "#7dff6b",
    "rejected": "#ff5b61",
    "excluded": "#8e9a95",
    "selected": "#ffffff",
}

DRAFT_VERSION = 1
DRAFT_DELAY_MS = 500  # a draft is written once the edits pause this long


def draft_name(group_id: int, user: str) -> str:
    """The file a draft of `user` for group `group_id` is kept in."""
    return f"rois_group_{int(group_id)}_{user}.draft.json"


def fingerprint(found: Candidates) -> str:
    """What tells candidate sets apart: the brightest pixel of every basin."""
    peaks = np.ascontiguousarray(found.peak, np.int64)

    return hashlib.sha1(peaks.tobytes()).hexdigest()


def read_draft(path: Path, found: Candidates) -> None | Edits:
    """The edits in the draft at `path`, if it is one for `found`."""
    try:
        content = read_json(path)

        if content.get("version") != DRAFT_VERSION:
            return None

        if content.get("candidates") != fingerprint(found):
            return None

        return Edits.from_json(content["edits"])

    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_draft(path: Path, found: Candidates, edits: Edits) -> None:
    """Keep `edits` on `found` at `path` (`read_draft` reads them back)."""
    write_json(
        path,
        {
            "version": DRAFT_VERSION,
            "candidates": fingerprint(found),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "edits": edits.as_json(),
        },
    )


def outline(pixels: np.ndarray, width: int) -> tuple[QPainterPath, QRectF]:
    """
    The outline of the pixels `pixels` (indices into an image `width` wide)
    along the edges of the pixels, as lines in image units (pixel `(y, x)`
    spans `x` to `x + 1` and `y` to `y + 1`), and the box around it.
    """
    ys, xs = np.divmod(np.asarray(pixels, np.int64), width)
    top, left = int(ys.min()), int(xs.min())

    # One pixel of room all round, so every edge is between two cells of it
    inside = np.zeros((int(ys.max()) - top + 3, int(xs.max()) - left + 3), bool)
    inside[ys - top + 1, xs - left + 1] = True
    path = QPainterPath()

    # Edges between a row and the next (horizontal), joined into runs
    across = inside[1:] != inside[:-1]

    for row, start, end in _runs(across):
        y = top + row
        path.moveTo(left + start - 1, y)
        path.lineTo(left + end - 1, y)

    # Edges between a column and the next (vertical)
    down = (inside[:, 1:] != inside[:, :-1]).T

    for column, start, end in _runs(down):
        x = left + column
        path.moveTo(x, top + start - 1)
        path.lineTo(x, top + end - 1)

    box = QRectF(left, top, inside.shape[1] - 2, inside.shape[0] - 2)

    return path, box


def _runs(lines: np.ndarray) -> list[tuple[int, int, int]]:
    """The runs of `True` along each row of `lines`: `(row, start, end)`."""
    padded = np.zeros((lines.shape[0], lines.shape[1] + 2), np.int8)
    padded[:, 1:-1] = lines
    steps = np.diff(padded, axis=1)
    starts = np.argwhere(steps == 1)
    ends = np.argwhere(steps == -1)

    return [(int(r), int(s), int(e)) for (r, s), (_, e) in zip(starts, ends)]


def levels16(image: np.ndarray) -> np.ndarray:
    """
    `image` on 16 bits, between its `LOW_PERCENT` and `HIGH_PERCENT`
    percentiles (what the screen shows of it), for `gray8`.
    """
    black, white = np.percentile(image, [LOW_PERCENT, HIGH_PERCENT])
    span = max(float(white) - float(black), 1e-12)
    scaled = np.clip((image - black) / span, 0.0, 1.0)

    return (scaled * 65535 + 0.5).astype(np.uint16)


def gray8(levels: np.ndarray, gamma: float, brightness: float) -> np.ndarray:
    """The 16-bit `levels` shown: raised to `gamma`, times `brightness`."""
    ramp = np.linspace(0.0, 1.0, 65536) ** gamma * brightness
    table = np.clip(ramp * 255 + 0.5, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(table[levels])


def _log_value(value: float, low: float, high: float, steps: int) -> int:
    """Where `value` sits on a logarithmic slider from `low` to `high`."""
    share = math.log(value / low) / math.log(high / low)

    return round(min(max(share, 0.0), 1.0) * steps)


def _log_number(position: int, low: float, high: float, steps: int) -> float:
    """The value at `position` of a logarithmic slider from `low` to `high`."""
    return low * (high / low) ** (position / steps)


def size_position(size: int, total: int) -> int:
    """Where `size` px sits on the size slider of a basin of `total` px."""
    if total <= 1 or size <= 1:
        return 0

    return _log_value(min(size, total), 1, total, SIZE_STEPS)


def size_at(position: int, total: int) -> int:
    """The size at `position` of the size slider of a basin of `total` px."""
    if total <= 1:
        return 1

    return min(total, max(1, round(_log_number(position, 1, total, SIZE_STEPS))))


def threshold_position(threshold: float) -> int:
    """Where `threshold` sits on the threshold slider."""
    return round((TOP_THRESHOLD - threshold) / THRESHOLD_STEP)


def threshold_at(position: int) -> float:
    """The threshold at `position` of the threshold slider (never `-0.0`)."""
    return round(TOP_THRESHOLD - position * THRESHOLD_STEP, 6) + 0.0


def load(mean: Path, candidates: Path) -> dict:
    """
    The mean image of a group and its candidates, ready to curate. Raises
    `FileNotFoundError` naming the one missing.
    """
    for path in (candidates, mean):
        if not Path(path).is_file():
            raise FileNotFoundError(str(path))

    image = np.asarray(tifffile.imread(mean), np.float32)
    found = read_candidates(candidates)

    if image.shape != found.basins.shape:
        raise ValueError(
            f"The mean image is {image.shape} but the candidates are "
            f"{found.basins.shape}: made from another image."
        )

    return {"levels": levels16(image), "found": found, "curation": Curation(found)}


@contextmanager
def _quiet(widget: QObject):
    """`widget` set without its signals firing."""
    blocked = widget.blockSignals(True)

    try:
        yield widget
    finally:
        widget.blockSignals(blocked)


class _Loader(QThread):
    """Runs `load` on its own thread; `loaded` gets a dict or a problem."""

    loaded = Signal(object)

    def __init__(self, mean: Path, candidates: Path, parent=None):
        super().__init__(parent)
        self.mean, self.candidates = mean, candidates

    def run(self) -> None:  # noqa: D102 (Qt's)
        try:
            self.loaded.emit(load(self.mean, self.candidates))
        except FileNotFoundError as error:
            self.loaded.emit(("missing", str(error)))
        except Exception as error:
            self.loaded.emit(("failed", f"{type(error).__name__}: {error}"))


class CellCanvas(QWidget):
    """
    The mean image and the outlines on it. The wheel zooms at the pointer
    and a drag pans; `clicked(row, column)` gives the pixel clicked.
    `shapes` are what is drawn over the image: `(outline, box, kind)`, in
    image units, with `kind` one of `COLORS`, `-dashed` added for a dashed
    line; `cross` the pixel the selected cell grows from, when it has no
    pixels; `message` shows while there is no image.
    """

    clicked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: None | QImage = None
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        self.shapes: list[tuple[QPainterPath, QRectF, str]] = []
        self.cross: None | tuple[int, int] = None
        self.outlines = True
        self.message = ""
        self._press: None | QPointF = None
        self._dragged = False
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def set_image(self, gray: np.ndarray) -> None:
        height, width = gray.shape
        self.image = QImage(
            gray.data, width, height, width, QImage.Format.Format_Grayscale8
        ).copy()  # the array goes away
        self.update()

    def fit(self) -> None:
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        self.update()

    def source_rect(self) -> QRectF:
        """
        The part of the image shown, at the zoom and center. Zoomed in, it
        widens (or heightens) to the shape of the canvas, as far as the image
        goes, so the canvas is filled.
        """
        if self.image is None:
            return QRectF()

        width, height = self.image.width(), self.image.height()
        w, h = width / self.zoom, height / self.zoom

        if self.zoom > 1 and self.width() > 0 and self.height() > 0:
            shape = self.width() / self.height()

            if w / h < shape:
                w = min(width, h * shape)
            else:
                h = min(height, w / shape)

        x = min(max(self.center.x() * width - w / 2, 0), width - w)
        y = min(max(self.center.y() * height - h / 2, 0), height - h)

        return QRectF(x, y, w, h)

    def target_rect(self) -> QRectF:
        """Where the image goes in the canvas: as large as fits, in shape."""
        inner, source = QRectF(self.rect()), self.source_rect()

        if source.isEmpty() or inner.isEmpty():
            return inner

        scale = min(inner.width() / source.width(), inner.height() / source.height())
        w, h = source.width() * scale, source.height() * scale

        return QRectF(
            inner.x() + (inner.width() - w) / 2,
            inner.y() + (inner.height() - h) / 2,
            w,
            h,
        )

    def image_point(self, position: QPointF) -> None | QPointF:
        """Where a canvas point is on the image (in pixels), if on it."""
        target, source = self.target_rect(), self.source_rect()

        if self.image is None or not target.contains(position):
            return None

        return QPointF(
            source.x() + (position.x() - target.x()) / target.width() * source.width(),
            source.y()
            + (position.y() - target.y()) / target.height() * source.height(),
        )

    def canvas_point(self, row: float, column: float) -> QPointF:
        """Where a point of the image (in pixels) is on the canvas."""
        target, source = self.target_rect(), self.source_rect()

        return QPointF(
            target.x() + (column - source.x()) / source.width() * target.width(),
            target.y() + (row - source.y()) / source.height() * target.height(),
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self.image is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(
                QRectF(self.rect()).adjusted(24, 24, -24, -24),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.message,
            )
            painter.end()
            return

        target, source = self.target_rect(), self.source_rect()
        scale = target.width() / source.width()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 1)
        painter.drawImage(target, self.image, source)

        if self.outlines:
            painter.save()
            painter.setClipRect(target)
            painter.setTransform(
                QTransform(
                    scale,
                    0,
                    0,
                    target.height() / source.height(),
                    target.x() - source.x() * scale,
                    target.y() - source.y() * target.height() / source.height(),
                )
            )

            pens: dict[str, QPen] = {}

            for path, box, kind in self.shapes:
                if box.intersects(source):
                    if kind not in pens:
                        pens[kind] = self._pen(kind, scale)

                    painter.setPen(pens[kind])
                    painter.drawPath(path)

            painter.restore()

            if self.cross is not None:
                at = self.canvas_point(self.cross[0] + 0.5, self.cross[1] + 0.5)
                painter.setPen(QPen(QColor(COLORS["selected"]), OUTLINE_PX * 1.8))
                painter.drawLine(at - QPointF(CROSS_PX, 0), at + QPointF(CROSS_PX, 0))
                painter.drawLine(at - QPointF(0, CROSS_PX), at + QPointF(0, CROSS_PX))

        painter.end()

    @staticmethod
    def _pen(kind: str, scale: float) -> QPen:
        """
        The pen of the outlines of `kind`, in image units at `scale` (screen
        points per pixel): as wide at every zoom and on every screen (a
        cosmetic pen would be thinner on a high-density one).
        """
        name, _, dashed = kind.partition("-")
        width = OUTLINE_PX * (1.8 if name == "selected" else 1.0) / scale
        pen = QPen(QColor(COLORS[name]), width)

        if dashed:
            pen.setDashPattern([3, 2])  # in widths

        return pen

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        at = self.image_point(event.position())

        if at is None or not event.angleDelta().y():
            return

        step = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        old, new = self.zoom, min(max(self.zoom * step, 1.0), MAX_ZOOM)
        point = QPointF(at.x() / self.image.width(), at.y() / self.image.height())

        # The point under the pointer stays under it
        c = self.center
        self.center = QPointF(
            point.x() + (c.x() - point.x()) * old / new,
            point.y() + (c.y() - point.y()) * old / new,
        )
        self.zoom = new
        self._clamp()
        event.accept()

    def _clamp(self) -> None:
        half = 0.5 / self.zoom
        c = self.center
        self.center = QPointF(
            min(max(c.x(), half), 1 - half), min(max(c.y(), half), 1 - half)
        )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
            self._dragged = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self._press is None or self.image is None:
            return

        moved = event.position() - self._press

        if abs(moved.x()) + abs(moved.y()) < DRAG_PX and not self._dragged:
            return

        self._dragged = True
        self._press = event.position()
        target, source = self.target_rect(), self.source_rect()

        if target.width() and target.height():
            self.center = self.center - QPointF(
                moved.x() / target.width() * source.width() / self.image.width(),
                moved.y() / target.height() * source.height() / self.image.height(),
            )
            self._clamp()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self._press is not None and not self._dragged:
            at = self.image_point(event.position())

            if at is not None:
                row = min(int(at.y()), self.image.height() - 1)
                column = min(int(at.x()), self.image.width() - 1)
                self.clicked.emit(row, column)

        self._press = None
        self._dragged = False


class CurationWindow(QWidget):
    """
    The ROIs of group `group_id` (see the module description): the
    candidates in the file `candidates`, over the mean image in the file
    `mean`, with the edits kept in the file `draft` on this computer.
    `closed` fires as the window closes.
    """

    closed = Signal()

    def __init__(
        self,
        title: str,
        group_id: int,
        mean: Path,
        candidates: Path,
        draft: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"ROIs: {title}")
        self.setStyleSheet(style.window_stylesheet())  # as the main window
        self.group_id = int(group_id)
        self.mean, self.candidates = Path(mean), Path(candidates)
        self.draft = Path(draft)
        self.found: None | Candidates = None
        self.curation: None | Curation = None
        self.levels: None | np.ndarray = None  # the mean image (`levels16`)
        self.selected: None | int = None  # a basin
        self.undo_stack: list[Edits] = []
        self.redo_stack: list[Edits] = []
        self._outlines: dict[int, tuple[tuple, QPainterPath, QRectF]] = {}
        self._pending_threshold: None | float = None

        self.canvas = CellCanvas()
        self.canvas.message = "Reading the Cellpose candidates and the mean image..."
        self.canvas.clicked.connect(self.click)
        self.line = QLabel("")  # what the last click did, or what went wrong
        self.line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.addWidget(self.canvas, stretch=1)
        middle.addWidget(self._panel())
        layout = QVBoxLayout(self)
        layout.addLayout(middle, stretch=1)
        layout.addWidget(self.line)

        # The draft is written once the edits pause
        self.saving = QTimer(self)
        self.saving.setSingleShot(True)
        self.saving.setInterval(DRAFT_DELAY_MS)
        self.saving.timeout.connect(self.save_draft)

        # A threshold drag applies the latest position, not every one passed
        self.thresholding = QTimer(self)
        self.thresholding.setSingleShot(True)
        self.thresholding.setInterval(0)
        self.thresholding.timeout.connect(self._apply_threshold)

        self._shortcuts()
        self._enable()
        self.resize(1300, 850)

        self.loader = _Loader(self.mean, self.candidates, self)
        self.loader.loaded.connect(self._loaded)
        self.loader.start()

    # ----------------------------------------------------------------- #
    # Building
    # ----------------------------------------------------------------- #

    def _panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        column = QVBoxLayout(panel)
        column.setContentsMargins(8, 0, 4, 0)

        column.addWidget(self._heading("All cells"))
        self.threshold_line = QLabel("")
        column.addWidget(self.threshold_line)
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(0, threshold_position(FLOOR))
        self.threshold.setValue(threshold_position(THRESHOLD))
        self.threshold.setToolTip(
            "The threshold of every cell without a size of its own: to the "
            "right, lower, and every cell larger"
        )
        self.threshold.sliderPressed.connect(self.remember)
        self.threshold.valueChanged.connect(self._threshold_moved)
        column.addWidget(self.threshold)
        ends = QHBoxLayout()
        ends.addWidget(self._small("smaller"))
        ends.addStretch(1)
        ends.addWidget(self._small("larger"))
        column.addLayout(ends)

        self.flow_check = QCheckBox("Flow check")
        self.flow_check.setChecked(True)
        self.flow_check.setToolTip(
            "Only the candidates whose shape at threshold 0 agrees with the "
            "flows (the Cellpose flow check); off, every candidate with a cell at "
            "the "
            "threshold"
        )
        self.flow_check.toggled.connect(self._flow_check_toggled)
        column.addWidget(self.flow_check)

        area = QHBoxLayout()
        area.addWidget(QLabel("Area"))
        self.min_area = self._area_box("")
        self.max_area = self._area_box("no max")
        area.addWidget(self.min_area, stretch=1)
        area.addWidget(QLabel("to"))
        area.addWidget(self.max_area, stretch=1)
        column.addLayout(area)
        self.show_excluded = QCheckBox("Show excluded")
        self.show_excluded.setToolTip("The cells outside the area range, dashed gray")
        self.show_rejected = QCheckBox("Show rejected")
        self.show_rejected.setToolTip("The cells rejected, dashed red")

        for box in (self.show_excluded, self.show_rejected):
            box.toggled.connect(lambda _: self.redraw())
            column.addWidget(box)

        self.counts = QLabel("")
        self.counts.setWordWrap(True)
        self.counts.setStyleSheet(style.text_style("computed"))
        column.addWidget(self.counts)

        column.addWidget(self._heading("Selected cell"))
        self.cell_line = QLabel("")
        self.cell_line.setWordWrap(True)
        column.addWidget(self.cell_line)
        size = QHBoxLayout()
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(0, SIZE_STEPS)
        self.size.setToolTip("Its size, from one pixel to its whole basin ([ and ])")
        self.size.sliderPressed.connect(self.remember)
        self.size.valueChanged.connect(self._size_moved)
        size.addWidget(self.size, stretch=1)
        self.size_line = QLabel("")
        size.addWidget(self.size_line)
        column.addLayout(size)
        buttons = QHBoxLayout()
        self.follow = QPushButton("Follow threshold")
        self.follow.setToolTip(self._tip("Give it the threshold of all again", "T"))
        self.follow.clicked.connect(self.follow_threshold)
        self.reject = QPushButton("Reject")
        self.reject.setToolTip(self._tip("Reject it, or restore it", "Delete"))
        self.reject.clicked.connect(self.toggle_rejection)
        buttons.addWidget(self.follow)
        buttons.addWidget(self.reject)
        column.addLayout(buttons)

        column.addWidget(self._heading("Image"))
        self.brightness = self._level_slider(BRIGHTNESS, "Brightness")
        self.gamma = self._level_slider(GAMMA, "Gamma (lower: dim parts brighter)")

        for slider, name in ((self.brightness, "Brightness"), (self.gamma, "Gamma")):
            row = QHBoxLayout()
            label = QLabel(name)
            label.setMinimumWidth(72)
            row.addWidget(label)
            row.addWidget(slider, stretch=1)
            column.addLayout(row)

        self.outlines = QCheckBox("Outlines")
        self.outlines.setChecked(True)
        self.outlines.setToolTip(self._tip("Show the outlines", "O"))
        self.outlines.toggled.connect(self._outlines_toggled)
        column.addWidget(self.outlines)

        column.addStretch(1)
        history = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip(self._tip("Undo", "Ctrl+Z"))
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QPushButton("Redo")
        self.redo_button.setToolTip(self._tip("Redo", "Ctrl+Shift+Z"))
        self.redo_button.clicked.connect(self.redo)
        fit = QPushButton("Fit")
        fit.setToolTip(self._tip("Show the whole image", "0"))
        fit.clicked.connect(self.canvas.fit)
        history.addWidget(self.undo_button)
        history.addWidget(self.redo_button)
        history.addWidget(fit)
        column.addLayout(history)

        return panel

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; margin-top: 6px;")

        return label

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(style.text_style("computed"))

        return label

    @staticmethod
    def _tip(title: str, key: str) -> str:
        """A tooltip naming its shortcut, as this system writes it."""
        native = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)

        return f"{title} ({native})"

    def _area_box(self, empty: str) -> QDoubleSpinBox:
        """A bound of the area range, in um2; `empty` is how 0 shows."""
        box = QDoubleSpinBox()
        box.setRange(0.0, 100_000.0)
        box.setDecimals(0)
        box.setSingleStep(5.0)
        box.setSuffix(" µm²")
        box.setKeyboardTracking(False)

        if empty:
            box.setSpecialValueText(empty)

        box.editingFinished.connect(self._area_changed)

        return box

    def _level_slider(self, bounds: tuple[float, float, float], tip: str) -> QSlider:
        """A slider from `bounds[0]` to `bounds[1]`, at `bounds[2]` first."""
        low, high, start = bounds
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(round(low / LEVEL_STEP), round(high / LEVEL_STEP))
        slider.setValue(round(start / LEVEL_STEP))
        slider.setToolTip(tip)
        slider.valueChanged.connect(lambda _: self._show_image())

        return slider

    def _shortcuts(self) -> None:
        for key, do in (
            ("Esc", lambda: self.select(None)),
            ("Delete", self.toggle_rejection),
            ("Backspace", self.toggle_rejection),
            ("[", lambda: self.nudge(-1)),
            ("]", lambda: self.nudge(1)),
            ("T", self.follow_threshold),
            ("O", self.outlines.toggle),
            ("0", self.canvas.fit),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
        ):
            QShortcut(QKeySequence(key), self).activated.connect(do)

    # ----------------------------------------------------------------- #
    # Loading
    # ----------------------------------------------------------------- #

    def _loaded(self, result) -> None:
        if isinstance(result, tuple):
            kind, text = result
            self.canvas.message = self._problem(kind, text)
            self.canvas.update()
            return

        self.levels = result["levels"]
        self.found, self.curation = result["found"], result["curation"]
        self._show_image()

        edits = read_draft(self.draft, self.found)

        if edits is not None:
            self.curation.edits = edits
            self._show_edits()
            self.say("Picked up your draft from where you left it.")
        elif self.draft.is_file():
            self.say(
                "Your draft was for other candidates (made again since): "
                "starting from the cells Cellpose keeps."
            )

        self.redraw()

    def _problem(self, kind: str, text: str) -> str:
        if kind == "missing" and Path(text) == self.candidates:
            return (
                "No Cellpose candidates for this group yet. The cluster makes "
                "them after an approval; by hand: python -m odyn.cluster "
                f"<main folder> cellpose --group {self.group_id}"
            )

        if kind == "missing":
            return (
                "No mean image for this group yet. The cluster makes it after "
                "an approval; by hand: python -m odyn.cluster <main folder> "
                f"mean-images --group {self.group_id}"
            )

        return f"Could not read the candidates: {text}"

    def _show_image(self) -> None:
        if self.levels is not None:
            gamma = self.gamma.value() * LEVEL_STEP
            brightness = self.brightness.value() * LEVEL_STEP
            self.canvas.set_image(gray8(self.levels, gamma, brightness))

    # ----------------------------------------------------------------- #
    # Changing
    # ----------------------------------------------------------------- #

    def remember(self) -> None:
        """Keep the edits as they are, for Undo (before each change)."""
        if self.curation is not None:
            self.undo_stack.append(self.curation.edits)
            self.redo_stack.clear()

    def changed(self) -> None:
        """After every change: redraw, and keep the draft soon."""
        self.redraw()
        self.saving.start()

    def click(self, row: int, column: int) -> None:
        """
        A click on a cell selects it, or lets it go if it was selected; a
        click elsewhere adds the candidate there or grows the cell of that
        basin to it.
        """
        curation = self.curation

        if curation is None:
            return

        basin = curation.cell_at(row, column)

        if basin is not None:
            self.select(None if basin == self.selected else basin)
            return

        basin = int(self.found.basins[row, column])

        if not basin:
            self.select(None)
            self.say(
                "No Cellpose candidate covers this point (its probability is "
                "below the lowest threshold kept)."
            )
            return

        grows = basin in curation.cells
        self.remember()

        if curation.add(row, column) is None:
            self.undo_stack.pop()
            self.say("That point is in the basin of a cell, but not connected to it.")
            return

        cell = curation.cells.get(basin)

        if grows:
            self.say("Grew that cell until it reaches the point clicked.")
        elif cell is not None:
            self.say(
                "Added the candidate under the point clicked, at threshold "
                f"{cell.threshold:.2f}. The size slider grows or shrinks it."
            )

        self.selected = basin
        self.changed()

    def select(self, basin: None | int) -> None:
        self.selected = basin
        self.say("")
        self.redraw()

    def _threshold_moved(self, position: int) -> None:
        if not self.threshold.isSliderDown():
            self.remember()  # by the keyboard: a step each

        self._pending_threshold = threshold_at(position)
        self._show_threshold()
        self.thresholding.start()

    def _apply_threshold(self) -> None:
        if self.curation is None or self._pending_threshold is None:
            return

        self.curation.set_threshold(self._pending_threshold)
        self._pending_threshold = None
        self.changed()

    def _flow_check_toggled(self, on: bool) -> None:
        if self.curation is not None:
            self.remember()
            self.curation.set_flow_check(on)
            self.changed()

    def _area_changed(self) -> None:
        if self.curation is None:
            return

        low, high = self.min_area.value(), self.max_area.value() or None
        edits = self.curation.edits

        if (low, high) != (edits.min_um2, edits.max_um2):
            self.remember()
            self.curation.set_area_range(low, high)
            self.changed()

        self.canvas.setFocus()  # so the keys work on cells again

    def _size_moved(self, position: int) -> None:
        if self.curation is None or self.selected is None:
            return

        if not self.size.isSliderDown():
            self.remember()

        total = len(self.curation.flood(self.selected)[0])
        self.curation.resize(self.selected, size_at(position, total))
        self.changed()

    def nudge(self, step: int) -> None:
        """One pixel more (or fewer) for the selected cell."""
        if self.curation is None or self.selected is None:
            return

        cell = self.curation.cells.get(self.selected)
        size = 0 if cell is None else len(cell.pixels)
        total = len(self.curation.flood(self.selected)[0])

        if not 1 <= size + step <= total:
            return

        self.remember()
        self.curation.resize(self.selected, size + step)
        self.changed()

    def follow_threshold(self) -> None:
        if self.curation is None or self.selected is None:
            return

        if self.selected in self.curation.edits.sizes:
            self.remember()
            self.curation.follow_threshold(self.selected)
            self.changed()

    def toggle_rejection(self) -> None:
        if self.curation is None or self.selected not in self.curation.cells:
            return

        self.remember()

        if self.curation.cells[self.selected].rejected:
            self.curation.restore(self.selected)
        else:
            self.curation.reject(self.selected)

        self.changed()

    def undo(self) -> None:
        self._step_back(self.undo_stack, self.redo_stack)

    def redo(self) -> None:
        self._step_back(self.redo_stack, self.undo_stack)

    def _step_back(self, source: list[Edits], target: list[Edits]) -> None:
        if self.curation is None or not source:
            return

        target.append(self.curation.edits)
        self.curation.edits = source.pop()
        self._show_edits()
        self.changed()

    def _show_edits(self) -> None:
        """Put the controls of all cells where the edits have them."""
        edits = self.curation.edits

        with _quiet(self.threshold):
            self.threshold.setValue(threshold_position(edits.threshold))

        with _quiet(self.flow_check):
            self.flow_check.setChecked(edits.flow_check)

        with _quiet(self.min_area), _quiet(self.max_area):
            self.min_area.setValue(edits.min_um2)
            self.max_area.setValue(edits.max_um2 or 0.0)

    def _outlines_toggled(self, on: bool) -> None:
        self.canvas.outlines = on
        self.canvas.update()

    def say(self, text: str) -> None:
        self.line.setText(text)

    # ----------------------------------------------------------------- #
    # Showing
    # ----------------------------------------------------------------- #

    def redraw(self) -> None:
        """The outlines, the counts and the selected cell, as the edits are."""
        self._enable()

        if self.curation is None:
            return

        shapes, cross = [], None
        show_rejected = self.show_rejected.isChecked()
        show_excluded = self.show_excluded.isChecked()

        for basin, cell in sorted(self.curation.cells.items()):
            if basin == self.selected:
                continue

            if cell.rejected and not show_rejected:
                continue

            if cell.excluded and not cell.rejected and not show_excluded:
                continue

            if cell.rejected:
                kind = "rejected-dashed"
            elif cell.excluded:
                kind = "excluded-dashed"
            else:
                kind = cell.kind

            shapes.append((*self._outline(cell), kind))

        if self.selected is not None:
            cell = self.curation.cells.get(self.selected)

            if cell is None:
                cross = divmod(self.curation.seed(self.selected), self.curation.width)
            else:
                kind = "selected" if cell.kept else "selected-dashed"
                shapes.append((*self._outline(cell), kind))

        self.canvas.shapes, self.canvas.cross = shapes, cross
        self.canvas.update()
        self._show_threshold()
        self._show_counts()
        self._show_selected()

    def _outline(self, cell) -> tuple[QPainterPath, QRectF]:
        """The outline of `cell`, made again only when its pixels changed."""
        key = (self.curation.seed(cell.basin), len(cell.pixels))
        kept = self._outlines.get(cell.basin)

        if kept is None or kept[0] != key:
            kept = (key, *outline(cell.pixels, self.curation.width))
            self._outlines[cell.basin] = kept

        return kept[1], kept[2]

    def _show_threshold(self) -> None:
        threshold = threshold_at(self.threshold.value())
        default = " (the Cellpose default)" if threshold == THRESHOLD else ""
        self.threshold_line.setText(f"Threshold {threshold:.2f}{default}")

    def _show_counts(self) -> None:
        cells = list(self.curation.cells.values())
        kept = [cell for cell in cells if cell.kept]
        added = sum(cell.kind == "added" for cell in kept)
        resized = sum(cell.kind == "resized" for cell in kept)
        rejected = sum(cell.rejected for cell in cells)
        outside = sum(cell.excluded and not cell.rejected for cell in cells)
        cells = "cell" if len(kept) == 1 else "cells"
        self.counts.setText(
            f"{len(kept)} {cells} kept ({added} added, {resized} resized), "
            f"{rejected} rejected, {outside} outside the area range"
        )

    def _show_selected(self) -> None:
        basin = self.selected

        if basin is None:
            self.cell_line.setText("Click a cell to select it.")
            self.size_line.setText("")

            with _quiet(self.size):
                self.size.setValue(0)

            return

        cell = self.curation.cells.get(basin)
        size = 0 if cell is None else len(cell.pixels)
        total = len(self.curation.flood(basin)[0])

        if not self.size.isSliderDown():  # it would fight the pointer
            with _quiet(self.size):
                self.size.setValue(size_position(size, total))

        area = size * self.curation.um2_per_px
        self.size_line.setText(f"{size} px, {area:.1f} µm²")

        if cell is None:
            self.cell_line.setText(
                f"Cell {basin}: no pixels at this threshold (the cross is where "
                "it grows from)."
            )
            return

        state = (
            ", rejected"
            if cell.rejected
            else ", outside the area range" if cell.excluded else ""
        )
        self.cell_line.setText(
            f"Cell {basin}, {cell.kind}{state}: threshold {cell.threshold:.2f}"
        )
        self.reject.setText("Restore" if cell.rejected else "Reject")

    def _enable(self) -> None:
        ready = self.curation is not None
        picked = ready and self.selected is not None
        has_cell = picked and self.selected in self.curation.cells

        for control in (
            self.threshold,
            self.flow_check,
            self.min_area,
            self.max_area,
            self.brightness,
            self.gamma,
        ):
            control.setEnabled(ready)

        self.size.setEnabled(picked)
        self.reject.setEnabled(has_cell)
        self.follow.setEnabled(picked and self.selected in self.curation.edits.sizes)
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))

    # ----------------------------------------------------------------- #
    # The draft
    # ----------------------------------------------------------------- #

    def save_draft(self) -> None:
        """Keep the edits on this computer (as they change, and on closing)."""
        self.saving.stop()

        if self.curation is None or not (self.undo_stack or self.draft.is_file()):
            return  # nothing done: no draft

        try:
            write_draft(self.draft, self.found, self.curation.edits)
        except OSError as error:
            self.say(f"Could not keep the draft here: {error}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self.thresholding.isActive():
            self.thresholding.stop()
            self._apply_threshold()

        self.save_draft()
        self.loader.wait()
        self.closed.emit()
        super().closeEvent(event)
