"""
The window the ROIs of a group are curated in, opened from the QC page for
an approved experiment: the Cellpose candidates of the group
(`odyn.segmentation`) over its mean image, as cells to keep, resize, add and
reject (`Curation`).

The cells start as Cellpose keeps them at its defaults. A click on a cell
selects it, and it stays selected, however small, rejected or outside the
area range, until a click on it again or Esc. A click anywhere else adds the
candidate there, or grows the cell of that basin until it takes the point
clicked (`Curation.add`).

- **Above the image**, the legend of the outlines, in the colors of the
  survey viewer: the cells of Cellpose cyan, resized ones pink, added ones
  green, rejected ones red and those outside the area range gray (both
  dashed), the selected one white.
- **The bar at the right**, as in the other windows: undo, redo, fit, the
  outlines, the cells outside the area range and the rejected ones, the
  flow check, the brightness and gamma of the image, and the shortcuts
  (`KEYS`, on a card in the middle of the image).
- **Under the image**, for all cells: the threshold of every cell without a
  size of its own (to the right, lower: every cell larger), the area range
  (um2: cells outside it are excluded), and the counts.
- **The card of the selected cell**, over the image, in the corner that
  leaves the cell in view: its number and threshold, what sets it apart
  (`cell_state`), its size in pixels and um2 on a logarithmic slider,
  Follow threshold and Reject or Restore.

The wheel zooms at the pointer and a drag pans. The outlines run along the
edges of the pixels.

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

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
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
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..segmentation import FLOOR, THRESHOLD, Candidates, Curation, Edits
from ..segmentation import read_candidates
from ..shared import read_json, write_json
from . import style
from .symbols import icon

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

# The legend above the image: each kind of outline and what it means
LEGEND = {
    "cellpose": "Cellpose",
    "resized": "Resized",
    "added": "Added",
    "rejected-dashed": "Rejected",
    "excluded-dashed": "Outside the area range",
    "selected": "Selected",
}

# The shortcuts, as the tooltips and the shortcuts card name them (on the
# card in this order, then what the mouse does)
KEYS = {
    "smaller": ("[", "Decrease cell area by 1 px"),
    "larger": ("]", "Increase cell area by 1 px"),
    "follow": ("T", "Follow the global threshold"),
    "reject": ("Backspace, Delete", "Reject or restore the cell"),
    "deselect": ("Esc", "Deselect the cell"),
    "undo": ("Ctrl+Z", "Undo"),
    "redo": ("Ctrl+Shift+Z", "Redo"),
    "fit": ("0", "Show whole image"),
    "outlines": ("O", "Show outlines"),
    "show-excluded": ("E", "Show cells outside the area range"),
    "show-rejected": ("R", "Show rejected cells"),
    "image": ("I", "Brightness and gamma"),
    "help": ("?", "Shortcuts"),
}
MOUSE = (
    ("Click", "Select a cell, or add the one there"),
    ("Drag", "Pan"),
    ("Wheel", "Zoom"),
)

# The area range applies once its typing or its arrows pause this long
AREA_DELAY_MS = 300

# The card of the selected cell, kept this far from the edges of the image
# and from the cell
CARD_WIDTH_PX = 340
CARD_MARGIN_PX = 12

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


def _native(key: str) -> str:
    """A key as this system writes it (on a Mac, Ctrl+Z is ⌘Z)."""
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)


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
    pixels; `message` shows while there is no image. `viewed` fires when the
    part of the image shown changes (a zoom, a pan, a new size).
    """

    clicked = Signal(int, int)
    viewed = Signal()

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
        self.viewed.emit()

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
        self.viewed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        super().resizeEvent(event)
        self.viewed.emit()

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


class _NoWheel(QObject):
    """Swallows the wheel events of what it is installed on."""

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt's name)
        return event.type() == QEvent.Type.Wheel


class _Swatch(QWidget):
    """An outline of the legend, on a chip as dark as the image."""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(22, 14)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRoundedRect(QRectF(self.rect()), 3, 3)
        painter.setPen(CellCanvas._pen(self.kind, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(5, 3.5, -5, -3.5), 3, 3)
        painter.end()


class _Help(QFrame):
    """The shortcuts card: a click on it puts it away."""

    dismissed = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.dismissed.emit()


def cell_state(cell) -> str:
    """
    What sets `cell` apart, for the second line of its card: added or
    resized, rejected, outside the area range (empty for a cell of Cellpose
    that follows the global threshold and is kept).
    """
    parts = [cell.kind] if cell.kind != "cellpose" else []
    parts += ["rejected"] if cell.rejected else []
    parts += ["outside the area range"] if cell.excluded else []

    if len(parts) > 1:
        return f"{', '.join(parts[:-1])} and {parts[-1]}"

    return "".join(parts)


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

        # The draft is written once the edits pause
        self.saving = self._timer(DRAFT_DELAY_MS, self.save_draft)
        # A threshold drag applies the latest position, not every one passed
        self.thresholding = self._timer(0, self._apply_threshold)
        # The area range applies once the typing or the arrows pause
        self.ranging = self._timer(AREA_DELAY_MS, self._area_changed)

        self.no_wheel = _NoWheel(self)
        self.canvas = CellCanvas()
        self.canvas.message = "Reading the Cellpose candidates and the mean image..."
        self.canvas.clicked.connect(self.click)
        self.canvas.viewed.connect(self._place_card)
        self.card = self._card(self.canvas)  # floats over the image
        self.help = self._help_card(self.canvas)  # over the card too
        self.canvas.viewed.connect(self._place_help)
        self.line = QLabel("")  # what the last click did, or what went wrong
        self.line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        view = QVBoxLayout()
        view.setContentsMargins(0, 0, 0, 0)
        view.setSpacing(0)
        view.addWidget(self._legend())
        view.addWidget(self.canvas, stretch=1)
        stage = QHBoxLayout()
        stage.setContentsMargins(0, 0, 0, 0)
        stage.setSpacing(0)
        stage.addLayout(view, stretch=1)
        stage.addWidget(self._toolbar())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 0, 6)
        layout.addLayout(stage, stretch=1)
        layout.addWidget(self._controls())
        layout.addLayout(self._status())

        self._shortcuts()
        self._enable()
        self.resize(1300, 850)

        self.loader = _Loader(self.mean, self.candidates, self)
        self.loader.loaded.connect(self._loaded)
        self.loader.start()

    # ----------------------------------------------------------------- #
    # Building
    # ----------------------------------------------------------------- #

    def _timer(self, delay_ms: int, then) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(delay_ms)
        timer.timeout.connect(then)

        return timer

    def _toolbar(self) -> QWidget:
        """The tools, on a bar at the right, as the other windows have them."""
        bar = QToolBar("ROIs")
        bar.setOrientation(Qt.Orientation.Vertical)
        bar.setMovable(False)
        bar.setIconSize(QSize(style.BAR_ICON_PX, style.BAR_ICON_PX))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(*style.BAR_MARGINS)
        column.addWidget(bar)

        self.undo_action = self._action(bar, "undo", self.undo)
        self.redo_action = self._action(bar, "redo", self.redo)
        bar.addSeparator()
        self._action(bar, "fit", self.canvas.fit)
        self.outlines = self._action(
            bar, "outlines", self._outlines_toggled, checked=True
        )
        self.show_excluded = self._action(
            bar, "show-excluded", lambda _: self.redraw(), checked=False
        )
        self.show_rejected = self._action(
            bar, "show-rejected", lambda _: self.redraw(), checked=False
        )
        bar.addSeparator()
        self.flow_check = self._action(
            bar,
            "flow-check",
            self._flow_check_toggled,
            checked=True,
            title="Only cells that pass the flow check at 0",
        )
        # A menu, which I opens too (`_shortcuts`)
        self.image_action = self._action(bar, "image", None)
        self.image_action.setMenu(self._image_menu())
        self.image_button = bar.widgetForAction(self.image_action)
        self.image_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.image_button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        bar.addSeparator()
        self.help_action = self._action(bar, "help", self._show_help, checked=False)

        return holder

    def _action(
        self,
        bar: QToolBar,
        name: str,
        then,
        checked: None | bool = None,
        title: str = "",
    ) -> QAction:
        """
        A bar button doing `then`, with the icon `name` and its shortcut and
        title in `KEYS` (or `title`, with no shortcut), both in its tooltip;
        a toggle when `checked` says how it starts.
        """
        key, title = KEYS.get(name, ("", title))
        action = QAction(icon(name), title, self)
        action.setToolTip(self._tip(name) if key else title)

        if key and then is not None:
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)

        if checked is not None:
            action.setCheckable(True)
            action.setChecked(checked)
            action.toggled.connect(then)
        elif then is not None:
            action.triggered.connect(lambda _=False: then())

        bar.addAction(action)
        button = bar.widgetForAction(action)

        if isinstance(button, QToolButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        return action

    def _image_menu(self) -> QMenu:
        """Brightness and gamma of the mean image, under the image tool."""
        self.brightness = self._level_slider(BRIGHTNESS, "Brightness")
        self.gamma = self._level_slider(GAMMA, "Gamma (lower: dim parts brighter)")
        panel = QWidget()
        rows = QFormLayout(panel)
        rows.setContentsMargins(12, 8, 12, 8)
        rows.addRow("Brightness", self.brightness)
        rows.addRow("Gamma", self.gamma)
        panel.setMinimumWidth(280)
        menu = QMenu(self)
        holder = QWidgetAction(menu)
        holder.setDefaultWidget(panel)
        menu.addAction(holder)

        return menu

    def _controls(self) -> QWidget:
        """Under the image, for all cells: the threshold and the area range."""
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 6, 8, 0)

        row.addWidget(QLabel("Threshold"))
        row.addWidget(self._small("smaller cells"))
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(0, threshold_position(FLOOR))
        self.threshold.setValue(threshold_position(THRESHOLD))
        self.threshold.sliderPressed.connect(self.remember)
        self.threshold.valueChanged.connect(self._threshold_moved)
        row.addWidget(self.threshold, stretch=3)
        row.addWidget(self._small("larger cells"))
        self.threshold_line = QLabel("")
        self.threshold_line.setMinimumWidth(
            self.threshold_line.fontMetrics().horizontalAdvance("-0.00 (default)")
        )
        row.addWidget(self.threshold_line)

        row.addSpacing(18)
        row.addWidget(QLabel("Area"))
        self.min_area = self._area_box("")
        self.max_area = self._area_box("no max")
        row.addWidget(self.min_area)
        row.addWidget(QLabel("to"))
        row.addWidget(self.max_area)

        return bar

    def _legend(self) -> QWidget:
        """What the colors of the outlines mean, above the image."""
        row = QWidget()
        items = QHBoxLayout(row)
        items.setContentsMargins(0, 0, 0, 6)
        items.setSpacing(5)

        for kind, text in LEGEND.items():
            items.addWidget(_Swatch(kind))
            items.addWidget(self._small(text))
            items.addSpacing(10)

        items.addStretch(1)

        return row

    def _help_card(self, parent: QWidget) -> QFrame:
        """The shortcuts and what the mouse does, in the middle of the image."""
        card = _Help(parent)
        card.setObjectName("help")
        card.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)
        card.installEventFilter(self.no_wheel)
        card.setStyleSheet(
            "QFrame#help { background: palette(window); border: 1px solid "
            f"palette(mid); border-radius: {style.RADIUS}px; }}"
            "QLabel#key { background: palette(base); border: 1px solid "
            "palette(mid); border-radius: 4px; padding: 0px 5px; }"
        )
        column = QVBoxLayout(card)
        column.setContentsMargins(18, 12, 18, 14)
        column.setSpacing(10)
        title = QLabel("Shortcuts")
        title.setStyleSheet("font-weight: bold;")
        column.addWidget(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(5)
        keys = [(_native(key), text, "key") for key, text in KEYS.values()]
        mouse = [(word, text, "") for word, text in MOUSE]

        for row, entry in enumerate([*keys, None, *mouse]):
            if entry is None:  # a gap between the keys and the mouse
                grid.setRowMinimumHeight(row, 4)
                continue

            key, text, kind = entry
            name = QLabel(key)
            name.setObjectName(kind)
            grid.addWidget(name, row, 0, alignment=Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(QLabel(text), row, 1)

        column.addLayout(grid)
        card.dismissed.connect(lambda: self.help_action.setChecked(False))
        card.hide()

        return card

    def _status(self) -> QHBoxLayout:
        """What the last click did, or what went wrong; the counts."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 8, 0)
        row.addWidget(self.line, stretch=1)
        self.counts = QLabel("")
        self.counts.setStyleSheet(style.text_style("computed"))
        row.addWidget(self.counts)

        return row

    def _card(self, parent: QWidget) -> QFrame:
        """
        The selected cell, floating over the image where it leaves the cell
        in view (`_place_card`): what it is, its size, its two buttons.
        """
        card = QFrame(parent)
        card.setObjectName("card")
        # Clicks and the wheel on it stay off the image (and off its slider:
        # the wheel is for zooming, not for resizing a cell by accident)
        card.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)
        card.installEventFilter(self.no_wheel)
        card.setStyleSheet(
            "QFrame#card { background: palette(window); border: 1px solid "
            f"palette(mid); border-radius: {style.RADIUS}px; }}"
        )
        card.setFixedWidth(CARD_WIDTH_PX)
        column = QVBoxLayout(card)
        column.setContentsMargins(12, 8, 12, 4)
        column.setSpacing(4)

        # Two lines, the second often empty, so the card keeps its height
        self.cell_line = QLabel("")
        self.cell_line.setStyleSheet("font-weight: bold;")
        self.cell_state = self._small("")
        self.cell_state.setMinimumHeight(self.cell_state.fontMetrics().height())
        title = QVBoxLayout()
        title.setSpacing(0)
        title.addWidget(self.cell_line)
        title.addWidget(self.cell_state)
        column.addLayout(title)

        size = QHBoxLayout()
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(0, SIZE_STEPS)
        self.size.installEventFilter(self.no_wheel)
        self.size.sliderPressed.connect(self.remember)
        self.size.sliderReleased.connect(self._place_card)
        self.size.valueChanged.connect(self._size_moved)
        size.addWidget(self.size, stretch=1)
        self.size_line = QLabel("")
        self.size_line.setStyleSheet(style.text_style("computed"))
        self.size_line.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        widest = self.size_line.fontMetrics().horizontalAdvance("9999 px, 9999.9 µm²")
        self.size_line.setMinimumWidth(widest)  # the slider keeps its length
        size.addWidget(self.size_line)
        column.addLayout(size)

        tools = QToolBar()
        tools.setIconSize(QSize(18, 18))
        tools.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.follow = QAction(icon("follow-threshold"), "Follow threshold", self)
        self.follow.setToolTip(self._tip("follow"))
        self.follow.triggered.connect(lambda _=False: self.follow_threshold())
        self.reject = QAction(icon("reject"), "Reject", self)
        self.reject.setToolTip(self._tip("reject"))
        self.reject.triggered.connect(lambda _=False: self.toggle_rejection())
        tools.addAction(self.follow)
        tools.addAction(self.reject)
        column.addWidget(tools)
        card.hide()

        return card

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(style.text_style("computed"))

        return label

    @staticmethod
    def _tip(name: str) -> str:
        """The tooltip of `KEYS[name]`: what it does, and its key."""
        key, title = KEYS[name]

        return f"{title} ({_native(key)})"

    def _area_box(self, empty: str) -> QDoubleSpinBox:
        """A bound of the area range, in um2; `empty` is how 0 shows."""
        box = QDoubleSpinBox()
        box.setRange(0.0, 100_000.0)
        box.setDecimals(0)
        box.setSingleStep(5.0)
        box.setSuffix(" µm²")

        if empty:
            box.setSpecialValueText(empty)

        # Applied as it changes, by the arrows or by typing, once that pauses
        box.valueChanged.connect(lambda _: self.ranging.start())

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
            ("Esc", self._escape),
            ("Delete", self.toggle_rejection),
            ("Backspace", self.toggle_rejection),
            ("[", lambda: self.nudge(-1)),
            ("]", lambda: self.nudge(1)),
            ("T", self.follow_threshold),
            ("I", self.image_button.showMenu),
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
                f"{cell.threshold:.2f}. Its size slider grows or shrinks it."
            )

        self.selected = basin
        self.changed()

    def _escape(self) -> None:
        """Esc: put the shortcuts away, or else let the selected cell go."""
        if self.help_action.isChecked():
            self.help_action.setChecked(False)
        else:
            self.select(None)

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
        self._place_card()

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
        default = " (default)" if threshold == THRESHOLD else ""
        self.threshold_line.setText(f"{threshold:.2f}{default}")

    def _show_counts(self) -> None:
        cells = list(self.curation.cells.values())
        kept = [cell for cell in cells if cell.kept]
        added = sum(cell.kind == "added" for cell in kept)
        resized = sum(cell.kind == "resized" for cell in kept)
        rejected = sum(cell.rejected for cell in cells)
        outside = sum(cell.excluded and not cell.rejected for cell in cells)
        noun = "cell" if len(kept) == 1 else "cells"
        self.counts.setText(
            f"{len(kept)} {noun} kept ({added} added, {resized} resized), "
            f"{rejected} rejected, {outside} outside the area range"
        )

    def _show_selected(self) -> None:
        basin = self.selected

        if basin is None:
            return

        cell = self.curation.cells.get(basin)
        size = 0 if cell is None else len(cell.pixels)
        total = len(self.curation.flood(basin)[0])

        if not self.size.isSliderDown():  # it would fight the pointer
            with _quiet(self.size):
                self.size.setValue(size_position(size, total))

        area = size * self.curation.um2_per_px
        self.size_line.setText(f"{size} px, {area:.1f} µm²")
        self.reject.setText(
            "Restore" if cell is not None and cell.rejected else "Reject"
        )

        if cell is None:  # the cross shows where it grows from
            threshold = self.curation.edits.threshold
            self.cell_line.setText(f"Cell {basin}, threshold {threshold:.2f}")
            self.cell_state.setText("no pixels at this threshold")
            return

        self.cell_line.setText(f"Cell {basin}, threshold {cell.threshold:.2f}")
        self.cell_state.setText(cell_state(cell))

    def _selected_box(self) -> None | QRectF:
        """Where the selected cell is on the canvas: its box, or its seed."""
        if self.curation is None or self.selected is None:
            return None

        cell = self.curation.cells.get(self.selected)

        if cell is not None:
            box = self._outline(cell)[1]
            corner = self.canvas.canvas_point(box.top(), box.left())
            far = self.canvas.canvas_point(box.bottom(), box.right())

            return QRectF(corner, far)

        row, column = divmod(self.curation.seed(self.selected), self.curation.width)
        at = self.canvas.canvas_point(row + 0.5, column + 0.5)

        return QRectF(at, at)

    def _place_card(self) -> None:
        """
        Show the card of the selected cell in the bottom right corner of the
        view, or else in the first other corner that leaves the cell in view
        (bottom left, top right, top left), or else in the corner farthest
        from it.
        """
        cell = self._selected_box()

        if cell is None:
            self.card.hide()
            return

        if self.size.isSliderDown():
            return  # not from under the pointer: once the slider is let go

        self.card.adjustSize()
        width, height = self.card.width(), self.card.height()
        right = self.canvas.width() - width - CARD_MARGIN_PX
        bottom = self.canvas.height() - height - CARD_MARGIN_PX
        corners = [
            QRectF(x, y, width, height)
            for x, y in (
                (right, bottom),
                (CARD_MARGIN_PX, bottom),
                (right, CARD_MARGIN_PX),
                (CARD_MARGIN_PX, CARD_MARGIN_PX),
            )
        ]
        room = CARD_MARGIN_PX
        around = cell.adjusted(-room, -room, room, room)
        clear = [corner for corner in corners if not corner.intersects(around)]

        if clear:
            place = clear[0]
        else:
            place = max(
                corners,
                key=lambda corner: (corner.center() - cell.center()).manhattanLength(),
            )

        self.card.move(place.topLeft().toPoint())
        self.card.show()

    def _show_help(self, on: bool) -> None:
        self.help.setVisible(on)
        self._place_help()

    def _place_help(self) -> None:
        """The shortcuts card, in the middle of the image."""
        if self.help.isVisible():
            self.help.adjustSize()
            x = (self.canvas.width() - self.help.width()) // 2
            y = (self.canvas.height() - self.help.height()) // 2
            self.help.move(max(x, 0), max(y, 0))
            self.help.raise_()

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
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

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

        if self.ranging.isActive():
            self.ranging.stop()
            self._area_changed()

        self.save_draft()
        self.loader.wait()
        self.closed.emit()
        super().closeEvent(event)
