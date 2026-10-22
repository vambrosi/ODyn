"""
Videos side by side, playing in step: the grid the QC window compares test
corrections (and raw and final means) in.

- `read_frames` reads a video's frames (grayscale) into memory; `FrameReader`
  does it off the main thread, so a share's slow reads never freeze the app.
- `VideoTile` shows one video's current frame, its name and a line of numbers.
  A click selects it (its border lights up) or lets it go.
- `VideoGrid` lays the tiles out, `columns` wide, and plays them together: one
  timer, one seek bar, play and pause, and a speed. The wheel zooms and a drag
  pans, on any tile, and every tile follows, so the same spot is compared;
  zoomed in, the part shown fills the tile. The pointer over one tile shows as
  a cross at the same spot on the others.
- Tracking the candidates: a click **focuses** a tile, shift-click adds or
  removes one; `clear_focus` lets them all go, changing nothing else.
  `star_focused` marks them (the star stays in every view); `set_aside_focused`
  takes them out of play, into a strip under the grid in the order set aside,
  and `bring_back` puts the last one (or a clicked one) back.
  `show_focused_only` keeps the tiles focused at that moment, until turned
  off. The buttons and keys for these are the window's (`mcor_window`).

No database in here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PySide6.QtCore import (
    QElapsedTimer,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# How fast the videos can play, in frames per second. Screens refresh near
# 60 Hz, so faster speeds step more than one frame per refresh
SPEEDS = (10, 20, 30, 60, 120, 240)
DEFAULT_SPEED = 10
MAX_REFRESH = 60

# The mirrored pointer: the arms of its cross, in px, and its colors
CROSS_PX = 8
CROSS_COLOR = QColor(0, 255, 160)
CROSS_EDGE = QColor(0, 0, 0)

# How far the wheel zooms per step, and the most it zooms in
ZOOM_STEP = 1.25
MAX_ZOOM = 16.0

# A focused tile's border, in px
BORDER_PX = 3

# The strip of tiles set aside: its height, and how dark they are drawn
ASIDE_HEIGHT_PX = 110
ASIDE_WIDTH_PX = 150
QWIDGETSIZE_MAX = 16777215  # Qt's "no limit"
ASIDE_SHADE = QColor(0, 0, 0, 150)
STAR_COLOR = QColor(255, 200, 40)

# The least room between a tile's name and its numbers, in px
NAME_GAP_PX = 16


def read_frames(path: str | Path) -> np.ndarray:
    """Every frame of the video at `path`, grayscale, as `(frames, h, w)` uint8."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise OSError(f"No frames could be read from '{path}'.")

    return np.stack(frames)


class FrameStream:
    """
    A video's frames read from its file as they are shown, for videos too
    long to hold in memory (a full run's: thousands of frames). Indexed and
    measured as the array `read_frames` returns; playing on reads the next
    frame, a jump seeks (each frame of an MJPG video stands alone, so it is
    quick).
    """

    def __init__(self, path: str | Path):
        import cv2

        self.capture = cv2.VideoCapture(str(path))
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next = 0

        if not self.capture.isOpened() or self.count <= 0:
            self.capture.release()
            raise OSError(f"No frames could be read from '{path}'.")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> np.ndarray:
        import cv2

        if index != self.next:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = self.capture.read()
        self.next = index + 1

        if not ok:
            return np.zeros((1, 1), np.uint8)

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def close(self) -> None:
        self.capture.release()


@dataclass
class Video:
    """
    A video to show: its name, a line of numbers, and its file, read whole
    into memory, or as shown (`stream`, for long ones).
    """

    name: str
    path: Path
    numbers: str = ""
    candidate: bool = True  # can be set aside (not the raw video)
    color: None | QColor = None  # its color in the plots, as a square by its name
    stream: bool = False
    frames: None | np.ndarray | FrameStream = field(default=None, repr=False)


class FrameReader(QThread):
    """
    Reads the videos' frames off the main thread, one after the other.
    `read` says which one is done (by index), `failed` which one could not be
    read and why.
    """

    read = Signal(int)
    failed = Signal(int, str)

    def __init__(self, videos: list[Video]):
        super().__init__()
        self.videos = videos

    def run(self) -> None:
        for index, video in enumerate(self.videos):
            if self.isInterruptionRequested():
                return

            try:
                video.frames = (
                    FrameStream(video.path) if video.stream else read_frames(video.path)
                )
            except OSError as error:
                self.failed.emit(index, str(error))
                continue

            self.read.emit(index)


@dataclass
class View:
    """
    What every tile shows of its frame: a zoom and the point at the middle,
    and where the pointer is (as a fraction of the frame, over which tile),
    mirrored on the others with `mirror`.
    """

    zoom: float = 1.0
    center: QPointF = field(default_factory=lambda: QPointF(0.5, 0.5))
    pointer: None | QPointF = None
    pointer_tile: None | object = None
    mirror: bool = True


class VideoTile(QWidget):
    """
    One video's current frame, its name and numbers; a star when starred, a
    border when focused, darker when set aside. `clicked(add)` fires on a
    click (`add`: with Shift), `menu_requested` on a right click.
    """

    clicked = Signal(bool)
    menu_requested = Signal(QPoint)  # where, on the screen
    zoomed = Signal(float, QPointF)  # by how much, at which point of the frame
    panned = Signal(QPointF)  # by how much, as a fraction of the frame
    hovered = Signal(object)  # the pointer, as a fraction of the frame, or None

    def __init__(self, video: Video, view: View, parent=None):
        super().__init__(parent)
        self.video = video
        self.view = view
        self.index = 0
        self.focused = False
        self.starred = False
        self.aside_rank: None | int = None  # 1 for the first set aside
        self._image: None | QImage = None
        self.problem = ""  # why the video could not be read
        self._press: None | QPointF = None
        self._dragged = False
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)  # for the mirrored pointer

    @property
    def frame_count(self) -> int:
        return 0 if self.video.frames is None else len(self.video.frames)

    def show_frame(self, index: int) -> None:
        frames = self.video.frames

        if frames is None or not len(frames):
            self._image = None
        else:
            self.index = min(index, len(frames) - 1)
            frame = np.ascontiguousarray(frames[self.index])
            height, width = frame.shape
            self._image = QImage(
                frame.data, width, height, width, QImage.Format.Format_Grayscale8
            ).copy()  # the array may change under it

        self.update()

    def numbers_fit(self, name: None | str = None) -> bool:
        """
        Whether the numbers fit beside the name (`name`, or the video's), on
        the tile's top line; when not, they are left out.
        """
        if not self.video.numbers:
            return False

        metrics = self.fontMetrics()
        used = metrics.horizontalAdvance(name or self.video.name)
        used += metrics.horizontalAdvance(self.video.numbers)

        return used + NAME_GAP_PX <= self.width() - 12

    def set_focused(self, focused: bool) -> None:
        self.focused = focused
        self.update()

    def _inner(self) -> QRectF:
        """The tile under its top line: where the frame goes."""
        return QRectF(self.rect()).adjusted(
            BORDER_PX, BORDER_PX + 18, -BORDER_PX, -BORDER_PX
        )

    def source_rect(self) -> QRectF:
        """
        The part of the frame shown, at the view's zoom and center. Zoomed in,
        it widens (or heightens) to the tile's shape, as far as the frame
        goes, so the tile is filled.
        """
        if self._image is None:
            return QRectF()

        width, height = self._image.width(), self._image.height()
        w, h = width / self.view.zoom, height / self.view.zoom
        inner = self._inner()

        if self.view.zoom > 1 and inner.width() > 0 and inner.height() > 0:
            shape = inner.width() / inner.height()

            if w / h < shape:
                w = min(width, h * shape)
            else:
                h = min(height, w / shape)

        x = min(max(self.view.center.x() * width - w / 2, 0), width - w)
        y = min(max(self.view.center.y() * height - h / 2, 0), height - h)

        return QRectF(x, y, w, h)

    def target_rect(self) -> QRectF:
        """Where the frame goes in the tile: as large as fits, kept in shape."""
        inner = self._inner()
        source = self.source_rect()

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

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        painter = QPainter(self)
        palette = self.palette()
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self._image is not None:
            painter.drawImage(self.target_rect(), self._image, self.source_rect())

        left = 6.0

        if self.video.color is not None:
            painter.fillRect(QRectF(left, 5, 10, 10), self.video.color)
            left += 14

        if self.starred:
            painter.setPen(STAR_COLOR)
            painter.drawText(
                QRectF(left, 2, 16, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                "\u2605",
            )
            left += 16

        painter.setPen(QColor(255, 220, 90))
        text = (
            self.video.name
            if self.video.frames is not None
            else (f"{self.video.name} (loading)")
        )

        if self.aside_rank is not None:
            text = f"{self.aside_rank}. {text}"

        painter.drawText(
            QRectF(left, 2, self.width() - left - 6, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

        if self.aside_rank is None and self.numbers_fit(text):
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(
                QRectF(6, 2, self.width() - 12, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self.video.numbers,
            )

        self._draw_pointer(painter)

        if self.aside_rank is not None:
            painter.fillRect(self.rect(), ASIDE_SHADE)

        if self.focused:
            pen = QPen(palette.color(QPalette.ColorRole.Highlight), BORDER_PX)
            painter.setPen(pen)
            half = BORDER_PX / 2
            painter.drawRect(QRectF(self.rect()).adjusted(half, half, -half, -half))

        painter.end()

    def tile_point(self, point: QPointF) -> None | QPointF:
        """Where a point of the frame (as a fraction) is in the tile, if shown."""
        target, source = self.target_rect(), self.source_rect()

        if self._image is None or source.isEmpty():
            return None

        x = point.x() * self._image.width()
        y = point.y() * self._image.height()

        if not source.contains(QPointF(x, y)):
            return None

        return QPointF(
            target.x() + (x - source.x()) / source.width() * target.width(),
            target.y() + (y - source.y()) / source.height() * target.height(),
        )

    def _draw_pointer(self, painter: QPainter) -> None:
        """The pointer over another tile, as a cross at the same frame point."""
        view = self.view

        if not view.mirror or view.pointer is None or view.pointer_tile is self:
            return

        at = self.tile_point(view.pointer)

        if at is None:
            return

        for color, width in ((CROSS_EDGE, 3), (CROSS_COLOR, 1.5)):
            painter.setPen(QPen(color, width))
            painter.drawLine(at - QPointF(CROSS_PX, 0), at + QPointF(CROSS_PX, 0))
            painter.drawLine(at - QPointF(0, CROSS_PX), at + QPointF(0, CROSS_PX))

    def _frame_point(self, position: QPointF) -> QPointF:
        """A point of the tile as a fraction of the frame."""
        target, source = self.target_rect(), self.source_rect()

        if self._image is None or target.isEmpty():
            return QPointF(0.5, 0.5)

        x = source.x() + (position.x() - target.x()) / target.width() * source.width()
        y = source.y() + (position.y() - target.y()) / target.height() * source.height()

        return QPointF(x / self._image.width(), y / self._image.height())

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        step = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        self.zoomed.emit(step, self._frame_point(event.position()))
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
            self._dragged = False

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self._image is not None and self.target_rect().contains(event.position()):
            self.hovered.emit(self._frame_point(event.position()))
        else:
            self.hovered.emit(None)

        if self._press is None or self._image is None:
            return

        target, source = self.target_rect(), self.source_rect()
        moved = event.position() - self._press

        if abs(moved.x()) + abs(moved.y()) < 4 and not self._dragged:
            return

        self._dragged = True
        self._press = event.position()

        if target.width() and target.height():
            self.panned.emit(
                QPointF(
                    -moved.x() / target.width() * source.width() / self._image.width(),
                    -moved.y()
                    / target.height()
                    * source.height()
                    / self._image.height(),
                )
            )

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.hovered.emit(None)
        super().leaveEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.menu_requested.emit(event.globalPos())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self._press is not None and not self._dragged:
            add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.clicked.emit(add)

        self._press = None
        self._dragged = False


class VideoGrid(QWidget):
    """
    Videos side by side, playing in step (see the module's description).
    `frame_changed(index)` fires as the frame shown changes, `focus_changed`
    as the focus, the stars or the tiles set aside change, and
    `menu_requested(tile, where)` on a right click on a tile (focused first).
    """

    frame_changed = Signal(int)
    focus_changed = Signal()
    hovered = Signal(object)  # the video under the pointer, or None
    menu_requested = Signal(object, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = View()
        self.tiles: list[VideoTile] = []
        self.aside: list[VideoTile] = []  # in the order set aside
        self.only: None | list[VideoTile] = None  # `show_focused_only`
        self.reader: None | FrameReader = None
        self.index = 0

        self.fps = DEFAULT_SPEED
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._step)

        # Playing follows the clock, not the timer's ticks: a tick late (many
        # tiles, the plots) shows a later frame, so the speed stays true
        self.clock = QElapsedTimer()
        self.played_from = 0  # the frame the clock started at

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.holder = QWidget()
        self.layout_grid = QGridLayout(self.holder)
        self.layout_grid.setSpacing(4)
        self.area.setWidget(self.holder)

        self.strip = QWidget()
        self.strip.setFixedHeight(ASIDE_HEIGHT_PX)
        self.layout_strip = QHBoxLayout(self.strip)
        self.layout_strip.setContentsMargins(0, 4, 0, 0)
        self.layout_strip.setSpacing(4)
        self.strip.hide()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._slid)
        self.position = QLabel("")
        self.speed = QComboBox()
        self.speed.addItems([f"{fps} frames/s" for fps in SPEEDS])
        self.speed.setCurrentIndex(SPEEDS.index(DEFAULT_SPEED))
        self.speed.currentIndexChanged.connect(self._set_speed)
        self.columns = QSpinBox()
        self.columns.setRange(1, 8)
        self.columns.setPrefix("columns: ")
        self.columns.valueChanged.connect(self._lay_out)

        # The keys belong to the window's shortcuts (Space, arrows), not to
        # whichever control was clicked last
        self.play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        controls = QHBoxLayout()
        middle = Qt.AlignmentFlag.AlignVCenter

        for widget, stretch in (
            (self.play_button, 0),
            (self.slider, 1),
            (self.position, 0),
            (self.speed, 0),
            (self.columns, 0),
        ):
            # macOS lays a button out with room for its native look, which
            # the stylesheet replaced: the widget itself, so the row lines up
            widget.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)
            controls.addWidget(widget, stretch, middle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.area, stretch=1)
        layout.addWidget(self.strip)
        layout.addLayout(controls)
        self._set_speed()
        self._enable()

    # ----------------------------------------------------------------------- #
    # Videos
    # ----------------------------------------------------------------------- #

    def set_videos(self, videos: Sequence[Video], columns: None | int = None) -> None:
        """Show `videos` (reading their frames in the background), from frame 0."""
        self.stop_reading()
        self.pause()

        self.close_streams()

        for tile in self.tiles:
            tile.setParent(None)
            tile.deleteLater()

        self.tiles = []
        self.aside = []
        self.only = None
        self.view = View(mirror=self.view.mirror)

        for video in videos:
            tile = VideoTile(video, self.view)
            tile.clicked.connect(lambda add, t=tile: self._clicked(t, add))
            tile.menu_requested.connect(lambda where, t=tile: self._menu(t, where))
            tile.zoomed.connect(self._zoom)
            tile.panned.connect(self._pan)
            tile.hovered.connect(lambda point, t=tile: self._hover(t, point))
            self.tiles.append(tile)

        self.columns.blockSignals(True)
        self.columns.setValue(columns or max(1, int(np.ceil(np.sqrt(len(videos))))))
        self.columns.blockSignals(False)
        self._lay_out()

        pending = [video for video in videos if video.frames is None]

        if pending:
            self.reader = FrameReader(list(videos))
            self.reader.read.connect(self._read)
            self.reader.failed.connect(self._failed)
            self.reader.start()

        self.seek(0)

    def stop_reading(self) -> None:
        if self.reader is not None:
            self.reader.requestInterruption()
            self.reader.wait()
            self.reader = None

    @property
    def frame_count(self) -> int:
        """The frames every loaded video has (they play in step)."""
        counts = [tile.frame_count for tile in self.tiles if tile.frame_count]

        return min(counts) if counts else 0

    def _read(self, index: int) -> None:
        self.tiles[index].show_frame(self.index)
        self._enable()

    def _failed(self, index: int, problem: str) -> None:
        tile = self.tiles[index]
        tile.video.numbers = "could not be read"
        tile.problem = problem
        tile.update()

    # ----------------------------------------------------------------------- #
    # Playing
    # ----------------------------------------------------------------------- #

    def toggle(self) -> None:
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.frame_count > 1:
            self._restart_clock()
            self.timer.start()
            self.play_button.setText("Pause")

    def _restart_clock(self) -> None:
        self.played_from = self.index
        self.clock.start()

    def pause(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")

    def seek(self, index: int) -> None:
        """Show frame `index` in every tile."""
        count = self.frame_count
        self.index = 0 if not count else max(0, min(index, count - 1))

        for tile in self.tiles:
            tile.show_frame(self.index)

        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)
        self.position.setText(f"{self.index + 1} / {count}" if count else "")
        self.frame_changed.emit(self.index)

    def _slid(self, index: int) -> None:
        """The slider moved: show its frame, and play on from there."""
        self.seek(index)

        if self.timer.isActive():
            self._restart_clock()

    def _step(self) -> None:
        self.seek(self.frame_at(self.clock.elapsed()))

    def frame_at(self, elapsed_ms: float) -> int:
        """The frame due `elapsed_ms` after playing started, going round."""
        count = self.frame_count

        if not count:
            return 0

        return (self.played_from + int(elapsed_ms * self.fps / 1000)) % count

    def step_by(self, frames: int) -> None:
        """Pause, and go `frames` forward (back when negative)."""
        self.pause()
        self.seek(self.index + frames)

    def change_speed(self, by: int) -> None:
        """Go `by` speeds faster (slower when negative), within `SPEEDS`."""
        index = min(max(self.speed.currentIndex() + by, 0), len(SPEEDS) - 1)
        self.speed.setCurrentIndex(index)

    def _set_speed(self, *_) -> None:
        """A tick per frame, up to the screen's refresh (frames then skip)."""
        self.fps = SPEEDS[self.speed.currentIndex()]
        self.timer.setInterval(int(1000 / min(self.fps, MAX_REFRESH)))

        if self.timer.isActive():
            self._restart_clock()

    def _enable(self) -> None:
        count = self.frame_count
        self.slider.setRange(0, max(0, count - 1))
        self.play_button.setEnabled(count > 1)
        self.slider.setEnabled(count > 1)

        if count:
            self.position.setText(f"{self.index + 1} / {count}")

    # ----------------------------------------------------------------------- #
    # Zoom, pan, pointer
    # ----------------------------------------------------------------------- #

    def _hover(self, tile: VideoTile, point: None | QPointF) -> None:
        """Mirror the pointer over `tile` on the others."""
        view = self.view

        if point is None and view.pointer_tile is not tile:
            return  # left a tile after entering the next

        before = view.pointer_tile
        view.pointer = point
        view.pointer_tile = None if point is None else tile

        if view.pointer_tile is not before:
            self.hovered.emit(None if point is None else tile.video)

        if view.mirror:
            for other in self.tiles:
                if other is not tile:
                    other.update()

    def set_mirror(self, on: bool) -> None:
        """Show the pointer over one tile on the others, or not."""
        self.view.mirror = on

        for tile in self.tiles:
            tile.update()

    def _zoom(self, step: float, at: QPointF) -> None:
        old = self.view.zoom
        new = min(max(old * step, 1.0), MAX_ZOOM)

        # The point under the pointer stays under it
        c = self.view.center
        self.view.center = QPointF(
            at.x() + (c.x() - at.x()) * old / new, at.y() + (c.y() - at.y()) * old / new
        )
        self.view.zoom = new
        self._clamp()

    def _pan(self, by: QPointF) -> None:
        self.view.center = self.view.center + by
        self._clamp()

    def _clamp(self) -> None:
        half = 0.5 / self.view.zoom
        c = self.view.center
        self.view.center = QPointF(
            min(max(c.x(), half), 1 - half), min(max(c.y(), half), 1 - half)
        )

        for tile in self.tiles:
            tile.update()

    def fit(self) -> None:
        self.view.zoom = 1.0
        self.view.center = QPointF(0.5, 0.5)
        self._clamp()

    # ----------------------------------------------------------------------- #
    # Focus, stars, set aside
    # ----------------------------------------------------------------------- #

    def _clicked(self, tile: VideoTile, add: bool) -> None:
        """
        Focus `tile` alone, or add or remove it (`add`). A tile set aside
        comes back instead.
        """
        if tile in self.aside:
            self.bring_back(tile)
            return

        if add:
            tile.set_focused(not tile.focused)
        else:
            for other in self.tiles:
                other.set_focused(other is tile)

        self._changed()

    def _menu(self, tile: VideoTile, where: QPoint) -> None:
        if tile not in self.aside and not tile.focused:
            self._clicked(tile, add=False)

        self.menu_requested.emit(tile, where)

    def _changed(self) -> None:
        self._enable()
        self.focus_changed.emit()

    @property
    def focused(self) -> list[Video]:
        return [tile.video for tile in self.tiles if tile.focused]

    @property
    def starred(self) -> list[Video]:
        return [tile.video for tile in self.tiles if tile.starred]

    @property
    def set_aside(self) -> list[Video]:
        """The videos set aside, the first set aside first."""
        return [tile.video for tile in self.aside]

    def clear_focus(self) -> None:
        """Let every tile go; what is shown stays as it is."""
        for tile in self.tiles:
            tile.set_focused(False)

        self._changed()

    def star_focused(self) -> None:
        """Star the focused tiles, or unstar them when all are starred."""
        focused = [tile for tile in self.tiles if tile.focused]
        star = not all(tile.starred for tile in focused)

        for tile in focused:
            tile.starred = star
            tile.update()

        self._changed()

    def set_aside_focused(self) -> None:
        """Take the focused candidates out of play, in the strip."""
        for tile in self.tiles:
            if tile.focused and tile.video.candidate and tile not in self.aside:
                tile.set_focused(False)
                self.aside.append(tile)

        self._lay_out()
        self._changed()

    def bring_back(self, tile: None | VideoTile = None) -> None:
        """Put `tile` back in play: the last one set aside by default."""
        if not self.aside:
            return

        tile = tile if tile in self.aside else self.aside[-1]
        self.aside.remove(tile)

        if self.only is not None:
            self.only.append(tile)

        self._lay_out()
        self._changed()

    def show_focused_only(self, on: bool) -> None:
        """Show the tiles focused now, and only them, until turned off."""
        focused = [tile for tile in self.tiles if tile.focused]
        self.only = focused if on and focused else None
        self._lay_out()
        self._changed()

    def visible(self) -> list[Video]:
        return [tile.video for tile in self.tiles if not tile.isHidden()]

    def _lay_out(self, *_) -> None:
        for tile in self.tiles:
            self.layout_grid.removeWidget(tile)
            self.layout_strip.removeWidget(tile)

        while self.layout_strip.count():  # the stretch
            self.layout_strip.takeAt(0)

        playing = [t for t in self.tiles if t not in self.aside]
        shown = [t for t in playing if self.only is None or t in self.only]
        columns = max(1, self.columns.value())

        for tile in self.tiles:
            tile.setVisible(tile in shown or tile in self.aside)
            tile.aside_rank = None
            tile.setMinimumSize(120, 120)
            tile.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)

        for position, tile in enumerate(shown):
            self.layout_grid.addWidget(tile, position // columns, position % columns)

        for rank, tile in enumerate(self.aside, start=1):
            tile.aside_rank = rank
            tile.setMinimumSize(80, 60)
            tile.setMaximumSize(ASIDE_WIDTH_PX, ASIDE_HEIGHT_PX - 4)
            self.layout_strip.addWidget(tile)

        self.layout_strip.addStretch(1)
        self.strip.setVisible(bool(self.aside))

    def close_streams(self) -> None:
        for tile in self.tiles:
            if isinstance(tile.video.frames, FrameStream):
                tile.video.frames.close()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.pause()
        self.stop_reading()
        self.close_streams()
        super().closeEvent(event)
