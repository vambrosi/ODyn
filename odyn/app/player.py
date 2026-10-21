"""
Videos side by side, playing in step: the grid the QC window compares test
corrections (and raw and final means) in.

- `read_frames` reads a video's frames (grayscale) into memory; `FrameReader`
  does it off the main thread, so a share's slow reads never freeze the app.
- `VideoTile` shows one video's current frame, its name and a line of numbers.
  A click selects it (its border lights up) or lets it go.
- `VideoGrid` lays the tiles out, `columns` wide, and plays them together: one
  timer, one seek bar, play and pause, and a speed. The wheel zooms and a drag
  pans, on any tile, and every tile follows, so the same spot is compared.
  **Show selected** keeps the selected tiles only; **Show all** brings the
  others back.

No database in here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPen
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

# How fast the videos can play, in frames per second
SPEEDS = (2, 5, 10, 20, 30)
DEFAULT_SPEED = 10

# How far the wheel zooms per step, and the most it zooms in
ZOOM_STEP = 1.25
MAX_ZOOM = 16.0

# A selected tile's border, in px
BORDER_PX = 3

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


@dataclass
class Video:
    """A video to show: its name, a line of numbers, and its file."""

    name: str
    path: Path
    numbers: str = ""
    frames: None | np.ndarray = field(default=None, repr=False)


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
                video.frames = read_frames(video.path)
            except OSError as error:
                self.failed.emit(index, str(error))
                continue

            self.read.emit(index)


@dataclass
class View:
    """What every tile shows of its frame: a zoom and the point at the middle."""

    zoom: float = 1.0
    center: QPointF = field(default_factory=lambda: QPointF(0.5, 0.5))


class VideoTile(QWidget):
    """One video's current frame, its name and numbers; clicked, it is selected."""

    clicked = Signal()
    zoomed = Signal(float, QPointF)  # by how much, at which point of the frame
    panned = Signal(QPointF)  # by how much, as a fraction of the frame

    def __init__(self, video: Video, view: View, parent=None):
        super().__init__(parent)
        self.video = video
        self.view = view
        self.index = 0
        self.selected = False
        self._image: None | QImage = None
        self._press: None | QPointF = None
        self._dragged = False
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(f"{video.name}\n{video.numbers}".strip())

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
        the tile's top line; when not, they are left out (the tooltip has
        them).
        """
        if not self.video.numbers:
            return False

        metrics = self.fontMetrics()
        used = metrics.horizontalAdvance(name or self.video.name)
        used += metrics.horizontalAdvance(self.video.numbers)

        return used + NAME_GAP_PX <= self.width() - 12

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.update()

    def source_rect(self) -> QRectF:
        """The part of the frame shown, at the view's zoom and center."""
        if self._image is None:
            return QRectF()

        width, height = self._image.width(), self._image.height()
        w, h = width / self.view.zoom, height / self.view.zoom
        x = min(max(self.view.center.x() * width - w / 2, 0), width - w)
        y = min(max(self.view.center.y() * height - h / 2, 0), height - h)

        return QRectF(x, y, w, h)

    def target_rect(self) -> QRectF:
        """Where the frame goes in the tile: as large as fits, kept in shape."""
        inner = QRectF(self.rect()).adjusted(
            BORDER_PX, BORDER_PX + 18, -BORDER_PX, -BORDER_PX
        )
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

        painter.setPen(QColor(255, 220, 90))
        text = (
            self.video.name
            if self.video.frames is not None
            else (f"{self.video.name} (loading)")
        )
        painter.drawText(
            QRectF(6, 2, self.width() - 12, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )

        if self.numbers_fit(text):
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(
                QRectF(6, 2, self.width() - 12, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self.video.numbers,
            )

        if self.selected:
            pen = QPen(palette.color(QPalette.ColorRole.Highlight), BORDER_PX)
            painter.setPen(pen)
            half = BORDER_PX / 2
            painter.drawRect(QRectF(self.rect()).adjusted(half, half, -half, -half))

        painter.end()

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

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self._press is not None and not self._dragged:
            self.clicked.emit()

        self._press = None
        self._dragged = False


class VideoGrid(QWidget):
    """
    Videos side by side, playing in step (see the module's description).
    `frame_changed(index)` fires as the frame shown changes.
    """

    frame_changed = Signal(int)
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = View()
        self.tiles: list[VideoTile] = []
        self.only_selected = False
        self.reader: None | FrameReader = None
        self.index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.holder = QWidget()
        self.layout_grid = QGridLayout(self.holder)
        self.layout_grid.setSpacing(4)
        self.area.setWidget(self.holder)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        self.position = QLabel("")
        self.speed = QComboBox()
        self.speed.addItems([f"{fps} frames/s" for fps in SPEEDS])
        self.speed.setCurrentIndex(SPEEDS.index(DEFAULT_SPEED))
        self.speed.currentIndexChanged.connect(self._set_speed)
        self.columns = QSpinBox()
        self.columns.setRange(1, 8)
        self.columns.setPrefix("columns: ")
        self.columns.valueChanged.connect(self._lay_out)
        self.show_selected = QPushButton("Show selected")
        self.show_selected.clicked.connect(lambda: self.filter_selected(True))
        self.show_all = QPushButton("Show all")
        self.show_all.clicked.connect(lambda: self.filter_selected(False))
        self.reset_view = QPushButton("Fit")
        self.reset_view.setToolTip("Zoom back out to the whole frame")
        self.reset_view.clicked.connect(self.fit)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, stretch=1)
        controls.addWidget(self.position)
        controls.addWidget(self.speed)
        controls.addWidget(self.columns)
        controls.addWidget(self.reset_view)
        controls.addWidget(self.show_selected)
        controls.addWidget(self.show_all)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.area, stretch=1)
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

        for tile in self.tiles:
            tile.setParent(None)
            tile.deleteLater()

        self.tiles = []
        self.view = View()
        self.only_selected = False

        for video in videos:
            tile = VideoTile(video, self.view)
            tile.clicked.connect(lambda t=tile: self._toggle(t))
            tile.zoomed.connect(self._zoom)
            tile.panned.connect(self._pan)
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
        tile.setToolTip(problem)
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
            self.timer.start()
            self.play_button.setText("Pause")

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

    def _step(self) -> None:
        count = self.frame_count
        self.seek((self.index + 1) % count if count else 0)

    def _set_speed(self, *_) -> None:
        self.timer.setInterval(int(1000 / SPEEDS[self.speed.currentIndex()]))

    def _enable(self) -> None:
        count = self.frame_count
        self.slider.setRange(0, max(0, count - 1))
        self.play_button.setEnabled(count > 1)
        self.slider.setEnabled(count > 1)
        self.show_selected.setEnabled(any(t.selected for t in self.tiles))
        self.show_all.setEnabled(self.only_selected)

        if count:
            self.position.setText(f"{self.index + 1} / {count}")

    # ----------------------------------------------------------------------- #
    # Zoom, pan, selection
    # ----------------------------------------------------------------------- #

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

    def _toggle(self, tile: VideoTile) -> None:
        tile.set_selected(not tile.selected)
        self._enable()
        self.selection_changed.emit()

    @property
    def selected(self) -> list[Video]:
        return [tile.video for tile in self.tiles if tile.selected]

    def filter_selected(self, only: bool) -> None:
        """Show the selected tiles only (`only`), or every tile again."""
        self.only_selected = only and any(t.selected for t in self.tiles)
        self._lay_out()
        self._enable()

    def visible(self) -> list[Video]:
        return [tile.video for tile in self.tiles if not tile.isHidden()]

    def _lay_out(self, *_) -> None:
        for tile in self.tiles:
            self.layout_grid.removeWidget(tile)

        shown = [t for t in self.tiles if t.selected or not self.only_selected]
        columns = max(1, self.columns.value())

        for tile in self.tiles:
            tile.setVisible(tile in shown)

        for position, tile in enumerate(shown):
            self.layout_grid.addWidget(tile, position // columns, position % columns)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.pause()
        self.stop_reading()
        super().closeEvent(event)
