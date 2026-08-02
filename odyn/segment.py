# --------------------------------------------------------------------------- #
#
# STATUS: standalone prototype, i.e. no DB interaction. `pick_rois` writes
# a JSON file that maps onto a future `rois` table (one row per polygon).
#
# TODO:
#   - Use database mcor file paths directly
#   - Reload the last saved ROIs when the GUI opens
#
# --------------------------------------------------------------------------- #

"""
Quick segmentation of ROIs from an image (e.g. z-score image).
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.draw import polygon2mask
from skimage.measure import approximate_polygon, find_contours, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .utils import logger

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Defaults shared by find_rois and the GUI.
DEFAULT_DIAMETER_UM = 60.0
DEFAULT_THRESHOLD = 2.0
DEFAULT_MIN_AREA_FRACTION = 0.2
DEFAULT_BORDER_UM = 0.0

# How solid the inside of an ROI looks (0 would be outline only).
ROI_FILL_ALPHA = 0.15

# Color of the excluded regions drawn over the image.
EXCLUDE_COLOR = "#cccc00"

# Ratio between the two Gaussians of the difference-of-Gaussians filter.
# A ratio of 1.6 gives an approximation to the Laplacian-of-Gaussian filter.
#       (Check https://en.wikipedia.org/wiki/Difference_of_Gaussians.)
DOG_RATIO = 1.6

# Colors of the diverging colormap (same ones used in the MATLAB figures).
COLOR_LOW = (5, 48, 97)  # dark blue
COLOR_MID = (247, 247, 247)  # off white
COLOR_HIGH = (103, 0, 31)  # dark red

# A polygon is an (N, 2) array of (x, y) pixel coordinates.
type Polygon = np.ndarray


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def load_image(image: str | Path | np.ndarray) -> np.ndarray:
    """
    Read a response map from a `.npy` or `.tif` file (or pass an array through).
    """

    if isinstance(image, np.ndarray):
        array = image

    else:
        path = Path(image)
        if path.suffix == ".npy":
            array = np.load(path)
        else:
            import tifffile

            array = tifffile.imread(path)

    array = np.asarray(array, dtype="float32")

    # A single-frame TIFF may still have a leading axis of length 1
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image but got shape {array.shape}.")

    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def find_rois(
    image: str | Path | np.ndarray,
    *,
    um_per_px: float = 1.0,
    diameter_um: float = DEFAULT_DIAMETER_UM,
    threshold: float = DEFAULT_THRESHOLD,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    border_um: float = DEFAULT_BORDER_UM,
    exclude: list[Polygon] | None = None,
    omit_messages: bool = False,
) -> list[Polygon]:
    """
    Find ROIs in an image and return them as polygons.

    **PARAMETERS**
    - `image`: signed image (e.g. z-scores), array, or path to `.npy`/`.tif`
    - `um_per_px`: size of a pixel, used to read every other parameter in um
    - `diameter_um`: typical ROI diameter
    - `threshold`: exclude responses around zero (-threshold < z < threshold)
    - `min_area_fraction`: drop ROIs smaller than this fraction of the typical
            ROI (0 keeps everything)
    - `border_um`: drop ROIs that come within this distance of the image edge
    - `exclude`: polygons covering excluded areas (e.g. the midline sinus).
    - `omit_messages`: omit logging messages (used in the GUI).

    **EXAMPLES**
    ```python
    rois = find_rois("envelope.npy", diameter_um=50, threshold=2.5)
    ```
    """

    array = load_image(image)
    radius_px = 0.5 * diameter_um / um_per_px

    if radius_px < 1:
        raise ValueError("'diameter_um' is smaller than two pixels.")

    # Look for ROIs around peaks in either direction.
    # Keep the sign only for display purposes.
    response = np.abs(array)

    # Excluded areas are kept out of the blur, the mask and the seeds, so what
    # is inside them never bleeds into the ROIs found next to them.
    excluded = _excluded_mask(array.shape, exclude)
    weight = None if excluded is None else (~excluded).astype("float32")

    # Difference of Gaussians tuned to the ROI scale.
    #
    # Goals:
    #   - Removes small-scale noise;
    #   - Gives peaks near ROI centers.

    sigma = radius_px / np.sqrt(2)
    smooth = _blur(response, sigma, weight)
    dog = smooth - _blur(response, sigma * DOG_RATIO, weight)

    mask = smooth > threshold

    if excluded is not None:
        mask &= ~excluded

    if not mask.any():
        logger.warning(f"Nothing above threshold = {threshold}.")
        return []

    # One seed per ROI: peaks closer than a radius are "merged".
    peaks = peak_local_max(
        dog,
        min_distance=max(1, round(radius_px)),
        labels=mask,
        exclude_border=False,
    )

    if len(peaks) == 0:
        logger.warning("No peaks found; try a smaller 'diameter_um'.")
        return []

    markers = np.zeros(dog.shape, dtype="int32")
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)

    # Splits touching ROIs along the dim valley between their centers.
    labels = watershed(-dog, markers, mask=mask)

    # --- Filter and convert each region to a polygon --- #

    min_area_px = min_area_fraction * np.pi * radius_px**2
    border_px = border_um / um_per_px
    height, width = labels.shape

    # An ROI touching the excluded area is dropped.
    touching = set()

    if excluded is not None:
        rim = binary_dilation(excluded) & ~excluded
        touching = {int(label) for label in np.unique(labels[rim])} - {0}

    # Enough to drop the staircase of single pixels without losing the shape
    tolerance_px = max(1.0, 0.05 * radius_px)

    polygons = []

    for region in regionprops(labels):
        if region.area < min_area_px or region.label in touching:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        if border_px and (
            min_row < border_px
            or min_col < border_px
            or max_row > height - border_px
            or max_col > width - border_px
        ):
            continue

        # Contour the region alone, inside its bounding box. The padding gives
        # find_contours a zero rim so regions touching the box still close.
        patch = np.pad(labels[min_row:max_row, min_col:max_col] == region.label, 1)
        contours = find_contours(patch.astype("float32"), 0.5)

        if not contours:
            continue

        # (row, col) inside the padded box -> (x, y) in the full image
        contour = approximate_polygon(max(contours, key=len), tolerance_px)
        polygon = np.column_stack(
            [contour[:, 1] + min_col - 1, contour[:, 0] + min_row - 1]
        )

        polygons.append((region.centroid, polygon))

    # Number ROIs in reading order so they stay comparable between runs
    polygons.sort(key=lambda item: item[0])

    if not omit_messages:
        logger.info(f"Found {len(polygons)} ROIs.")

    return [polygon for _centroid, polygon in polygons]


def _excluded_mask(
    shape: tuple[int, ...], polygons: list[Polygon] | None
) -> np.ndarray | None:
    """Paint the excluded polygons onto a mask (`None` if there are none)."""

    if not polygons:
        return None

    mask = np.zeros(shape, dtype=bool)

    for polygon in polygons:
        polygon = np.asarray(polygon, dtype="float64")

        # Fewer than 3 corners has no inside (e.g. a half-drawn polygon)
        if len(polygon) >= 3:
            # Our polygons are (x, y), but polygon2mask wants (row, col)
            mask |= polygon2mask(shape, polygon[:, ::-1])

    return mask if mask.any() else None


def _blur(field: np.ndarray, sigma: float, weight: np.ndarray | None) -> np.ndarray:
    """
    Gaussian blur that ignores the excluded pixels.

    Setting the excluded pixels to zero and blurring would drag down every
    value within a blur radius of them, shrinking the ROIs sitting next to an
    excluded area (or losing them below the threshold). Blurring the weights
    as well and dividing averages over the pixels that are left instead.
    """

    if weight is None:
        return gaussian_filter(field, sigma)

    return gaussian_filter(field * weight, sigma) / np.maximum(
        gaussian_filter(weight, sigma), 1e-6
    )


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #


def pick_rois(
    image: str | Path | np.ndarray,
    *,
    um_per_px: float = 1.0,
    save_path: str | Path = "rois.json",
    diameter_um: float = DEFAULT_DIAMETER_UM,
    threshold: float = DEFAULT_THRESHOLD,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    border_um: float = DEFAULT_BORDER_UM,
    color_limit: float = 5.0,
    max_preview_px: int = 1024,
) -> None:
    """
    Open a GUI to fine tune the segmentation algorithm.

    Change the parameters until most ROIs are outlined, click the ones
    that came out wrong to delete them, draw any that are missing, and press
    _Save ROIs_. The ROIs are written to `save_path` as JSON.

    **EXCLUDED AREAS**
    Draw a yellow polygon over anything that should not be segmented.
    Any ROIs that touch those polygons are dropped from consideration.

    **DRAWING**
    - _Delete an ROI_: click to delete (use _Undo_ if it was the wrong one)
    - _Draw_: pick a polygon-draw tool (one for ROIs, one for excluded areas),
    click each corner, double-click to close the shape
    - _Reshape_: pick the polygon-edit tool, double-click a shape, then drag
    its corners

    **PARAMETERS**
    - `image`: signed response map (z-scores), array or path to `.npy`/`.tif`
    - `um_per_px`: size of a pixel, used to read every other parameter in um
    - `save_path`: where to write the ROIs
    - `diameter_um`, `threshold`, `min_area_fraction`, `border_um`: starting
    values for the parameter boxes (see `find_rois`)
    - `color_limit`: value shown as fully blue or fully red
    - `max_preview_px`: the picture is shrunk to this size before being sent to
    the browser (ROIs are still found on the full image)

    **EXAMPLES**
    ```python
    pick_rois("envelope.npy", diameter_um=50, save_path="e1_rois.json")
    ```
    """

    import sys

    import bokeh.plotting as bpl

    from threading import Thread
    from bokeh.io import output_notebook
    from bokeh.io.state import curstate
    from bokeh.models import (
        Button,
        ColumnDataSource,
        Div,
        LinearColorMapper,
        PolyDrawTool,
        PolyEditTool,
        Spinner,
        TapTool,
    )

    if "ipykernel" in sys.modules and not curstate().notebook:
        import os

        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"  # HACK: render inside VSCode
        output_notebook()

    array = load_image(image)
    height, width = array.shape
    save_path = Path(save_path).expanduser().resolve()

    # Shrinking only the *displayed* copy keeps the browser responsive on
    # 2000x2000 images. Coordinates stay in full-resolution pixels because the
    # glyph is stretched back to the original size below.
    step = max(1, max(array.shape) // max_preview_px)
    preview = array[::step, ::step]

    def to_source(polygons: list[Polygon]) -> dict:
        return {
            "xs": [polygon[:, 0].tolist() for polygon in polygons],
            "ys": [polygon[:, 1].tolist() for polygon in polygons],
        }

    def modify_doc(doc):
        # --- Picture --- #

        p = bpl.figure(
            x_range=(0, width),
            y_range=(height, 0),  # row 0 on top, like imshow
            width=800,
            aspect_ratio="auto",
        )
        p.xaxis.visible = p.yaxis.visible = False

        cmap = LinearColorMapper(
            palette=_diverging_palette(), low=-color_limit, high=color_limit
        )
        p.image(image=[preview], x=0, y=0, dw=width, dh=height, color_mapper=cmap)

        # --- ROI layers --- #
        # Two sources so that recomputing replaces the automatic ROIs without
        # touching anything drawn by hand.

        auto_src = ColumnDataSource(data=to_source([]))
        auto_r = p.patches(
            "xs",
            "ys",
            source=auto_src,
            fill_color="white",
            fill_alpha=ROI_FILL_ALPHA,
            line_color="white",
            line_width=2,
            selection_fill_alpha=0.5,
        )

        manual_src = ColumnDataSource(data=to_source([]))
        manual_r = p.patches(
            "xs",
            "ys",
            source=manual_src,
            fill_color="lime",
            fill_alpha=ROI_FILL_ALPHA,
            line_color="lime",
            line_width=2,
            selection_fill_alpha=0.5,
        )

        # --- Excluded areas --- #

        exclude_src = ColumnDataSource(data=to_source([]))
        exclude_r = p.patches(
            "xs",
            "ys",
            source=exclude_src,
            fill_color=EXCLUDE_COLOR,
            fill_alpha=0.25,
            line_color=EXCLUDE_COLOR,
            line_width=2,
        )

        vertex_r = p.scatter(
            "x",
            "y",
            source=ColumnDataSource(data=dict(x=[], y=[])),
            size=10,
            fill_color="lime",
            line_color="black",
        )

        tap = TapTool(renderers=[auto_r, manual_r], description="Delete ROI")
        p.add_tools(
            tap,
            PolyDrawTool(renderers=[manual_r], description="Draw ROI"),
            PolyDrawTool(renderers=[exclude_r], description="Draw excluded area"),
            PolyEditTool(
                renderers=[manual_r, exclude_r],
                vertex_renderer=vertex_r,
                description="Reshape a polygon",
            ),
        )

        # Start on click-to-delete; drawing is picked from the toolbar
        p.toolbar.active_tap = tap

        # --- Widgets --- #

        def spinner(value, title, step, low=0.0):
            return Spinner(low=low, step=step, value=value, title=title, width=130)

        sp_diameter = spinner(diameter_um, "Diameter (µm)", 1.0, low=2 * um_per_px)
        sp_threshold = spinner(threshold, "Min threshold", 0.1)
        sp_area = spinner(min_area_fraction, "Min area (fraction)", 0.05)
        sp_border = spinner(border_um, "Border margin (µm)", 5.0)
        sp_color = spinner(color_limit, "Color limit (z)", 0.5, low=0.1)

        undo = Button(label="Undo delete", button_type="warning")
        clear = Button(label="Clear excluded areas", button_type="warning")
        save = Button(label="Save ROIs", button_type="success")
        status = Div(text="")

        # --- State --- #

        deleted: list[tuple[ColumnDataSource, dict]] = []
        flags = {"busy": False}
        run = {"busy": False, "pending": False}

        def update_status(message: str = "") -> None:
            total = len(auto_src.data["xs"]) + len(manual_src.data["xs"])
            drawn = len(manual_src.data["xs"])
            areas = len(exclude_src.data["xs"])

            detail = f"{drawn} drawn by hand, {areas} excluded areas"
            status.text = f"<b>{total} ROIs</b> ({detail}). {message}"

        # Bokeh rejects callbacks whose arguments have defaults, so the
        # widget callbacks are thin wrappers around the real functions.
        def on_manual(attr, old, new):
            update_status()

        # Keeps the count in sync as ROIs are drawn by hand
        manual_src.on_change("data", on_manual)

        # --- Delete on click --- #

        def on_tap(src):
            def callback(attr, old, new):
                if flags["busy"] or not new:
                    return

                flags["busy"] = True
                try:
                    data = {key: list(values) for key, values in src.data.items()}

                    for index in sorted(new, reverse=True):
                        removed = {
                            key: values.pop(index) for key, values in data.items()
                        }
                        deleted.append((src, removed))

                    src.data = data
                    src.selected.indices = []

                finally:
                    flags["busy"] = False

                update_status()

            return callback

        auto_src.selected.on_change("indices", on_tap(auto_src))
        manual_src.selected.on_change("indices", on_tap(manual_src))

        def undo_callback():
            if not deleted:
                return

            src, removed = deleted.pop()
            src.data = {
                key: list(values) + [removed[key]] for key, values in src.data.items()
            }

            update_status()

        undo.on_click(undo_callback)

        # --- Recompute (off the UI thread, so the app stays responsive) --- #

        def recompute() -> None:
            # Drawing a polygon fires once per corner, so runs are coalesced:
            # anything asked for mid-run turns into a single re-run at the end.
            if run["busy"]:
                run["pending"] = True
                return

            run["busy"] = True

            parameters = dict(
                diameter_um=sp_diameter.value,
                threshold=sp_threshold.value,
                min_area_fraction=sp_area.value,
                border_um=sp_border.value,
            )

            # Read the polygons here, while we are still on the UI thread
            exclude = [
                np.column_stack([xs, ys])
                for xs, ys in zip(exclude_src.data["xs"], exclude_src.data["ys"])
                if len(xs) >= 3
            ]

            status.text = "<i>Looking for ROIs…</i>"

            def work():
                try:
                    polygons = find_rois(
                        array,
                        um_per_px=um_per_px,
                        exclude=exclude,
                        omit_messages=True,
                        **parameters,
                    )
                    data, message = to_source(polygons), ""

                except Exception as error:
                    data, message = to_source([]), f"<b>Failed:</b> {error}"

                def apply():
                    run["busy"] = False

                    auto_src.data = data
                    update_status(message)

                    if run["pending"]:
                        run["pending"] = False
                        recompute()

                doc.add_next_tick_callback(apply)

            Thread(target=work, daemon=True).start()

        def on_parameter(attr, old, new):
            recompute()

        for widget in (sp_diameter, sp_threshold, sp_area, sp_border):
            widget.on_change("value_throttled", on_parameter)

        # Excluded areas change the segmentation itself, so they re-run it
        exclude_src.on_change("data", on_parameter)

        def clear_callback():
            exclude_src.data = to_source([])

        clear.on_click(clear_callback)

        def on_color(attr, old, new):
            cmap.low, cmap.high = -new, new

        sp_color.on_change("value_throttled", on_color)

        # --- Save --- #

        def save_callback():
            rois = []

            for source, src in (("auto", auto_src), ("manual", manual_src)):
                for xs, ys in zip(src.data["xs"], src.data["ys"]):
                    rois.append(
                        {
                            "label": len(rois) + 1,
                            "source": source,
                            "polygon": [[x, y] for x, y in zip(xs, ys)],
                        }
                    )

            payload = {
                "image_shape": [height, width],
                "um_per_px": um_per_px,
                "parameters": {
                    "diameter_um": sp_diameter.value,
                    "threshold": sp_threshold.value,
                    "min_area_fraction": sp_area.value,
                    "border_um": sp_border.value,
                },
                "excluded_areas": [
                    [[x, y] for x, y in zip(xs, ys)]
                    for xs, ys in zip(exclude_src.data["xs"], exclude_src.data["ys"])
                ],
                "rois": rois,
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2))

            update_status(f"Saved {len(rois)} ROIs to <code>{save_path}</code>.")

        save.on_click(save_callback)

        # --- Layout --- #

        doc.add_root(
            bpl.row(
                p,
                bpl.column(
                    sp_diameter,
                    sp_threshold,
                    sp_area,
                    sp_border,
                    sp_color,
                    undo,
                    clear,
                    save,
                    status,
                ),
            )
        )

        recompute()

    bpl.show(modify_doc)


def _diverging_palette(n: int = 256) -> list[str]:
    """Blue -> white -> red palette, as hex colors."""

    def ramp(start, end, count):
        return [
            tuple(round(a + (b - a) * t) for a, b in zip(start, end))
            for t in np.linspace(0, 1, count)
        ]

    colors = ramp(COLOR_LOW, COLOR_MID, n // 2) + ramp(
        COLOR_MID, COLOR_HIGH, n - n // 2
    )

    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
