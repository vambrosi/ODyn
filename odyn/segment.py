# --------------------------------------------------------------------------- #
#
# STATUS: standalone prototype. It does not touch the database yet, so it can
# be thrown away or replaced by a segmentation package without affecting the
# rest of ODyn. `pick_rois` writes a JSON file whose shape maps 1-to-1 onto a
# future `rois` table (one row per polygon).
#
# TODO:
#   - Exclusion polygons (blood vessels, edges) instead of deleting one by one
#   - Feed the response map straight from the database instead of a file
#   - Reload the last saved ROIs when the GUI opens
#
# --------------------------------------------------------------------------- #

"""
Quick segmentation of glomeruli from a response map (e.g. a z-score image).
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from scipy.ndimage import gaussian_filter
from skimage.measure import approximate_polygon, find_contours, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .utils import logger

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Defaults shared by find_rois and the GUI so the two never drift.
DEFAULT_DIAMETER_UM = 60.0
DEFAULT_MIN_ZSCORE = 2.0
DEFAULT_MIN_AREA_FRACTION = 0.2
DEFAULT_BORDER_UM = 0.0

# Ratio between the two Gaussians of the difference-of-Gaussians filter.
# 1.6 is the usual approximation to a Laplacian-of-Gaussian.
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
    min_zscore: float = DEFAULT_MIN_ZSCORE,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    border_um: float = DEFAULT_BORDER_UM,
) -> list[Polygon]:
    """
    Find glomeruli in a response map and return them as polygons.

    **PARAMETERS**
    - `image`: signed response map (z-scores), array or path to `.npy`/`.tif`
    - `um_per_px`: size of a pixel, used to read every other parameter in um
    - `diameter_um`: typical glomerulus diameter (the main knob)
    - `min_zscore`: how strong a response has to be to count
    - `min_area_fraction`: drop ROIs smaller than this fraction of a circle of
    `diameter_um` (0 keeps everything, 1 keeps only full-sized glomeruli)
    - `border_um`: drop ROIs that come within this distance of the image edge

    **EXAMPLES**
    ```python
    rois = find_rois("envelope.npy", diameter_um=50, min_zscore=2.5)
    ```
    """

    array = load_image(image)
    radius_px = 0.5 * diameter_um / um_per_px

    if radius_px < 1:
        raise ValueError("'diameter_um' is smaller than two pixels.")

    # Glomeruli can respond in either direction, so we look for blobs in the
    # magnitude of the response and keep the sign only for display.
    response = np.abs(array)

    # Difference of Gaussians tuned to the glomerular scale. This does the job
    # of the old blur slider (removing pixel noise) *and* gives us a surface
    # that peaks at the center of each glomerulus and dips between neighbors.
    sigma = radius_px / np.sqrt(2)
    smooth = gaussian_filter(response, sigma)
    dog = smooth - gaussian_filter(response, sigma * DOG_RATIO)

    mask = smooth > min_zscore

    if not mask.any():
        logger.warning(f"Nothing above min_zscore = {min_zscore}.")
        return []

    # One seed per glomerulus: peaks closer than a radius are the same blob.
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

    # Splits touching glomeruli along the dim valley between their centers.
    labels = watershed(-dog, markers, mask=mask)

    # --- Filter and convert each region to a polygon --- #

    min_area_px = min_area_fraction * np.pi * radius_px**2
    border_px = border_um / um_per_px
    height, width = labels.shape

    # Enough to drop the staircase of single pixels without losing the shape
    tolerance_px = max(1.0, 0.05 * radius_px)

    polygons = []

    for region in regionprops(labels):
        if region.area < min_area_px:
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

    logger.info(f"Found {len(polygons)} ROIs.")

    return [polygon for _centroid, polygon in polygons]


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #


def pick_rois(
    image: str | Path | np.ndarray,
    *,
    um_per_px: float = 1.0,
    save_path: str | Path = "rois.json",
    diameter_um: float = DEFAULT_DIAMETER_UM,
    min_zscore: float = DEFAULT_MIN_ZSCORE,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    border_um: float = DEFAULT_BORDER_UM,
    color_limit: float = 5.0,
    max_preview_px: int = 1024,
) -> None:
    """
    Open a GUI to segment glomeruli and fix the result by hand.

    Change the parameters until most glomeruli are outlined, click the ones
    that came out wrong to delete them, draw any that are missing, and press
    _Save ROIs_. The ROIs are written to `save_path` as JSON.

    **DRAWING**
    - _Delete_: click an ROI (use _Undo_ if it was the wrong one)
    - _Draw_: pick the polygon-draw tool, click each corner, double-click to
    close the shape
    - _Reshape_: pick the polygon-edit tool, double-click an ROI, then drag
    its corners

    **PARAMETERS**
    - `image`: signed response map (z-scores), array or path to `.npy`/`.tif`
    - `um_per_px`: size of a pixel, used to read every other parameter in um
    - `save_path`: where to write the ROIs
    - `diameter_um`, `min_zscore`, `min_area_fraction`, `border_um`: starting
    values for the sliders (see `find_rois`)
    - `color_limit`: z-score shown as fully blue or fully red
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
            height=800,
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

        auto_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        auto_r = p.patches(
            "xs",
            "ys",
            source=auto_src,
            fill_color="white",
            fill_alpha=0.15,
            line_color="white",
            line_width=2,
            selection_fill_alpha=0.5,
        )

        manual_src = ColumnDataSource(data=dict(xs=[], ys=[]))
        manual_r = p.patches(
            "xs",
            "ys",
            source=manual_src,
            fill_color="lime",
            fill_alpha=0.15,
            line_color="lime",
            line_width=2,
            selection_fill_alpha=0.5,
        )

        vertex_r = p.scatter(
            "x",
            "y",
            source=ColumnDataSource(data=dict(x=[], y=[])),
            size=10,
            fill_color="lime",
            line_color="black",
        )

        tap = TapTool(renderers=[auto_r, manual_r])
        p.add_tools(
            tap,
            PolyDrawTool(renderers=[manual_r]),
            PolyEditTool(renderers=[manual_r], vertex_renderer=vertex_r),
        )

        # Start on click-to-delete; drawing is picked from the toolbar
        p.toolbar.active_tap = tap

        # --- Widgets --- #

        def spinner(value, title, step, low=0.0):
            return Spinner(low=low, step=step, value=value, title=title, width=130)

        sp_diameter = spinner(diameter_um, "Diameter (µm)", 1.0, low=2 * um_per_px)
        sp_zscore = spinner(min_zscore, "Min z-score", 0.1)
        sp_area = spinner(min_area_fraction, "Min area (fraction)", 0.05)
        sp_border = spinner(border_um, "Border margin (µm)", 5.0)
        sp_color = spinner(color_limit, "Color limit (z)", 0.5, low=0.1)

        undo = Button(label="Undo delete", button_type="warning")
        save = Button(label="Save ROIs", button_type="success")
        status = Div(text="")

        # --- State --- #

        deleted: list[tuple[ColumnDataSource, list, list]] = []
        flags = {"busy": False}
        run = {"id": 0}

        def update_status(message: str = "") -> None:
            n_auto = len(auto_src.data["xs"])
            n_manual = len(manual_src.data["xs"])
            status.text = f"<b>{n_auto + n_manual} ROIs</b> ({n_manual} drawn by hand). {message}"

        # --- Delete on click --- #

        def on_tap(src):
            def callback(attr, old, new):
                if flags["busy"] or not new:
                    return

                flags["busy"] = True
                try:
                    xs, ys = list(src.data["xs"]), list(src.data["ys"])
                    for index in sorted(new, reverse=True):
                        deleted.append((src, xs.pop(index), ys.pop(index)))

                    src.data = dict(xs=xs, ys=ys)
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

            src, xs, ys = deleted.pop()
            src.data = dict(
                xs=list(src.data["xs"]) + [xs], ys=list(src.data["ys"]) + [ys]
            )
            update_status()

        undo.on_click(undo_callback)

        # --- Recompute (off the UI thread, so the app stays responsive) --- #

        def recompute(attr=None, old=None, new=None):
            run["id"] += 1
            run_id = run["id"]

            parameters = dict(
                diameter_um=sp_diameter.value,
                min_zscore=sp_zscore.value,
                min_area_fraction=sp_area.value,
                border_um=sp_border.value,
            )

            status.text = "<i>Looking for glomeruli…</i>"

            def work():
                try:
                    polygons = find_rois(array, um_per_px=um_per_px, **parameters)
                    data, message = to_source(polygons), ""

                except Exception as error:
                    data, message = dict(xs=[], ys=[]), f"<b>Failed:</b> {error}"

                def apply():
                    # A newer run started while this one was still going
                    if run_id != run["id"]:
                        return

                    auto_src.data = data
                    update_status(message)

                doc.add_next_tick_callback(apply)

            Thread(target=work, daemon=True).start()

        for widget in (sp_diameter, sp_zscore, sp_area, sp_border):
            widget.on_change("value_throttled", recompute)

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
                    "min_zscore": sp_zscore.value,
                    "min_area_fraction": sp_area.value,
                    "border_um": sp_border.value,
                },
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
                    sp_zscore,
                    sp_area,
                    sp_border,
                    sp_color,
                    undo,
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

    colors = ramp(COLOR_LOW, COLOR_MID, n // 2) + ramp(COLOR_MID, COLOR_HIGH, n - n // 2)

    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
