# --------------------------------------------------------------------------- #
#
# STATUS: standalone prototype, i.e. no DB interaction. `pick_rois` writes
# a JSON file that maps onto a future `rois` table (one row per polygon).
#
# TODO:
#   - Add exclusion polygons instead of deleting one by one
#   - Use database mcor file paths directly
#   - Reload the last saved ROIs when the GUI opens
#
# --------------------------------------------------------------------------- #

"""
Quick segmentation of glomeruli from an image (e.g. z-scores).
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

# Defaults shared by find_rois and the GUI.
DEFAULT_DIAMETER_UM = 60.0
DEFAULT_THRESHOLD = 2.0
DEFAULT_MIN_AREA_FRACTION = 0.2
DEFAULT_BORDER_UM = 0.0

# The excluded strip is off (0 µm tall) by default.
# The goal is to cover the midline sinus.
DEFAULT_STRIP_HEIGHT_UM = 0.0

# Drawing parameters
#   - ROIs excluded by the strip will have the EXCLUDED_COLOR.
#   - STRIP_COLOR is the color of the overlaid strip.
ROI_FILL_ALPHA = 0.15
EXCLUDED_COLOR = "#909090"
STRIP_COLOR = "#cccc00"

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

    # Difference of Gaussians tuned to the ROI scale.
    #
    # Goals:
    #   - Removes small-scale noise;
    #   - Gives peaks near ROI centers.

    sigma = radius_px / np.sqrt(2)
    smooth = gaussian_filter(response, sigma)
    dog = smooth - gaussian_filter(response, sigma * DOG_RATIO)

    mask = smooth > threshold

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

    if not omit_messages:
        logger.info(f"Found {len(polygons)} ROIs.")

    return [polygon for _centroid, polygon in polygons]


# --------------------------------------------------------------------------- #
# Test Images
# --------------------------------------------------------------------------- #


def test_images(
    *,
    shape: tuple[int, int] = (600, 600),
    um_per_px: float = 1.0,
    diameter_um: float = 45.0,
    spacing_um: float | None = None,
    overlap_fraction: float = 0.4,
    negative_fraction: float = 0.25,
    gap_um: float = 120.0,
    gap_noise: float = 2.5,
    noise: float = 0.8,
    noise_um: float = 4.0,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Generated image, to try the GUI out without opening real data.

    Draws a grid of jittered ovals of mixed sign, some of them overlapping,
    with an empty noisy band across the middle standing in for the midline
    sinus (for testing the excluded strip).

    **PARAMETERS**
    - `shape`, `um_per_px`: size of the picture
    - `diameter_um`: typical glomerulus size
    - `spacing_um`: distance between neighbors (2.5 diameters by default)
    - `overlap_fraction`: how often a glomerulus gets a partner on top of it
    - `negative_fraction`: how often a glomerulus responds downwards
    - `gap_um`: height of the empty band in the middle (0 removes it)
    - `gap_noise`: how much extra noise lives inside that band. The default is
    strong enough to produce junk ROIs there on purpose, which is what the
    excluded strip is for; set it to 0 for a clean picture
    - `noise`, `noise_um`: strength and grain size of the background noise
    - `seed`: change it for a different picture (`None` for a random one)

    **EXAMPLES**
    ```python
    pick_rois(test_images(), diameter_um=45)
    ```
    """

    rng = np.random.default_rng(seed)
    height, width = shape

    radius_px = 0.5 * diameter_um / um_per_px
    spacing_px = (spacing_um or 2.5 * diameter_um) / um_per_px
    gap_px = gap_um / um_per_px

    image = np.zeros(shape, dtype="float32")

    # Grid of centers, jittered, with the middle band left empty
    margin = 1.5 * radius_px
    rows = np.arange(margin, height - margin, spacing_px)
    columns = np.arange(margin, width - margin, spacing_px)

    centers = []
    for row in rows:
        for column in columns:
            centers.append((row, column))

            # A partner close enough to touch, to exercise the splitting
            if rng.random() < overlap_fraction:
                angle = rng.uniform(0, 2 * np.pi)
                distance = rng.uniform(1.1, 1.6) * radius_px
                centers.append(
                    (row + distance * np.sin(angle), column + distance * np.cos(angle))
                )

    for center_row, center_column in centers:
        # Jitter, and skip anything that falls inside the gap
        center_row += rng.uniform(-0.2, 0.2) * spacing_px
        center_column += rng.uniform(-0.2, 0.2) * spacing_px

        if abs(center_row - height / 2) < gap_px / 2 + radius_px:
            continue

        # Ovals: two different semi-axes, at a random angle
        semi_major = radius_px * rng.uniform(0.8, 1.2)
        semi_minor = semi_major * rng.uniform(0.6, 0.9)
        angle = rng.uniform(0, np.pi)

        sign = -1.0 if rng.random() < negative_fraction else 1.0
        amplitude = sign * rng.uniform(3.0, 9.0)

        # Only evaluate near the center, so big images stay quick
        reach = int(np.ceil(2.5 * semi_major))
        row0 = max(0, int(center_row) - reach)
        row1 = min(height, int(center_row) + reach + 1)
        column0 = max(0, int(center_column) - reach)
        column1 = min(width, int(center_column) + reach + 1)

        if row0 >= row1 or column0 >= column1:
            continue

        rows_px, columns_px = np.mgrid[row0:row1, column0:column1]
        delta_row = rows_px - center_row
        delta_column = columns_px - center_column

        along = delta_column * np.cos(angle) + delta_row * np.sin(angle)
        across = -delta_column * np.sin(angle) + delta_row * np.cos(angle)

        image[row0:row1, column0:column1] += amplitude * np.exp(
            -((along / semi_major) ** 2 + (across / semi_minor) ** 2)
        )

    # Real z-score maps are grainy rather than pixel-noisy, so the noise is
    # smoothed and then scaled back up to the strength that was asked for.
    def grainy(strength: float) -> np.ndarray:
        sigma = max(noise_um / um_per_px, 1e-6)
        field = gaussian_filter(rng.normal(0, 1, shape).astype("float32"), sigma)
        return strength * field / (field.std() or 1.0)

    image += grainy(noise)

    # The gap between the bulbs stays wobbly after motion correction
    if gap_px and gap_noise:
        band = slice(int((height - gap_px) / 2), int((height + gap_px) / 2))
        image[band] += grainy(gap_noise)[band]

    return image


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
    strip_center_um: float | None = None,
    strip_height_um: float = DEFAULT_STRIP_HEIGHT_UM,
    color_limit: float = 5.0,
    max_preview_px: int = 1024,
) -> None:
    """
    Open a GUI to fine tune the segmentation algorithm.

    Change the parameters until most ROIs are outlined, click the ones
    that came out wrong to delete them, draw any that are missing, and press
    _Save ROIs_. The ROIs are written to `save_path` as JSON.

    **EXCLUDED STRIP**
    The yellow band covers the midline sinus. ROIs touching it turn grey
    and are left out of the saved file.

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
    - `diameter_um`, `threshold`, `min_area_fraction`, `border_um`: starting
    values for the sliders (see `find_rois`)
    - `strip_center_um`: where the excluded strip starts (middle of the image
    by default)
    - `strip_height_um`: how tall the excluded strip starts (0 turns it off)
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
        Slider,
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

    def to_source(polygons: list[Polygon], color: str) -> dict:
        # `excluded`, `line_color` and `fill_alpha` are per-ROI so that the
        # excluded strip can grey ROIs out without removing them.
        return {
            "xs": [polygon[:, 0].tolist() for polygon in polygons],
            "ys": [polygon[:, 1].tolist() for polygon in polygons],
            "excluded": [False] * len(polygons),
            "line_color": [color] * len(polygons),
            "fill_alpha": [ROI_FILL_ALPHA] * len(polygons),
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

        # --- Excluded strip --- #
        # Spans the full width; only its vertical extent is adjustable.

        strip_src = ColumnDataSource(data=dict(top=[0.0], bottom=[0.0]))
        p.quad(
            left=0,
            right=width,
            top="top",
            bottom="bottom",
            source=strip_src,
            fill_color=STRIP_COLOR,
            fill_alpha=0.25,
            line_alpha=0,
        )

        # --- ROI layers --- #
        # Two sources so that recomputing replaces the automatic ROIs without
        # touching anything drawn by hand.

        auto_src = ColumnDataSource(data=to_source([], "white"))
        auto_r = p.patches(
            "xs",
            "ys",
            source=auto_src,
            fill_color="white",
            fill_alpha="fill_alpha",
            line_color="line_color",
            line_width=2,
            selection_fill_alpha=0.5,
        )

        manual_src = ColumnDataSource(data=to_source([], "lime"))
        manual_r = p.patches(
            "xs",
            "ys",
            source=manual_src,
            fill_color="lime",
            fill_alpha="fill_alpha",
            line_color="line_color",
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
        sp_threshold = spinner(threshold, "Min threshold", 0.1)
        sp_area = spinner(min_area_fraction, "Min area (fraction)", 0.05)
        sp_border = spinner(border_um, "Border margin (µm)", 5.0)
        sp_color = spinner(color_limit, "Color limit (z)", 0.5, low=0.1)

        image_height_um = height * um_per_px
        center = image_height_um / 2 if strip_center_um is None else strip_center_um

        sl_center = Slider(
            start=0,
            end=image_height_um,
            value=min(max(center, 0.0), image_height_um),
            step=um_per_px,
            title="Excluded strip center (µm)",
            width=260,
        )
        sl_height = Slider(
            start=0,
            end=image_height_um / 2,
            value=min(max(strip_height_um, 0.0), image_height_um / 2),
            step=um_per_px,
            title="Excluded strip height (µm)",
            width=260,
        )

        undo = Button(label="Undo delete", button_type="warning")
        save = Button(label="Save ROIs", button_type="success")
        status = Div(text="")

        # --- State --- #

        deleted: list[tuple[ColumnDataSource, dict]] = []
        flags = {"busy": False}
        run = {"id": 0}

        def count_excluded() -> int:
            return sum(
                sum(bool(flag) for flag in src.data["excluded"])
                for src in (auto_src, manual_src)
            )

        def update_status(message: str = "") -> None:
            total = len(auto_src.data["xs"]) + len(manual_src.data["xs"])
            excluded = count_excluded()
            drawn = len(manual_src.data["xs"])

            kept = f"<b>{total - excluded} ROIs</b>"
            detail = f"{drawn} drawn by hand, {excluded} inside the strip"
            status.text = f"{kept} ({detail}). {message}"

        # --- Excluded strip --- #
        # Nothing is deleted here: ROIs are only flagged and greyed out, so
        # moving the strip brings them back. Saving is what makes it stick.

        def apply_strip() -> None:
            # Re-entrant: writing to a source re-triggers the data callback.
            if flags["busy"]:
                return

            half = 0.5 * sl_height.value / um_per_px
            middle = sl_center.value / um_per_px
            top, bottom = middle - half, middle + half

            flags["busy"] = True
            try:
                strip_src.data = dict(top=[top], bottom=[bottom])

                for src, color in ((auto_src, "white"), (manual_src, "lime")):
                    data = {key: list(values) for key, values in src.data.items()}

                    # The strip spans the full width, so touching it is purely
                    # a question of the ROI's vertical extent.
                    data["excluded"] = [
                        bool(half) and min(ys) <= bottom and max(ys) >= top
                        for ys in data["ys"]
                    ]
                    data["line_color"] = [
                        EXCLUDED_COLOR if flag else color for flag in data["excluded"]
                    ]
                    data["fill_alpha"] = [
                        0.0 if flag else ROI_FILL_ALPHA for flag in data["excluded"]
                    ]

                    src.data = data

            finally:
                flags["busy"] = False

            update_status()

        # Bokeh rejects callbacks whose arguments have defaults, so the
        # widget callbacks are thin wrappers around the real functions.
        def on_strip(attr, old, new):
            apply_strip()

        sl_center.on_change("value", on_strip)
        sl_height.on_change("value", on_strip)

        # Keeps hand-drawn ROIs in sync as they are added or reshaped
        manual_src.on_change("data", on_strip)

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

            apply_strip()  # the restored ROI may land inside the strip

        undo.on_click(undo_callback)

        # --- Recompute (off the UI thread, so the app stays responsive) --- #

        def recompute() -> None:
            run["id"] += 1
            run_id = run["id"]

            parameters = dict(
                diameter_um=sp_diameter.value,
                threshold=sp_threshold.value,
                min_area_fraction=sp_area.value,
                border_um=sp_border.value,
            )

            status.text = "<i>Looking for glomeruli…</i>"

            def work():
                try:
                    polygons = find_rois(
                        array, um_per_px=um_per_px, omit_messages=True, **parameters
                    )
                    data, message = to_source(polygons, "white"), ""

                except Exception as error:
                    data, message = to_source([], "white"), f"<b>Failed:</b> {error}"

                def apply():
                    # A newer run started while this one was still going
                    if run_id != run["id"]:
                        return

                    auto_src.data = data
                    apply_strip()  # re-flag the new ROIs against the strip
                    update_status(message)

                doc.add_next_tick_callback(apply)

            Thread(target=work, daemon=True).start()

        def on_parameter(attr, old, new):
            recompute()

        for widget in (sp_diameter, sp_threshold, sp_area, sp_border):
            widget.on_change("value_throttled", on_parameter)

        def on_color(attr, old, new):
            cmap.low, cmap.high = -new, new

        sp_color.on_change("value_throttled", on_color)

        # --- Save --- #

        def save_callback():
            rois = []

            for source, src in (("auto", auto_src), ("manual", manual_src)):
                data = src.data
                for xs, ys, excluded in zip(data["xs"], data["ys"], data["excluded"]):
                    # This is where the strip stops being a preview
                    if excluded:
                        continue

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
                    "strip_center_um": sl_center.value,
                    "strip_height_um": sl_height.value,
                },
                "rois": rois,
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2))

            update_status(
                f"Saved {len(rois)} ROIs to <code>{save_path}</code> "
                f"({count_excluded()} left out)."
            )

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
                    sl_center,
                    sl_height,
                    undo,
                    save,
                    status,
                ),
            )
        )

        apply_strip()
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
