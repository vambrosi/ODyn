"""
Functions to draw regions and apply masks to images.

Regions are:
    - polygons in the pixel coordinates of an image;
    - can be used to include or exclude pixels via a mask.

The associated mask is `True` only when the pixel is inside an include polygon
but outside every exclude polygon. No include polygons is equivalent to having
an include polygon containing the whole image.
"""

from __future__ import annotations

import numpy as np

from skimage.draw import polygon2mask

from .utils import clamp, Object, logger

INCLUDE = "include"
EXCLUDE = "exclude"
KINDS = (INCLUDE, EXCLUDE)

# Longest side of the drawing area, in screen pixels
FRAME_SIDE_PX = 800

# How near the first corner a click has to be to close the polygon, in screen
# pixels. Screen rather than image pixels because it is what the click looks
# like that decides what it means, and zooming changes the two independently
CLOSE_SCREEN_PX = 10


def region_mask(regions: Object, shape: None | tuple[int, int] = None) -> np.ndarray:
    """
    Turn saved regions into a boolean mask, `True` where a pixel is kept.

    `regions` is what `Group.pick_regions` recorded: the shape it was drawn on
    and a list of `{"kind": ..., "vertices": [[x, y], ...]}`.

    `shape` defaults to the one the regions were drawn on. Passing a different
    one raises rather than stretching: a mask that silently does not line up
    with its image is worse than no mask.
    """
    drawn_on = tuple(regions["shape"])

    if shape is not None and tuple(shape) != drawn_on:
        raise ValueError(
            f"Regions were drawn on a {drawn_on} image, but the mask was asked "
            f"for {tuple(shape)}. Draw them again on an image of that size."
        )

    included = np.zeros(drawn_on, dtype=bool)
    excluded = np.zeros(drawn_on, dtype=bool)
    counts = {INCLUDE: 0, EXCLUDE: 0}

    for region in regions["polygons"]:
        vertices = np.asarray(region["vertices"], dtype=float)

        # Skip bigons (since they don't have an interior)
        if len(vertices) < 3:
            continue

        # Vertices are xy-cordinates due of the drawing tool, but polygon2mask
        # and yx-coordinates, which is how the images are laid out.
        painted = polygon2mask(drawn_on, vertices[:, ::-1])

        if region["kind"] == EXCLUDE:
            excluded |= painted
            counts[EXCLUDE] += 1

        else:
            included |= painted
            counts[INCLUDE] += 1

    # No include polygons means the whole image, so excludes alone still work
    if counts[INCLUDE] == 0:
        included[:] = True

    mask = included & ~excluded

    logger.info(
        f"{counts[INCLUDE]} include and {counts[EXCLUDE]} exclude polygons keep "
        f"{mask.sum()} of {mask.size} pixels ({100 * mask.mean():.1f}%)."
    )

    return mask


def mask_outside(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Copy of `array` with everything outside `mask` set to NaN.

    Works on one image `(H, W)` or a series of them `(..., H, W)`.

    We use NaN rather than zero, because it can be more easily ignored in plots
    and averages (by using `np.nanmean`, for example). Use the mask directly if
    the shape is not relevant (`array[..., mask]`).
    """
    mask = np.asarray(mask, dtype=bool)

    if array.shape[-2:] != mask.shape:
        raise ValueError(
            f"Mask is {mask.shape} but the last two axes of the "
            f"array are {array.shape[-2:]}."
        )

    # float, because NaN cannot be stored in an integer array
    out = array.astype(np.result_type(array.dtype, np.float32), copy=True)
    out[..., ~mask] = np.nan

    return out


def pick_regions(image: np.ndarray, on_save) -> None:
    """
    Draw include and exclude polygons over `image`, in a notebook.

    `on_save` is given the list of polygons when the Save is pressed.

    Clicking puts a corner down and every other step is a labelled button.
    Bokeh's own `PolyDrawTool` is not used: it is driven by press-and-hold
    gestures that its documentation gets wrong, and buttons are both easier
    to explain and unaffected by the next release.
    """

    # We use buttons directly because Bokeh PolyDrawTool is not ideal.
    #   - Keyboard shortcuts cannot be used reliably inside VSCode;
    #   - Hold to start/stop is not very intuitive (and not documented);
    #   - It feels less resposive in general.
    #
    # (Also, some JS/Python duplication is forced in `pick_mcor_parameters`.)
    #
    # TODO: Change to some other more reliable GUI.

    import sys

    import bokeh.plotting as bpl

    from bokeh.events import Tap
    from bokeh.io import output_notebook
    from bokeh.io.state import curstate
    from bokeh.layouts import column, row
    from bokeh.models import (
        Button,
        ColumnDataSource,
        Div,
        LinearColorMapper,
        RadioButtonGroup,
    )
    from bokeh.palettes import Greys256

    if "ipykernel" in sys.modules and not curstate().notebook:
        import os

        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"  # HACK: render inside VSCode
        output_notebook()

    image = np.asarray(image, dtype=float)
    height, width = image.shape

    def modify_doc(doc):
        # Cut the tails to avoid flattening the image
        # NaN is allowed to be compatible with masked images
        finite = image[np.isfinite(image)]
        low, high = (
            (float(np.percentile(finite, 1)), float(np.percentile(finite, 99)))
            if finite.size
            else (0.0, 1.0)
        )

        # Scale needs to be handed explicitly, because Bokeh aspect_ratio is
        # not respected when the surrounding layout is not free to resize.
        scale = FRAME_SIDE_PX / max(height, width)

        # y downwards to agree with other plots.
        figure = bpl.figure(
            x_range=(0, width),
            y_range=(height, 0),
            frame_height=round(height * scale),
            frame_width=round(width * scale),
            match_aspect=True,
            tools="pan,wheel_zoom,reset",
        )

        figure.xaxis.visible = figure.yaxis.visible = False
        figure.image(
            image=[image],
            x=0,
            y=0,
            dw=width,
            dh=height,
            color_mapper=LinearColorMapper(palette=Greys256, low=low, high=high),
            level="image",
        )

        colors = {INCLUDE: "#2ca02c", EXCLUDE: "#d62728"}

        # Everything drawn so far, in the order it was drawn, and the corners
        # of the one in progress. Both are mutated in place, and the two data
        # sources below are redrawn from them, so there is one source of truth
        polygons: list[Object] = []
        corners: list[list[float]] = []
        picked = {"kind": INCLUDE}

        finished = {kind: ColumnDataSource(dict(xs=[], ys=[])) for kind in KINDS}
        drawing = ColumnDataSource(dict(x=[], y=[]))

        for kind in KINDS:
            figure.patches(
                xs="xs",
                ys="ys",
                source=finished[kind],
                fill_alpha=0.2,
                fill_color=colors[kind],
                line_color=colors[kind],
                line_width=2,
            )

        # The polygon being drawn is left open, to show it is unfinished
        trail = figure.line(
            x="x",
            y="y",
            source=drawing,
            line_color=colors[INCLUDE],
            line_width=2,
            line_dash="dashed",
        )

        marks = figure.scatter(
            x="x",
            y="y",
            source=drawing,
            size=8,
            fill_color="white",
            line_color=colors[INCLUDE],
            line_width=2,
        )

        status = Div()
        kind_picker = RadioButtonGroup(
            labels=["Include (green)", "Exclude (red)"], active=0
        )
        finish = Button(label="Close polygon", button_type="primary")
        undo = Button(label="Undo corner")
        remove = Button(label="Delete last polygon")
        save = Button(label="Save regions", button_type="success")

        def refresh():
            for kind in KINDS:
                drawn = [p for p in polygons if p["kind"] == kind]
                finished[kind].data = dict(
                    xs=[[x for x, _ in p["vertices"]] for p in drawn],
                    ys=[[y for _, y in p["vertices"]] for p in drawn],
                )

            drawing.data = dict(x=[x for x, _ in corners], y=[y for _, y in corners])

            counts = {k: sum(p["kind"] == k for p in polygons) for k in KINDS}
            status.text = (
                f"{counts[INCLUDE]} included, {counts[EXCLUDE]} excluded, and "
                f"{len(corners)} corners in the current region (3 to finish)."
            )

        def near_first(x, y):
            """Whether a click is within `CLOSE_SCREEN_PX` of the first corner."""
            if len(corners) < 3:
                return False

            # Image pixels per screen pixel, read from the ranges rather than
            # from 'scale', so that zooming in does not widen the snap
            per_x = abs(figure.x_range.end - figure.x_range.start) / figure.frame_width
            per_y = abs(figure.y_range.end - figure.y_range.start) / figure.frame_height

            first_x, first_y = corners[0]

            return (
                abs(x - first_x) <= CLOSE_SCREEN_PX * per_x
                and abs(y - first_y) <= CLOSE_SCREEN_PX * per_y
            )

        def on_tap(event):
            # Close the polygon if the click lands back on the first corner.
            # Only once there is a polygon to close, so that an early click
            # near the start drops a corner instead of doing nothing at all
            if near_first(event.x, event.y):
                on_finish()

            # Clamped, so a corner meant for the edge of the picture lands on it
            else:
                corners.append(
                    [
                        clamp(float(event.x), 0.0, float(width)),
                        clamp(float(event.y), 0.0, float(height)),
                    ]
                )
                refresh()

        def on_kind(attr, old, new):
            picked["kind"] = KINDS[new]
            trail.glyph.line_color = colors[picked["kind"]]
            marks.glyph.line_color = colors[picked["kind"]]

        # A handler with no arguments is called with none: bokeh counts the
        # parameters that have no default and dispatches on that
        def on_finish():
            if len(corners) < 3:
                return

            polygons.append(
                {
                    "kind": picked["kind"],
                    "vertices": list(corners),
                }
            )
            corners.clear()
            refresh()

        def on_undo():
            if corners:
                corners.pop()
                refresh()

        def on_remove():
            # The half-finished one first, to keep the order.
            if corners:
                corners.clear()

            elif polygons:
                polygons.pop()

            refresh()

        def on_save_click():
            on_save([dict(p) for p in polygons])
            counts = {k: sum(p["kind"] == k for p in polygons) for k in KINDS}
            status.text = (
                f"Saved {counts[INCLUDE]} to keep and {counts[EXCLUDE]} to drop."
            )

        figure.on_event(Tap, on_tap)
        kind_picker.on_change("active", on_kind)
        finish.on_click(on_finish)
        undo.on_click(on_undo)
        remove.on_click(on_remove)
        save.on_click(on_save_click)

        refresh()
        doc.add_root(
            column(
                Div(
                    text=(
                        "<b>Click</b> to start drawing. Click the first corner "
                        "again to close the region."
                    )
                ),
                row(kind_picker, undo, remove, finish, save),
                status,
                figure,
            )
        )

    bpl.show(modify_doc)
