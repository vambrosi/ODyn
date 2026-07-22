# --------------------------------------------------------------------------- #
#
# TODO:
#   - TEST PLAY MOVIE!
#   - Change save directory default?
#   - Make list of files in play_movie optional?
#   - Add movies to outputs
#   - Add support for files (outputs) not in server (computer prefix?)
#   - Integrate docstrings with default values, to avoid copy-paste.
#   - Remove added stuff in docstrings
#   - Use validated defaults on docstrings?
#   - Add help for expected file name and folder structure?
#   - Add delete_temp_files reminder in run_motion_correction
#   - Update raw_mmap_pairs when did final mcor?
#
# MAYBE TODO:
#   - Add pipeline function/classes to facilitate data analysis:
#       - Convolution layers (moving weighted averages)
#       - Thresholding (entrywise biased Heaviside or ReLU functions)
#       - All the steps in the MATLAB segmentation GUI?
#
# --------------------------------------------------------------------------- #

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import cast, Final, TYPE_CHECKING

from pathlib import Path

import pandas as pd
import tifffile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .utils import *
from .utils import _method_calls_dataframe
from .utils import CallFrame

if TYPE_CHECKING:
    from .database import Database

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Default motion-correction parameters (in um). Single source shared by
# run_motion_correction and pick_mcor_parameters so the two never drift.
DEFAULT_STRIDES_UM = [128.0, 128.0]
DEFAULT_OVERLAP_UM = [96.0, 96.0]
DEFAULT_MAX_SHIFT_UM = [128.0, 128.0]
DEFAULT_MAX_DEVIATION_UM = 12.0


class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"


# --------------------------------------------------------------------------- #
# Main Data Processing\Analysis Class
# --------------------------------------------------------------------------- #


class Group:
    """
    Class that runs data processing/analysis.

    **USAGE**
    ```python
    db    = Database(main_folder)
    group = db.groups[some_index]
    ```

    **RELEVANT PROPERTIES**
    ```python
        group.acquisitions      # `DataFrame` with acquisition metadata
        group.events            # `DataFrame` with olfactometer events
        group.experiments       # `DataFrame` with experiment metadata
        group.mcor_files        # `DataFrame` with mcor files metadata
        group.method_calls      # `DataFrame` with `@record_call` functions
        group.programs          # `DataFrame` with one entry per _Event.csv_ file
        group.trials            # `DataFrame` with all olfactometer trials
    ```

    **RELEVANT METHODS**
    ```python
        group.latest_calls(method_name)

        group.pick_mcor_parameters(...)
        group.run_motion_correction(...)

        group.delete_temp_files()
        group.play_movie(...)
    ```
    """

    def __init__(self, group_id: int, db: Database) -> None:
        self.group_id: Final[int] = group_id
        self.db = db

        # Initialize "private" variables
        self._acquisitions: None | pd.DataFrame = None
        self._events: None | pd.DataFrame = None
        self._experiments: None | pd.DataFrame = None
        self._mcor_files: None | pd.DataFrame = None
        self._method_calls: None | pd.DataFrame = None
        self._outputs: None | pd.DataFrame = None
        self._programs: None | pd.DataFrame = None
        self._trials: None | pd.DataFrame = None

        self._call_stack: list[CallFrame] = []
        self._raw_mmap_pairs: None | tuple[list[str], list[str]] = None
        self.movies: dict[tuple[MovieType, ...], LazyMovie] = {}

    def __repr__(self):
        msg = f"Group {self.group_id}"

        # Show the experiment name if there is only one
        if len(self.experiments) == 1:
            msg += f" (exp_name = {self.experiments["exp_name"].iloc[0]})"

        return msg

    @property
    def main_folder(self):
        return self.db.main_folder

    # ----------------------------------------------------------------------- #
    # SQLite Tables as DataFrames
    # ----------------------------------------------------------------------- #

    @property
    def acquisitions(self) -> pd.DataFrame:
        """`DataFrame` with acquisition metadata"""
        if self._acquisitions is not None:
            return self._acquisitions

        query = f"""
            SELECT a.* FROM group_experiments AS g
                JOIN experiments  AS e ON e.exp_id = g.exp_id
                JOIN acquisitions AS a ON a.exp_id = e.exp_id
                WHERE g.group_id = {self.group_id};
        """

        self._acquisitions = pd.read_sql_query(
            query, self.db.con, parse_dates=["acq_start", "odor_start", "odor_end"]
        )
        self._acquisitions.set_index("acq_id", inplace=True)

        return self._acquisitions

    @property
    def events(self) -> pd.DataFrame:
        """`DataFrame` with olfactometer events"""
        if self._events is not None:
            return self._events

        query = f"""
            SELECT e.* FROM group_experiments AS g
                JOIN experiments AS x ON x.exp_id     = g.exp_id
                JOIN programs    AS p ON p.exp_id     = x.exp_id
                JOIN events      AS e ON e.program_id = p.program_id
                WHERE g.group_id = {self.group_id};
        """

        self._events = pd.read_sql_query(query, self.db.con)
        self._events.set_index("event_id", inplace=True)

        return self._events

    @property
    def experiments(self) -> pd.DataFrame:
        """`DataFrame` with experiment metadata"""
        if self._experiments is not None:
            return self._experiments

        query = f"""
            SELECT e.* FROM group_experiments AS ge
                JOIN experiments AS e ON e.exp_id = ge.exp_id
                WHERE ge.group_id = {self.group_id};
        """
        self._experiments = pd.read_sql_query(
            query, self.db.con, parse_dates=["exp_start", "added_to_db_at"]
        )
        self._experiments.set_index("exp_id", inplace=True)

        return self._experiments

    @property
    def mcor_files(self) -> pd.DataFrame:
        """`DataFrame` with mcor files metadata"""
        if self._mcor_files is not None:
            return self._mcor_files

        query = f"""
            SELECT m.* FROM group_experiments AS g
                JOIN experiments  AS e ON e.exp_id = g.exp_id
                JOIN acquisitions AS a ON a.exp_id = e.exp_id
                JOIN mcor_files   AS m ON m.acq_id = a.acq_id
                WHERE g.group_id = {self.group_id};
        """

        self._mcor_files = pd.read_sql_query(query, self.db.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def method_calls(self) -> pd.DataFrame:
        """`DataFrame` with `@record_call` functions"""
        if self._method_calls is not None:
            return self._method_calls

        query = f"SELECT * FROM method_calls WHERE group_id = {self.group_id};"

        self._method_calls = pd.read_sql_query(
            query, self.db.con, parse_dates=["called_at"]
        )
        self._method_calls.set_index("method_call_id", inplace=True)

        self._method_calls["parameter_inputs"] = self._method_calls[
            "parameter_inputs"
        ].apply(json.loads)
        self._method_calls["parameters_used"] = self._method_calls[
            "parameters_used"
        ].apply(json.loads)

        self._method_calls["call_output"] = self._method_calls["call_output"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else None
        )

        return self._method_calls

    @property
    def outputs(self) -> pd.DataFrame:
        """`DataFrame` with output files of functions"""
        if self._outputs is not None:
            return self._outputs

        query = f"""
            SELECT o.* FROM group_experiments AS ge
                JOIN method_calls AS mc ON ge.group_id = mc.group_id
                JOIN outputs AS o ON mc.method_call_id = o.method_call_id
                WHERE ge.group_id = {self.group_id};
        """

        self._outputs = pd.read_sql_query(query, self.db.con)
        self._outputs.set_index("output_id", inplace=True)

        return self._outputs

    @property
    def programs(self) -> pd.DataFrame:
        """`DataFrame` with one entry per _Event.csv_ file"""
        if self._programs is not None:
            return self._programs

        query = f"""
            SELECT p.* FROM group_experiments AS g
                JOIN experiments AS e ON e.exp_id = g.exp_id
                JOIN programs    AS p ON p.exp_id = e.exp_id
                WHERE g.group_id = {self.group_id};
        """

        self._programs = pd.read_sql_query(
            query, self.db.con, parse_dates=["program_start"]
        )
        self._programs.set_index("program_id", inplace=True)

        return self._programs

    @property
    def trials(self) -> pd.DataFrame:
        """`DataFrame` with all olfactometer trials"""
        if self._trials is not None:
            return self._trials

        query = f"""
            SELECT t.* FROM group_experiments AS g
                JOIN experiments AS x ON x.exp_id     = g.exp_id
                JOIN programs    AS p ON p.exp_id     = x.exp_id
                JOIN trials      AS t ON t.program_id = p.program_id
                WHERE g.group_id = {self.group_id};
        """

        self._trials = pd.read_sql_query(
            query, self.db.con, parse_dates=["trial_start", "odor_start", "odor_end"]
        )
        self._trials.set_index("trial_id", inplace=True)

        return self._trials

    # ----------------------------------------------------------------------- #
    # Related to Records of Method Calls
    # ----------------------------------------------------------------------- #

    @property
    def current_call_id(self) -> int:
        assert (
            self._call_stack
        ), "'current_call_id' is only available inside a '@record_call'"
        return self._call_stack[-1].call_id

    def add_flag(self, flag) -> None:
        """Set bits on the current call's flag (bitwise OR). Use inside `@record_call`."""
        self._call_stack[-1].flag |= int(flag)

    def add_output_file(self, path: str | Path) -> None:
        """
        Record an output file in the outputs table. Use inside @record_call.
        """
        rel_path = str(Path(path).relative_to(self.db.main_folder))
        with self.db.con as con:
            con.execute(
                "INSERT INTO outputs (method_call_id, file_path, removed) VALUES (?, ?, FALSE);",
                [self.current_call_id, rel_path],
            )

        self._outputs = None

    def fail(self, flag, message: str = "") -> None:
        """Flag the current call and abort it by raising RuntimeError."""
        self.add_flag(flag)
        raise RuntimeError(message)

    def set_output(self, output: Object) -> None:
        """Record this call's output as JSON. Last write wins."""
        self._call_stack[-1].output = output

    # ----------------------------------------------------------------------- #
    # Database Queries
    # ----------------------------------------------------------------------- #

    def latest_calls(self, method_name: str) -> pd.DataFrame:
        """Return DataFrame with all calls to `method_name`."""

        query = """
            SELECT * FROM method_calls
                WHERE group_id = ? AND method_name LIKE ?
                ORDER BY method_call_id DESC
            """
        return _method_calls_dataframe(
            self.db.con, query, [self.group_id, f"%{method_name}"]
        )

    def latest_output(self, method_name: str) -> None | Object:
        """Return output of the most recent call to 'method_name'."""

        row = self.db.con.execute(
            """
            SELECT call_output FROM method_calls
                WHERE group_id = ? AND method_name = ? AND call_output IS NOT NULL
                ORDER BY method_call_id DESC LIMIT 1
            """,
            [self.group_id, method_name],
        ).fetchone()

        return json.loads(row["call_output"]) if row else None

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _reset_caches(self) -> None:
        self._acquisitions = None
        self._events = None
        self._experiments = None
        self._mcor_files = None
        self._method_calls = None
        self._outputs = None
        self._programs = None
        self._trials = None

        self.db._acquisitions = None
        self.db._events = None
        self.db._experiments = None
        self.db._mcor_files = None
        self.db._method_calls = None
        self.db._outputs = None
        self.db._programs = None
        self.db._trials = None

    # ----------------------------------------------------------------------- #
    # Motion Correction functions
    # ----------------------------------------------------------------------- #

    def delete_temp_files(self) -> None:
        """
        \033[1;35mDELETE_TEMP_FILES**
        Deletes all temp files associated with this experiment

        **USAGE*
            group = Group(experimentFolder)
            group.delete_temp_files()

        **EXAMPLES*
            group.delete_temp_files()
        """

        # Get file paths for mmaps in the temp folder
        path = Path(get_tempdir())

        exp_movies = []
        for exp_name in self.experiments["exp_name"]:
            exp_movies.append((exp_name, sorted(path.glob(f"{exp_name}*.mmap"))))

        # Remove files
        for exp_name, movie_paths in exp_movies:
            logger.info(f"Removing .mmap files that start with {exp_name}...")
            if movie_paths:
                total_size = 0.0  # in bytes
                for movie_path in movie_paths:
                    total_size += movie_path.stat().st_size
                    movie_path.unlink(missing_ok=True)

                total_size = total_size / (1_000_000_000)  # in GBs
                ending = "s" if len(movie_paths) > 1 else ""
                logger.info(
                    f"Deleted {len(movie_paths)} file{ending} ({total_size:.1f} GB)."
                )

            else:
                logger.info("No .mmap files found.")

        self._raw_mmap_pairs = None

    @record_call
    def pick_mcor_parameters(self, *, frame_fraction: float = 0.1) -> None:
        """
        Open a GUI to pick motion-correction parameters

        **PARAMETERS**
        - `frame_fraction`: percentage of frames (of the first raw file) to
        use in the preview.
        """
        import sys

        import numpy as np
        import bokeh.plotting as bpl

        from threading import Thread
        from bokeh.io import output_notebook
        from bokeh.io.state import curstate
        from bokeh.events import MouseMove
        from bokeh.models import (
            Button,
            ColumnDataSource,
            CustomJS,
            Div,
            LinearColorMapper,
            PointDrawTool,
            RadioButtonGroup,
            Spinner,
            TapTool,
        )
        from bokeh.palettes import Greys256
        from caiman.motion_correction import sliding_window_dims

        if "ipykernel" in sys.modules and not curstate().notebook:
            import os

            os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"  # HACK: render inside VSCode
            output_notebook()

        # Metadata from the database
        exp = self.experiments.iloc[0]
        dims = (int(exp["height_px"]), int(exp["width_px"]))  # (rows, cols) = (y, x)
        um_per_px = (exp["height_um"] / dims[0], exp["width_um"] / dims[1])
        raw_path = self.db.main_folder / self.acquisitions["raw_path"].iloc[0]
        step = max(1, round(1 / frame_fraction))

        # Initial values from the last pick or the shared defaults
        prev = self.latest_output("Group.pick_mcor_parameters") or {}
        strides_um = prev.get("strides_um", DEFAULT_STRIDES_UM)
        overlap_um = prev.get("overlap_um", DEFAULT_OVERLAP_UM)
        max_shift_um = prev.get("max_shift_um", DEFAULT_MAX_SHIFT_UM)
        max_deviation_um = prev.get("max_deviation_um", DEFAULT_MAX_DEVIATION_UM)

        # @record_call writes call_output=NULL on return, and Save button
        # updates the records with the chosen parameters.
        call_id = self.current_call_id
        con = self.db.con

        # Shift limits are capped at dim/4 (caiman needs 2 * max_shift < dim, and
        # run_motion_correction clamps max_shift to dim/4 anyway).
        init = {
            "sy": clamp(int(strides_um[0] / um_per_px[0]), 1, dims[0] // 2),
            "sx": clamp(int(strides_um[1] / um_per_px[1]), 1, dims[1] // 2),
            "oy": clamp(int(overlap_um[0] / um_per_px[0]), 1, dims[0] // 2),
            "ox": clamp(int(overlap_um[1] / um_per_px[1]), 1, dims[1] // 2),
            "my": clamp(int(max_shift_um[0] / um_per_px[0]), 1, dims[0] // 4),
            "mx": clamp(int(max_shift_um[1] / um_per_px[1]), 1, dims[1] // 4),
            "dev": clamp(int(max_deviation_um / min(um_per_px)), 1, min(dims) // 4),
        }

        def patch_data(sy, sx, oy, ox):
            # same tile_and_correct() that caiman uses
            xs, ys, ws, hs = [], [], [], []
            for _inds, (r0, c0), (h, w) in sliding_window_dims(
                dims, (oy, ox), (sy, sx)
            ):
                ys.append(r0 + h / 2)
                xs.append(c0 + w / 2)
                hs.append(h)
                ws.append(w)
            return dict(x=xs, y=ys, w=ws, h=hs)

        def modify_doc(doc):
            # render immediately and the background is added later
            p = bpl.figure(
                x_range=(0, dims[1]), y_range=(dims[0], 0), width=600, height=600
            )
            p.xaxis.visible = p.yaxis.visible = False

            cmap = LinearColorMapper(palette=Greys256, low=0.0, high=1.0)
            img = ColumnDataSource(data=dict(image=[np.zeros(dims, dtype="float32")]))
            p.image(
                image="image",
                source=img,
                x=0,
                y=0,
                dw=dims[1],
                dh=dims[0],
                color_mapper=cmap,
            )

            def empty():
                return dict(x=[], y=[], w=[], h=[])

            # Patches-view layers
            grid = ColumnDataSource(data=empty())  # full faint patch grid
            grid_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=grid,
                alpha=0.1,
                line_color="white",
            )

            # Select-view layer (click to highlight)
            select_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=grid,
                alpha=0.2,
                line_color="white",
                selection_color="red",
            )

            overlap = ColumnDataSource(data=empty())  # shaded patch overlap
            overlap_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=overlap,
                fill_color="orange",
                fill_alpha=0.35,
                line_alpha=0,
            )

            # Shifts-view layers
            band = ColumnDataSource(data=empty())  # max_shift border strip
            band_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=band,
                fill_color="red",
                fill_alpha=0.12,
                line_alpha=0,
            )

            halo = ColumnDataSource(data=empty())  # max_deviation wiggle room
            halo_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=halo,
                fill_alpha=0,
                line_color="lime",
                line_dash="dashed",
            )

            # Patches view: the solid color patch
            anchor = ColumnDataSource(data=empty())
            anchor_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=anchor,
                fill_alpha=0,
                line_color="yellow",
                line_width=2,
            )

            # Shifts view: same patch dashed at its actual position, plus a solid
            # copy shifted so its top-left sits on the red max_shift handle.
            actual_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=anchor,
                fill_alpha=0,
                line_color="yellow",
                line_dash="dashed",
            )
            moved = ColumnDataSource(data=empty())
            moved_r = p.rect(
                "x",
                "y",
                width="w",
                height="h",
                source=moved,
                fill_alpha=0,
                line_color="yellow",
                line_width=2,
            )

            def handle(color):
                src = ColumnDataSource(data=dict(x=[0], y=[0]))
                r = p.scatter(
                    "x", "y", source=src, size=12, fill_color=color, line_color="black"
                )
                return src, r

            h_size, r_sz = handle("yellow")  # corner: size = stride + overlap (x, y)
            h_overlap, r_ov = handle("orange")  # interior: overlap (x, y)
            h_shift, r_ms = handle("red")  # corner: max_shift (x, y)

            drag = PointDrawTool(renderers=[r_sz, r_ov, r_ms], add=False)
            tap = TapTool(renderers=[select_r])
            p.add_tools(drag, tap)
            p.toolbar.active_drag = drag

            # Spinners for fine-tuning and readout
            def spinner(value, hi, step, title):
                return Spinner(
                    low=0, high=hi, step=step, value=value, title=title, width=110
                )

            sp_sx = spinner(
                init["sx"] * um_per_px[1],
                dims[1] * um_per_px[1],
                um_per_px[1],
                "Stride x (µm)",
            )
            sp_sy = spinner(
                init["sy"] * um_per_px[0],
                dims[0] * um_per_px[0],
                um_per_px[0],
                "Stride y (µm)",
            )
            sp_ox = spinner(
                init["ox"] * um_per_px[1],
                dims[1] * um_per_px[1],
                um_per_px[1],
                "Overlap x (µm)",
            )
            sp_oy = spinner(
                init["oy"] * um_per_px[0],
                dims[0] * um_per_px[0],
                um_per_px[0],
                "Overlap y (µm)",
            )
            sp_mx = spinner(
                init["mx"] * um_per_px[1],
                (dims[1] // 2) * um_per_px[1],
                um_per_px[1],
                "Max shift x (µm)",
            )
            sp_my = spinner(
                init["my"] * um_per_px[0],
                (dims[0] // 2) * um_per_px[0],
                um_per_px[0],
                "Max shift y (µm)",
            )
            sp_dev = spinner(
                init["dev"] * min(um_per_px),
                (min(dims) // 4) * min(um_per_px),
                max(um_per_px),
                "Max deviation (µm)",
            )

            view = RadioButtonGroup(labels=["Patches", "Shifts", "Select"], active=0)
            save = Button(label="Save Parameters", button_type="success")
            status = Div(text="<i>Loading background image…</i>")

            def apply_view():
                # 0 = Patches, 1 = Shifts, 2 = Select
                mode = view.active

                grid_r.visible = overlap_r.visible = mode == 0
                r_sz.visible = r_ov.visible = mode == 0  # size / overlap handles

                anchor_r.visible = mode == 0  # solid patch (Patches only)
                actual_r.visible = moved_r.visible = mode == 1  # dashed + shifted
                band_r.visible = halo_r.visible = r_ms.visible = mode == 1

                select_r.visible = mode == 2  # click-to-highlight grid

                # Select -> tap tool
                if mode == 2:
                    p.toolbar.active_drag = None
                    p.toolbar.active_tap = tap

                # Patches/Shifts -> drag tool
                else:
                    p.toolbar.active_drag = drag
                    p.toolbar.active_tap = None

            view.on_change("active", lambda attr, old, new: apply_view())

            # All relevant data (in pixels)
            S = {k: init[k] for k in ("sx", "sy", "ox", "oy", "mx", "my", "dev")}
            flags = {"sync": False}

            def clamp_state():
                S["sx"] = clamp(S["sx"], 1, dims[1])
                S["sy"] = clamp(S["sy"], 1, dims[0])
                S["ox"] = clamp(S["ox"], 0, dims[1] - S["sx"])
                S["oy"] = clamp(S["oy"], 0, dims[0] - S["sy"])
                S["mx"] = clamp(S["mx"], 1, dims[1] // 2 - 1)
                S["my"] = clamp(S["my"], 1, dims[0] // 2 - 1)
                S["dev"] = clamp(S["dev"], 0, min(dims) // 4)

            def redraw():
                clamp_state()
                ww, wh = S["sx"] + S["ox"], S["sy"] + S["oy"]
                H, W = dims

                grid.data = patch_data(S["sy"], S["sx"], S["oy"], S["ox"])
                anchor.data = dict(x=[ww / 2], y=[wh / 2], w=[ww], h=[wh])

                ov = empty()  # overlap rects only when overlap is positive
                if S["ox"] > 0:
                    ov["x"].append((S["sx"] + ww) / 2)
                    ov["y"].append(wh / 2)
                    ov["w"].append(S["ox"])
                    ov["h"].append(wh)
                if S["oy"] > 0:
                    ov["x"].append(ww / 2)
                    ov["y"].append((S["sy"] + wh) / 2)
                    ov["w"].append(ww)
                    ov["h"].append(S["oy"])
                overlap.data = ov

                # shifted copy: top-left on the max_shift handle (mx, my)
                mxc, myc = S["mx"] + ww / 2, S["my"] + wh / 2
                moved.data = dict(x=[mxc], y=[myc], w=[ww], h=[wh])
                halo.data = dict(
                    x=[mxc],
                    y=[myc],
                    w=[ww + 2 * S["dev"]],
                    h=[wh + 2 * S["dev"]],
                )
                band.data = dict(
                    x=[W / 2, W / 2, S["mx"] / 2, W - S["mx"] / 2],
                    y=[S["my"] / 2, H - S["my"] / 2, H / 2, H / 2],
                    w=[W, W, S["mx"], S["mx"]],
                    h=[S["my"], S["my"], H, H],
                )

                flags["sync"] = True  # programmatic updates must not re-fire
                h_size.data = dict(x=[ww], y=[wh])  # lower-right corner of anchor
                h_overlap.data = dict(x=[S["sx"]], y=[S["sy"]])  # neighbor onsets
                h_shift.data = dict(x=[S["mx"]], y=[S["my"]])
                sp_sx.value = S["sx"] * um_per_px[1]
                sp_sy.value = S["sy"] * um_per_px[0]
                sp_ox.value = S["ox"] * um_per_px[1]
                sp_oy.value = S["oy"] * um_per_px[0]
                sp_mx.value = S["mx"] * um_per_px[1]
                sp_my.value = S["my"] * um_per_px[0]
                sp_dev.value = S["dev"] * min(um_per_px)
                flags["sync"] = False

            # handle drags
            def on_size(attr, old, new):
                # corner sets size (stride + overlap); stride stays, overlap absorbs
                if flags["sync"]:
                    return
                cx = clamp(round(h_size.data["x"][0]), S["sx"], dims[1])
                cy = clamp(round(h_size.data["y"][0]), S["sy"], dims[0])
                S["ox"] = cx - S["sx"]
                S["oy"] = cy - S["sy"]
                redraw()

            def on_overlap(attr, old, new):
                # interior sets neighbor onset (stride); size stays, overlap absorbs
                if flags["sync"]:
                    return
                ww, wh = S["sx"] + S["ox"], S["sy"] + S["oy"]
                ix = clamp(round(h_overlap.data["x"][0]), 1, ww)
                iy = clamp(round(h_overlap.data["y"][0]), 1, wh)
                S["sx"], S["ox"] = ix, ww - ix
                S["sy"], S["oy"] = iy, wh - iy
                redraw()

            def on_shift(attr, old, new):
                if flags["sync"]:
                    return
                S["mx"] = round(h_shift.data["x"][0])
                S["my"] = round(h_shift.data["y"][0])
                redraw()

            h_size.on_change("data", on_size)
            h_overlap.on_change("data", on_overlap)
            h_shift.on_change("data", on_shift)

            # --- Preview while dragging ---
            # PointDrawTool only emits its 'data' change on release, but MouseMove
            # fires continuously and the handle coordinates are mutated in place,
            # so we read them on every move and redraw just the dragged primitives.
            # The full grid and spinners still settle on release (Python callbacks).

            p.js_on_event(
                MouseMove,
                CustomJS(
                    args=dict(
                        size=h_size,
                        inter=h_overlap,
                        shift=h_shift,
                        dev=sp_dev,
                        view=view,
                        anchor=anchor,
                        overlap=overlap,
                        moved=moved,
                        halo=halo,
                        band=band,
                        H=dims[0],
                        W=dims[1],
                        minfac=min(um_per_px),
                    ),
                    code="""
                const mode = view.active;  // 0 Patches, 1 Shifts, 2 Select
                if (mode === 2) return;

                // geometry straight from the (live) handle positions, in pixels
                let sx = Math.round(inter.data['x'][0]);
                let sy = Math.round(inter.data['y'][0]);
                const ww = Math.max(sx, Math.min(Math.round(size.data['x'][0]), W));
                const wh = Math.max(sy, Math.min(Math.round(size.data['y'][0]), H));
                sx = Math.max(1, Math.min(sx, ww));
                sy = Math.max(1, Math.min(sy, wh));
                const mx = Math.max(1, Math.min(Math.round(shift.data['x'][0]), Math.floor(W/2)-1));
                const my = Math.max(1, Math.min(Math.round(shift.data['y'][0]), Math.floor(H/2)-1));
                const devpx = Math.round(dev.value / minfac);

                // skip redundant work (e.g. plain hover, no handle moved)
                const sig = mode+':'+ww+','+wh+','+sx+','+sy+','+mx+','+my+','+devpx;
                if (view.__sig === sig) return;
                view.__sig = sig;

                if (mode === 0) {  // Patches: anchor + overlap shading
                    anchor.data = {x:[ww/2], y:[wh/2], w:[ww], h:[wh]};
                    anchor.change.emit();
                    const xs=[], ys=[], ws=[], hs=[];
                    if (ww - sx > 0) { xs.push((sx+ww)/2); ys.push(wh/2); ws.push(ww-sx); hs.push(wh); }
                    if (wh - sy > 0) { xs.push(ww/2); ys.push((sy+wh)/2); ws.push(ww); hs.push(wh-sy); }
                    overlap.data = {x:xs, y:ys, w:ws, h:hs};
                    overlap.change.emit();
                } else {  // Shifts: shifted copy + halo + border band
                    const cx = mx + ww/2, cy = my + wh/2;
                    moved.data = {x:[cx], y:[cy], w:[ww], h:[wh]};
                    moved.change.emit();
                    halo.data = {x:[cx], y:[cy], w:[ww+2*devpx], h:[wh+2*devpx]};
                    halo.change.emit();
                    band.data = {x:[W/2, W/2, mx/2, W-mx/2], y:[my/2, H-my/2, H/2, H/2],
                                 w:[W, W, mx, mx], h:[my, my, H, H]};
                    band.change.emit();
                }
                """,
                ),
            )

            # Spinner edits (um -> pixels)
            def on_spinner(key, factor):
                def cb(attr, old, new):
                    if flags["sync"] or new is None:
                        return
                    S[key] = round(new / factor)
                    redraw()

                return cb

            sp_sx.on_change("value", on_spinner("sx", um_per_px[1]))
            sp_sy.on_change("value", on_spinner("sy", um_per_px[0]))
            sp_ox.on_change("value", on_spinner("ox", um_per_px[1]))
            sp_oy.on_change("value", on_spinner("oy", um_per_px[0]))
            sp_mx.on_change("value", on_spinner("mx", um_per_px[1]))
            sp_my.on_change("value", on_spinner("my", um_per_px[0]))
            sp_dev.on_change("value", on_spinner("dev", min(um_per_px)))

            def save_callback():
                output = {
                    "strides_um": [
                        float(S["sy"] * um_per_px[0]),
                        float(S["sx"] * um_per_px[1]),
                    ],
                    "overlap_um": [
                        float(S["oy"] * um_per_px[0]),
                        float(S["ox"] * um_per_px[1]),
                    ],
                    "max_shift_um": [
                        float(S["my"] * um_per_px[0]),
                        float(S["mx"] * um_per_px[1]),
                    ],
                    "max_deviation_um": float(S["dev"] * min(um_per_px)),
                }
                with con:
                    con.execute(
                        "UPDATE method_calls SET call_output = ? WHERE method_call_id = ?",
                        [json.dumps(output), call_id],
                    )
                status.text = (
                    "Saved! run_motion_correction() will use these parameters "
                    "by default (use_gui_parameters=False to override)."
                )

            save.on_click(save_callback)

            redraw()  # initialise every glyph, handle and spinner from S
            apply_view()  # set initial layer visibility for the active view

            doc.add_root(
                bpl.column(
                    view,
                    bpl.row(
                        p,
                        bpl.column(
                            bpl.row(sp_sy, sp_sx),
                            bpl.row(sp_oy, sp_ox),
                            bpl.row(sp_my, sp_mx),
                            sp_dev,
                            save,
                            status,
                        ),
                    ),
                )
            )

            # Load the subsampled movie in a different thread
            def load_background():
                cache = (
                    self.db.main_folder
                    / ODYN_FOLDER
                    / f"corr_{raw_path.stem}_{step}.npy"
                )

                if cache.exists():
                    corr = np.load(cache)

                else:
                    # subindices reads every Nth page
                    movie = cm.load(str(raw_path), subindices=slice(None, None, step))
                    corr = cm.local_correlations(movie, swap_dim=False)
                    corr[np.isnan(corr)] = 0
                    np.save(cache, corr)

                lo, hi = float(np.quantile(corr, 0.01)), float(np.quantile(corr, 0.99))

                def apply():
                    img.data = dict(image=[corr])
                    cmap.low, cmap.high = lo, hi
                    status.text = "Ready."

                doc.add_next_tick_callback(apply)  # thread-safe UI update

            doc.add_next_tick_callback(
                lambda: Thread(target=load_background, daemon=True).start()
            )

        bpl.show(modify_doc)

    @memorize_params
    @record_call
    def play_movie(
        self,
        *,
        grid: list[str] = ["raw", "mcor"],
        downsample_ratio: float = 0.03,
        opencv_codec: str = "MJPG",
        save_movie: bool = True,
        save_folder: str = r"./movies",
        backend: str = "embed_opencv",
        do_loop: bool = False,
        fr: float = 30,
        magnification: float = 1,
        plot_text: bool = True,
        q_max: float = 99.5,
        q_min: float = 0.00,
    ) -> None:
        """
        Play and save movies for quality control

        **USAGE**
        ```python
            group = db.groups[some_index]
            group.play_movie(...)
        ```

        **PARAMETERS**

        *Basic settings*
        - `use_last_parameters`: Use parameters from last run as the defaults
        - `grid`: How to concatenate movies horizontally ("raw", "mcor", "test" in some order)

        *Movie loading settings*
        - `downsample_ratio`: Percentage of frames to keep

        *Video saving*
        - `opencv_codec`: Codec used to encode the saved video
        - `save_movie`: Put `True` if you want to save the preview video to a file
        - `save_folder`: `"."` is the main_folder (r is to use \\ in the path)

        *Video settings*
        - `backend`: "opencv" for popup and "embed_opencv" for inline player
        - `do_loop`: Loop the video or not
        - `fr`: How fast to play the video (frames/s)
        - `magnification`: Magnification of video
        - `plot_text`: Add current frame label on the video
        - `q_max`: Quantile to consider as white
        - `q_min`: Quantile to consider as black

        **EXAMPLES**
        - Running this command would save a compilation of all raw movies
        ```python
        group.play_movie(grid=["raw"], save_folder="~/TempData/20260101/e1/movies")
        ```

        - This would save a video with raw movies on the left and test on the right
        ```python
        group.play_movie(grid=["raw", "test"])
        ```
        """
        # --- Validate parameters --- #

        # Check grid input
        assert len(grid) == 1 or (
            len(grid) == 2
            and MovieType.RAW.value in grid
            and (MovieType.MCOR.value in grid or MovieType.TEST.value in grid)
        ), """The parameter 'grid' must be one of the following:
                ['raw'], ['mcor'], ['test'],
                ['raw', 'test'], ['raw', 'mcor'],
                ['test', 'raw'], ['mcor', 'raw']"""

        params = locals()

        # --- Check if movie needs to be updated --- #

        # Exclude the current call (already recorded by decorator) so we find
        # the most recent *previous* call with the same grid.
        play_movie_calls = self.method_calls[
            (self.method_calls["method_name"] == "Group.play_movie")
            & (self.method_calls.index != self.current_call_id)
        ]
        last_call_id = play_movie_calls[
            play_movie_calls["parameters_used"].apply(lambda x: x.get("grid") == grid)
        ].index.max()

        # Trigger recompute if the parameters changed from the last call
        movie_types = tuple(MovieType(s) for s in grid)
        params.pop("self", None)

        if movie_types not in self.movies:
            self.movies[movie_types] = LazyMovie(self, movie_types)

        # Recompute if there is no comparable earlier call
        # (e.g. for rows that were backfilled by a migration)
        elif (
            pd.isna(last_call_id)
            or play_movie_calls.loc[last_call_id, "parameters_used"] != params
        ):
            self.movies[movie_types].mark_as_outdated()

        # --- Transform parameters --- #

        # TODO: Change name and default folder
        movie_type_str = "_".join(t.value for t in movie_types)
        first_exp_name = self.experiments.iloc[0]["exp_name"]
        filename = f"Group_{self.group_id}_{first_exp_name}_{movie_type_str}.avi"

        filepath = (
            (self.db.main_folder / save_folder / filename).resolve()
            if save_folder[0] == "."
            else (Path(save_folder) / filename).resolve()
        )

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # --- Play and maybe update movies --- #
        # Only updated if run_motion_correction was called invalidating the
        # relevant files (raw or mcor TIFFs, or test MMAPs)
        movie_name = str(filepath)

        if save_movie:
            logger.info(f"Saving movie to {movie_name}")

        self.movies[movie_types].maybe_update(downsample_ratio).play(
            backend=backend,
            do_loop=do_loop,
            fr=fr,
            magnification=magnification,
            plot_text=plot_text,
            q_max=q_max,
            q_min=q_min,
            opencv_codec=opencv_codec,
            save_movie=save_movie,
            movie_name=movie_name,
        )

    @memorize_params
    @record_call
    def run_motion_correction(
        self,
        *,
        use_gui_parameters: bool = True,
        is_test: bool = True,
        first_acq: int = 0,
        step_acq: int = 1,
        last_acq: int = 3,
        border_nan: bool | str = "copy",
        nonneg_movie: bool = False,
        pw_rigid: bool = True,
        shifts_opencv: bool = False,
        max_deviation_um: float = DEFAULT_MAX_DEVIATION_UM,
        max_shift_um: list[float] = DEFAULT_MAX_SHIFT_UM,
        overlap_um: list[float] = DEFAULT_OVERLAP_UM,
        strides_um: list[float] = DEFAULT_STRIDES_UM,
    ) -> None:
        """
        Method that does test/final motion correction

        **USAGE**
        ````python
        db    = Database(main_folder)
        group = db.groups[some_index]
        group.run_motion_correction(...)
        ```

        **LIST OF PARAMETERS**

        *Basic Parameters*
        - `use_gui_parameters`: Override parameters with latest `pick_mcor_parameters()` values
        - `use_last_parameters`: Use parameters from last run as the defaults
        - `is_test`: Whether to use a limited range of acquisitions in this run

        *Parameters in this section will be ignored if* `is_test == False`
        - `first_acq`: Index of the first acquisition to motion correct
        - `step_acq`: Get one acquisition for every 'step_acq' acquisitions
        - `last_acq`: Index of the last acquisition to motion correct

        *CaImAn motion correction parameters (before metadata adjustments)*
        - `border_nan`: copy along the boundary (if `True`, fill in with NaN)
        - `nonneg_movie`: make SAVED movie mostly non-negative
        - `pw_rigid`: Piecewise-rigid (`True`) or rigid motion correction
        - `shifts_opencv`: `True` = bicubic, `False` = FFT (True is faster)
        - `max_deviation_um`: max deviation for patch with respect to rigid shifts
        - `max_shift_um`: max allowed rigid shift
        - `overlap_um`: overlap between patches (patch = strides + overlaps)
        - `strides_um`: start a new patch every x or y um (only for pw-rigid)

        **EXAMPLES**
        ```python
        group.run_motion_correction(
            is_test=False,
            use_gui_parameters=False,
            overlap = [48.0, 48.0]
        )
        ```
        """

        # --- Optionally override parameters with GUI-picked values --- #
        # These override both the passed arguments and use_last_parameters.

        if use_gui_parameters:
            gui = self.latest_output("Group.pick_mcor_parameters")

            if gui is None:
                logger.warning(
                    "use_gui_parameters=True but no saved parameters found; "
                    "run pick_mcor_parameters() first. Using provided values."
                )
            else:
                logger.info("Loaded motion-correction parameters from the GUI.")
                max_deviation_um = cast(
                    float, gui.get("max_deviation_um", max_deviation_um)
                )
                max_shift_um = cast(list[float], gui.get("max_shift_um", max_shift_um))
                overlap_um = cast(list[float], gui.get("overlap_um", overlap_um))
                strides_um = cast(list[float], gui.get("strides_um", strides_um))

        # --- Validate and adjust parameters to reasonable values --- #

        # If is not test, include all acquisitions
        if not is_test:
            first_acq = 0
            step_acq = 1
            last_acq = len(self.acquisitions) - 1

        # max_shift_um has to less than size of image (in μm / 4)
        height_um = self.experiments["height_um"].min()
        width_um = self.experiments["width_um"].min()
        max_shift_um = [
            clamp(max_shift_um[0], 0, height_um / 4),
            clamp(max_shift_um[1], 0, width_um / 4),
        ]

        # --- Make sure movies will be updated next time they are played --- #

        if is_test:
            for movie_types in self.movies.keys():
                if MovieType.TEST in movie_types:
                    self.movies[movie_types].mark_as_outdated()
        else:
            for movie_types in self.movies.keys():
                if MovieType.MCOR in movie_types:
                    self.movies[movie_types].mark_as_outdated()

        # --- Get settings for CaImAn MotionCorrection class --- #

        # Get raw paths
        acquisitions_slice = self.acquisitions.iloc[first_acq : last_acq + 1 : step_acq]
        raw_paths = [self.db.main_folder / p for p in acquisitions_slice["raw_path"]]

        assert raw_paths, "No raw files within the index range."

        # CaImAn only uses pixels as units, so we make the conversion
        factor = [
            height_um / self.experiments["height_px"].min(),
            width_um / self.experiments["width_px"].min(),
        ]

        settings = {
            "border_nan": border_nan,
            "pw_rigid": pw_rigid,
            "shifts_opencv": shifts_opencv,
            "nonneg_movie": nonneg_movie,
            "max_deviation_rigid": int(max_deviation_um) / min(factor),
            "max_shifts": um_to_pixels(max_shift_um, factor),
            "overlaps": um_to_pixels(overlap_um, factor),
            "strides": um_to_pixels(strides_um, factor),
        }

        # --- Run the motion correction --- #

        logger.info("Starting motion correction...")

        _, dview, _ = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=None, single_thread=False
        )

        try:
            self.mc = MotionCorrect(raw_paths, dview=dview, **settings)
            self.mc.motion_correct(save_movie=True)

        except:
            raise

        # Always stop the server after motion correction
        finally:
            cm.stop_server(dview=dview)

        logger.info("Finished motion correction")

        # --- Save the results in TIFF files (if it is not a test) --- #
        if is_test:
            # Stores the list of raw and mmap paths to make a movie later
            self._raw_mmap_pairs = (
                [str(p) for p in raw_paths],
                self.mc.mmap_file.copy(),
            )
            return

        with self.db.con as con:
            logger.info("Saving mcor files...")

            # Load mmap files and save them as TIFFs
            # TODO: Confirm that mmap_file and fname have the same order
            for acq_id, mmap_path, raw_path in tqdm(
                zip(acquisitions_slice.index, self.mc.mmap_file, self.mc.fname),
                desc="Saving mcor files",
                total=len(self.mc.mmap_file),
            ):
                # Create folder if it doesn't exist
                mcor_folder = raw_path.parent.parent / "processed" / "mcor"
                mcor_folder.mkdir(parents=True, exist_ok=True)

                mcor_path = mcor_folder / (raw_path.stem + "_mcor.tif")

                mc = cm.load(mmap_path)

                # Saving TIFFs directly because caiman saves them as 64-bit
                with tifffile.TiffWriter(mcor_path) as tif:
                    tif.write(
                        [mc[i].copy() for i in range(mc.shape[0])],
                        shape=mc[0].shape,
                        dtype=mc.dtype,
                    )

                # Update the database with the latest mcor files
                insertion_query = """
                    INSERT OR REPLACE INTO mcor_files
                        ( acq_id
                        , mcor_path
                        , last_updated_by
                        ) VALUES (?, ?, ?);
                """
                con.execute(
                    insertion_query,
                    [
                        acq_id,
                        str(mcor_path.relative_to(self.db.main_folder)),
                        self.current_call_id,
                    ],
                )

            # Reset mcor DataFrames
            self._mcor_files = None
            self.db._mcor_files = None

    # ----------------------------------------------------------------------- #
    # Data Analysis
    # ----------------------------------------------------------------------- #

    # def compute_z_scores(self) -> cm.movie:
    #     # Check an sync config
    #     self._sync_config()

    #     # TODO: replace dummy values below
    #     odor_onset = 2
    #     odor_offset = 4
    #     # END OF DUMMY VALUES ------------

    #     # Get config parameters
    #     frame_rate = self.config["metadata"]["frame_rate"]
    #     post_odor_interval = self.config["z-scores"]["post_odor_interval"]
    #     baseline_pre_odor = self.config["z-scores"]["baseline_pre_odor"]

    #     frame_odor_onset = int(odor_onset * frame_rate)
    #     frame_onset = int((odor_onset + post_odor_interval) * frame_rate)
    #     frame_offset = int((odor_offset + post_odor_interval) * frame_rate)
    #     frame_duration = frame_offset - frame_onset + 1

    #     frame_baseline_start = (
    #         frame_odor_onset - frame_duration
    #         if baseline_pre_odor
    #         else frame_onset - frame_duration
    #     )

    #     mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
    #     mcor_files = mcor_folder.glob(f"[!.]?*_mcor.tif")

    #     assert mcor_files, "No .tif files in the mcor folder."

    #     # TIFFs were originally in int16, so we add 32768 to make then non-negative
    #     baseline_range = range(frame_baseline_start, frame_duration)
    #     baseline_avg = cm.load(mcor_files[0], subindices=baseline_range).mean(axis=0)
    #     baseline_avg += 32768

    #     signal_range = range(frame_onset, frame_duration)
    #     signal_avg = cm.load(mcor_files[0], subindices=signal_range).mean(axis=0)
    #     signal_avg += 32768

    #     # Computes z-score of dF/F
    #     dFF = (signal_avg - baseline_avg) / baseline_avg
    #     z_score = (dFF - dFF.mean()) / dFF.std()

    #     return z_score


# --------------------------------------------------------------------------- #
# Auxiliary Classes
# --------------------------------------------------------------------------- #


@dataclass
class LazyMovie:
    """
    Movies that only reload when needed.
    """

    owner: Group
    types: tuple[MovieType, ...]
    movie: None | cm.movie = None

    def mark_as_outdated(self):
        self.movie = None

    def maybe_update(self, downsample_ratio) -> cm.movie:
        if self.movie is not None:
            logger.info("Using cached movie...")
            return self.movie

        logger.info("Updating movie...")

        movie_chains = []

        if MovieType.TEST in self.types:
            # If test is included there must be test files
            assert (
                self.owner._raw_mmap_pairs is not None
            ), "There are no cached 'test' file paths."

        for movie_type in self.types:
            # Get the movie_paths
            if movie_type == MovieType.TEST:
                self.owner.method_calls

                # Redundant check for mypy
                assert self.owner._raw_mmap_pairs is not None
                _, movie_paths = self.owner._raw_mmap_pairs

            elif MovieType.TEST in self.types:
                # Redundant check for mypy
                assert self.owner._raw_mmap_pairs is not None

                # It must be raw in this case
                movie_paths, _ = self.owner._raw_mmap_pairs

            # If there is no test always include all files
            elif movie_type == MovieType.RAW:
                movie_paths = [
                    self.owner.db.main_folder / path
                    for path in self.owner.acquisitions["raw_path"]
                ]

            elif movie_type == MovieType.MCOR:
                movie_paths = [
                    self.owner.db.main_folder / path
                    for path in self.owner.mcor_files["mcor_path"]
                ]

            # There must be at least one movie
            # TODO: Gather movie_paths for every type first, so it fails
            #       before having spent time loading anything.
            assert movie_paths, f"Didn't find any {movie_type.value} files."

            logger.info(
                f"Adding {len(movie_paths)} {movie_type.value} files to the movie."
            )

            movie_chain = None
            for path in tqdm(movie_paths, desc=f"Loading {movie_type.value} movies"):
                movie = cm.load(path).resize(1, 1, downsample_ratio)
                movie_chain = (
                    movie
                    if movie_chain is None
                    else cm.concatenate([movie_chain, movie], axis=0)
                )
                logger.info(f"  {path}")

            movie_chains.append(movie_chain)

        movie_chain = cm.concatenate(movie_chains, axis=2)

        self.movie = movie_chain

        return self.movie


#     def play(self) -> None:
#         # TODO: - Use Caiman function as blueprint
#         #       - Add movie type, total time, current time, and speed (e.g. x2) to label
#         #       - Remove frame number from label
#         #       - Possibly add total time to config and remove fr in [play.load]
#         return
