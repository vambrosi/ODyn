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
import os

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast, Final, TYPE_CHECKING

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile

from matplotlib import colormaps

import caiman as cm
from caiman.base.movies import get_file_size
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .utils import *
from .utils import _method_calls_dataframe
from .utils import CallFrame, CallRecorder

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

# Limit used by tifffile.imwrite to switch to BigTIFF.
# Classic TIFF 32-bit offsets overflow past this limit.
BIGTIFF_BYTES = 2**32 - 2**25


class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"


class McorSource(Enum):
    """
    What motion corrected an `mcor_files` row.

    Must mirror the CHECK on `mcor_files.source` in `create.sql`.
    """

    CAIMAN = "caiman"
    PATCHWARP = "patchwarp"


# mcor files folder, relative to the experiment folder, and what the library appends
# to the raw file name. Should mirror McorSource enum and mcor_files.source CHECK.
MCOR_LAYOUT = {
    McorSource.CAIMAN: ("processed/mcor", "_mcor.tif"),
    McorSource.PATCHWARP: ("processed/patchwarp/post_warp", "_corrected_warped.tif"),
}


class McorFlag(IntFlag):
    """
    call_flag bits for the `Group` methods that write `mcor_files`
    (bit 0 reserved by `CallFlag.RAISED`).

    One enum for all of them because they share `_check_mcor_group`, so its
    bits have to mean the same thing whichever method set them.
    """

    ALREADY_HAS_FILES = 1 << 1  # group was not empty, nothing added
    REPLACED_EXISTING = 1 << 2  # previous files were dropped for these ones
    OWNED_BY_OTHER_GROUP = 1 << 3  # another group motion corrected these files
    SHARED_WITH_OTHER_GROUPS = 1 << 4  # other groups will see the change
    FILE_NOT_FOUND = 1 << 5  # no motion corrected file where one was expected
    WRONG_SHAPE = 1 << 6  # frame size does not match the experiment
    WRONG_FRAME_COUNT = 1 << 7  # frame count does not match the experiment
    UNREADABLE = 1 << 8  # file is there but its header could not be read
    NOTHING_TO_APPROVE = 1 << 9  # group has no mcor files at all
    SOME_NOT_APPROVED = 1 << 10  # approval left some acquisitions out


# --------------------------------------------------------------------------- #
# Main Data Processing\Analysis Class
# --------------------------------------------------------------------------- #


class Group(CallRecorder):
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
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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
    def approved_mcor_files(self) -> pd.DataFrame:
        """
        `DataFrame` with the mcor files approved for analysis

        Use this unless you are doing motion-correction specific analysis.
        Use `mcor_files` if you want ALL files from the latest mcor run.
        """
        return self.mcor_files[self.mcor_files["approved"].astype(bool)]

    @property
    def method_calls(self) -> pd.DataFrame:
        """`DataFrame` with `@record_call` functions"""
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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
        self.db._refresh_if_stale()

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

    def _latest_test_movies(self) -> tuple[list[Path], list[Path]]:
        """
        Raw files and their .mmap files, from the last test motion correction.

        Returns `(raw_paths, mmap_paths)`, matched by file name and in
        acquisition order. Read from `call_output` using `get_tempdir()`.

        This function allows `play_movie` to be called on test runs even
        after a kernel crash, as long as all relevant .mmap files are present.
        """
        calls = self.latest_calls("Group.run_motion_correction")

        # These columns will only appear in future calls, so we guard against it.
        if {"is_test", "mmap_names"} <= set(calls.columns):
            tests = calls[calls["is_test"].eq(True) & calls["mmap_names"].notna()]

        # Otherwise, make it empty to fail below.
        else:
            tests = calls.iloc[:0]

        if tests.empty:
            raise RuntimeError(f"{self!r} has no test motion correction. ")

        temp_folder = Path(get_tempdir())
        mmap_paths = [temp_folder / name for name in tests.iloc[0]["mmap_names"]]

        # Pair .mmap with the raw file it came from by name
        raw_paths: list[Path] = []
        matched: list[Path] = []

        for raw_path in map(Path, self.acquisitions["raw_path"]):
            # Pick the first with the right name
            for path in mmap_paths:
                if path.name.startswith(f"{raw_path.stem}_") and path.is_file():
                    raw_paths.append(self.db.main_folder / raw_path)
                    matched.append(path)
                    break

        # A test run only covers some of the acquisitions, so the ones left over
        # are expected. An .mmap with nothing to pair it to is not: either it
        # was deleted or it belongs somewhere else.
        if len(matched) != len(mmap_paths):
            raise RuntimeError(
                f"Some .mmap files are missing from '{temp_folder}'. "
                "Rerun 'run_motion_correction(is_test=True)'."
            )

        return raw_paths, matched

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

    @record_call
    def pick_mcor_parameters(
        self, *, frame_fraction: float = 0.1, image_type: str = "avg"
    ) -> None:
        """
        Open a GUI to pick motion-correction parameters

        **PARAMETERS**
        - `frame_fraction`: percentage of frames (of the first raw file) to
        use in the preview.
        - `image_type`: "avg" for average projection and "corr" for local correlations.
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

        # Anything read back out of a call_output is only known to be JSON, so
        # these say what was written rather than making the reader work it out.
        strides_um = cast(list[float], prev.get("strides_um", DEFAULT_STRIDES_UM))
        overlap_um = cast(list[float], prev.get("overlap_um", DEFAULT_OVERLAP_UM))
        max_shift_um = cast(list[float], prev.get("max_shift_um", DEFAULT_MAX_SHIFT_UM))
        max_deviation_um = cast(
            float, prev.get("max_deviation_um", DEFAULT_MAX_DEVIATION_UM)
        )

        # Check if it is one of the supported options
        if image_type not in ["avg", "corr"]:
            raise ValueError("image_type must be 'avg' or 'corr'.")

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
            # Same tiling caiman uses, so the drawing follows it rather than
            # copying it, and breaks loudly if that API ever changes
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
                x_range=(0, dims[1]),
                y_range=(dims[0], 0),
                height=800,
                aspect_ratio="auto",
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
            # The full grid and spinners still settle on release (Python
            # callbacks), so the same geometry exists in Python and in JS:
            # change one and change the other.

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
                    / "previews"
                    / f"{image_type}_{raw_path.stem}_{step}.npy"
                )

                if cache.exists():
                    bg_image = np.load(cache)

                else:
                    # subindices reads every Nth page
                    movie = cm.load(str(raw_path), subindices=slice(None, None, step))

                    bg_image = (
                        cm.local_correlations(movie, swap_dim=False)
                        if image_type == "corr"
                        else movie.mean(axis=0)
                    )
                    bg_image[np.isnan(bg_image)] = 0

                    cache.parent.mkdir(exist_ok=True)
                    np.save(cache, bg_image)

                lo, hi = float(np.quantile(bg_image, 0.01)), float(
                    np.quantile(bg_image, 0.99)
                )

                def apply():
                    img.data = dict(image=[bg_image])
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
        downsample_type: str = "average",
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
        - `downsample_type`: How to drop frames ("average" or "skip")

        **ALERT:** "average" (the default) reads every frame and averages them down.
        The result looks less noisy, but it smooths away residual motion. "skip" reads
        only the frames it keeps, which is much faster over the network, but it is noisier.

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

        # Check downsample_type input
        assert downsample_type in ("average", "skip"), (
            "The parameter 'downsample_type' must be 'average' or 'skip', "
            f"but instead got {downsample_type!r}."
        )

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

        self.movies[movie_types].maybe_update(downsample_ratio, downsample_type).play(
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

    @record_call
    def save_preview_movie(
        self,
        *,
        grid: list[str] = ["raw", "mcor"],
        downsample_ratio: float = 0.03,
        downsample_type: str = "average",
        save_folder: str = r"./movies",
        fr: float = 30,
        q_min: float = 0.0,
        q_max: float = 99.5,
        codec: str = "MJPG",
        extension: str = "avi",
    ) -> Path:
        """
        Save a preview video of this group's movies, side by side

        **USAGE**
        ```python
            group = db.groups[some_index]
            group.save_preview_movie()
            group.save_preview_movie(grid=["raw"], downsample_ratio=0.1)
        ```

        **PARAMETERS**
        *What goes in the movie*
        - `grid`: which movies to show, one or two of "raw", "test", "mcor"
        - `downsample_ratio`: fraction of frames to keep
        - `downsample_type`: how to drop frames ("average" or "skip")

        **ALERT:** "average" (the default) reads every frame and averages them
        down, which looks less noisy but smooths away residual motion. "skip"
        reads only the frames it keeps, which is much faster over the network,
        but noisier.

        *How it is saved*
        - `save_folder`: `"."` is the main_folder (r is to use \\ in the path)
        - `fr`: how fast to play the video (frames/s)
        - `q_min`, `q_max`: percentiles shown as black and white
        - `codec`, `extension`: how to encode

        **EXAMPLES**
        - Every raw movie of the group, one after the other
        ```python
        group.save_preview_movie(grid=["raw"])
        ```

        - Raw on the left, motion corrected on the right, to compare them
        ```python
        group.save_preview_movie(grid=["raw", "mcor"])
        ```
        """
        assert len(grid) == 1 or (
            len(grid) == 2
            and MovieType.RAW.value in grid
            and (MovieType.MCOR.value in grid or MovieType.TEST.value in grid)
        ), """The parameter 'grid' must be one of the following:
                ['raw'], ['mcor'], ['test'],
                ['raw', 'test'], ['raw', 'mcor'],
                ['test', 'raw'], ['mcor', 'raw']"""

        assert downsample_type in ("average", "skip"), (
            "The parameter 'downsample_type' must be 'average' or 'skip', "
            f"but instead got {downsample_type!r}."
        )

        movie_types = tuple(MovieType(name) for name in grid)

        folder = (
            (self.db.main_folder / save_folder).resolve()
            if save_folder[0] == "."
            else Path(save_folder).resolve()
        )
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / (
            f"Group_{self.group_id}_{self.experiments.iloc[0]['exp_name']}"
            f"_{'_'.join(t.value for t in movie_types)}.{extension}"
        )

        logger.info(f"Saving movie to {path}")

        movie = self._load_preview_movie(
            movie_types, downsample_ratio, downsample_type
        )

        _write_preview_movie(path, movie, fr=fr, q_min=q_min, q_max=q_max, codec=codec)

        # outputs stores paths relative to main_folder, so a save_folder
        # somewhere else cannot be recorded.
        if path.is_relative_to(self.db.main_folder):
            self.add_output_file(path)

        self.set_output({"grid": list(grid), "file": path.name})
        logger.info(f"{CHECK} Wrote {len(movie)} frames.")

        return path

    def _load_preview_movie(
        self,
        movie_types: tuple[MovieType, ...],
        downsample_ratio: float,
        downsample_type: str,
    ) -> cm.movie:
        """
        Load and downsample the movies for `movie_types`, side by side.

        Frames go along time and the types along width, so a two-type grid is
        one movie with the panels next to each other.
        """

        movie_chains = []

        raw_paths: list[Path] = []
        test_paths: list[Path] = []

        # Raises if the test files are missing, before anything is loaded
        if MovieType.TEST in movie_types:
            raw_paths, test_paths = self._latest_test_movies()

        for movie_type in movie_types:
            # Get the movie_paths
            if movie_type == MovieType.TEST:
                movie_paths = test_paths

            elif MovieType.TEST in movie_types:
                # It must be raw in this case
                movie_paths = raw_paths

            # If there is no test always include all files
            elif movie_type == MovieType.RAW:
                movie_paths = [
                    self.db.main_folder / path
                    for path in self.acquisitions["raw_path"]
                ]

            elif movie_type == MovieType.MCOR:
                movie_paths = [
                    self.db.main_folder / path
                    for path in self.mcor_files["mcor_path"]
                ]

            # There must be at least one movie
            # TODO: Gather movie_paths for every type first, so it fails
            #       before having spent time loading anything.
            assert movie_paths, f"Didn't find any {movie_type.value} files."

            logger.info(
                f"Adding {len(movie_paths)} {movie_type.value} files to the movie."
            )

            # "skip" reads only the frames it keeps, rather than reading a whole
            # movie to average it away. Same number of frames out either way.
            movie_chain = None
            for path in tqdm(movie_paths, desc=f"Loading {movie_type.value} movies"):
                if downsample_type == "skip":
                    movie = cm.load(
                        path, subindices=_frames_to_keep(path, downsample_ratio)
                    )

                    # Reading a single frame drops the time axis, and the movies
                    # are stacked along it. Happens whenever the ratio is small
                    # enough to keep one frame per file.
                    if movie.ndim == 2:
                        movie = movie[np.newaxis]

                else:
                    movie = cm.load(path).resize(1, 1, downsample_ratio)

                movie_chain = (
                    movie
                    if movie_chain is None
                    else cm.concatenate([movie_chain, movie], axis=0)
                )

                logger.info(f"  {path}")

            movie_chains.append(movie_chain)

        movie_chain = cm.concatenate(movie_chains, axis=2)

        return movie_chain

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

        # Record the values actually used (after GUI overrides, is_test
        # adjustments, and clamping) so the run is reproducible from
        # parameters_used, even if it fails below.
        self.update_parameters_used(
            {
                "first_acq": first_acq,
                "step_acq": step_acq,
                "last_acq": last_acq,
                "max_deviation_um": max_deviation_um,
                "max_shift_um": max_shift_um,
                "overlap_um": overlap_um,
                "strides_um": strides_um,
            }
        )

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

        n_processes = _worker_count()
        logger.info(f"Using {n_processes} workers.")
        self.update_parameters_used({"n_processes": n_processes})

        _, dview, _ = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=n_processes, single_thread=False
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
            # Record what caiman wrote so play_movie can find it after a restart
            self.set_output({"mmap_names": [Path(p).name for p in self.mc.mmap_file]})
            return

        logger.info("Saving mcor files...")

        insertion_query = """
            INSERT OR REPLACE INTO mcor_files
                ( acq_id
                , mcor_path
                , source
                , last_updated_by
                ) VALUES (?, ?, ?, ?);
        """

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

            # Threshold to switch to BigTIFF, checked against the size of the
            # file actually being saved (since raw/mcor have different sizes).
            # This needs to be checked because tifffile does not complain, it
            # just writes a file that cannot be read back.
            big = mc.nbytes > BIGTIFF_BYTES

            if big:
                logger.info(f"{mc.nbytes / 2**30:.1f} GB, saving as BigTIFF.")

            # Saving TIFFs directly because caiman saves them as 64-bit
            with tifffile.TiffWriter(mcor_path, bigtiff=big) as tif:
                tif.write(
                    [mc[i].copy() for i in range(mc.shape[0])],
                    shape=mc[0].shape,
                    dtype=mc.dtype,
                )

            # Record each file separately to not lock the DB.
            with self.db.con:
                self.db.con.execute(
                    insertion_query,
                    [
                        acq_id,
                        mcor_path.relative_to(self.db.main_folder).as_posix(),
                        McorSource.CAIMAN.value,
                        self.current_call_id,
                    ],
                )

        # Reset mcor DataFrames
        self._mcor_files = None
        self.db._mcor_files = None

    @record_call
    def add_mcor_files(
        self,
        *,
        source: str = "caiman",
        overwrite: bool = False,
        _change_mcor_group: bool = False,
    ) -> None:
        """
        Add motion corrected files not created by `run_motion_correction`

        **USAGE**
        ```python
            group = db.groups[some_index]
            group.add_mcor_files()
        ```

        **PARAMETERS**
        - `source`: which tool made the files ("caiman" or "patchwarp")
        - `overwrite`: drop the files the group has now and use these instead

        **IMPORTANT WARNING**
        A group's mcor files must all come from the same run, so either:
            - This function does not change `group.mcor_files` at all;
            - Or the whole `group.mcor_files` table is overwritten.

        The first case happens when:
            - `overwrite=False` and `group.mcor_files` is non-empty;
            - or `overwrite=True` but no valid mcor files where found.

        All other cases lead to dropping all `mcor_files` entries for this
        group and the addition of the valid mcor files that where found.

        Each experiment must be on only one "motion correction" group, and
        this function SHOULD ONLY BE CALLED ON THAT GROUP. Calling it from
        another group raises an error, and says which group to use instead.

        **HOW IT WORKS**
        For each raw file `raw/NAME.tif` in the group it expects a
        ```
            caiman     processed/mcor/NAME_mcor.tif
            patchwarp  processed/patchwarp/post_warp/NAME_corrected_warped.tif
        ```

        An mcor file is only added if its frame size and frame count match the
        raw file's. Anything missing or mismatched is reported and left out, so
        a group can end up with files for only some of its acquisitions.
        """

        # NOTE: _change_mcor_group is deliberately not documented above.
        #       See _check_mcor_group for more details.
        try:
            mcor_source = McorSource(source)

        except ValueError:
            raise ValueError(
                f"source must be one of {[s.value for s in McorSource]},"
                f"but instead got {source!r}."
            )

        folder, suffix = MCOR_LAYOUT[mcor_source]

        acquisitions = self.acquisitions
        experiments = self.experiments
        existing = self.mcor_files

        # Nothing is read from disk in this case, which matters over the network
        if len(existing) and not overwrite:
            self.add_flag(McorFlag.ALREADY_HAS_FILES)

            logger.warning(f"{self!r} has mcor files, so nothing was added. {CROSS}")
            logger.warning("Pass overwrite=True to drop old files and add new ones.")
            self.set_output({"source": mcor_source.value, "added": 0})
            return

        rows: list[list] = []
        report: dict[str, list[int]] = {
            "added": [],
            "file_not_found": [],
            "wrong_shape": [],
            "wrong_frame_count": [],
            "unreadable": [],
        }

        for acq_id, acquisition in tqdm(
            acquisitions.iterrows(), desc="Checking files", total=len(acquisitions)
        ):
            raw_path = Path(acquisition["raw_path"])
            mcor_folder = self.db.main_folder / raw_path.parent.parent / folder

            mcor_path = mcor_folder / (raw_path.stem + suffix)
            experiment = experiments.loc[acquisition["exp_id"]]
            expected = (int(experiment["height_px"]), int(experiment["width_px"]))

            try:
                shape, frames = _tiff_shape(mcor_path, int(experiment["frame_count"]))

            except FileNotFoundError:
                report["file_not_found"].append(acq_id)
                continue

            except Exception as error:
                logger.warning(f"  Could not read '{mcor_path.name}': {error}")
                report["unreadable"].append(acq_id)
                continue

            if shape != expected:
                logger.warning(
                    f"  '{mcor_path.name}' is {shape[0]}x{shape[1]}, "
                    f"expected {expected[0]}x{expected[1]}."
                )
                report["wrong_shape"].append(acq_id)
                continue

            if frames != int(experiment["frame_count"]):
                logger.warning(
                    f"  '{mcor_path.name}' has {frames} frames, "
                    f"expected {experiment['frame_count']}."
                )
                report["wrong_frame_count"].append(acq_id)
                continue

            rows.append(
                [
                    acq_id,
                    mcor_path.relative_to(self.db.main_folder).as_posix(),
                    mcor_source.value,
                    self.current_call_id,
                ]
            )
            report["added"].append(acq_id)

        # --- Report and flag before touching the database --- #

        for name, flag in (
            ("file_not_found", McorFlag.FILE_NOT_FOUND),
            ("wrong_shape", McorFlag.WRONG_SHAPE),
            ("wrong_frame_count", McorFlag.WRONG_FRAME_COUNT),
            ("unreadable", McorFlag.UNREADABLE),
        ):
            if report[name]:
                self.add_flag(flag)
                logger.warning(f"{name}: {len(report[name])} acquisitions")

        self.set_output(
            {
                "source": mcor_source.value,
                "replaced": len(existing) if rows else 0,
                **{name: len(ids) for name, ids in report.items()},
                "added_acq_ids": [int(a) for a in report["added"]],
            },
        )

        # Nothing is dropped unless there is something to put in its place
        if not rows:
            logger.info(f"Found no valid files, so nothing was updated. {CROSS}")
            return

        dropped = [int(acq_id) for acq_id in existing.index]

        if dropped:
            self._check_mcor_group(dropped, _change_mcor_group)
            self.add_flag(McorFlag.REPLACED_EXISTING)

        with self.db.con as con:
            if dropped:
                con.execute(
                    f"DELETE FROM mcor_files WHERE acq_id IN "
                    f"({','.join('?' * len(dropped))});",
                    dropped,
                )

            con.executemany(
                """
                INSERT INTO mcor_files
                    ( acq_id
                    , mcor_path
                    , source
                    , last_updated_by
                    ) VALUES (?, ?, ?, ?);
                """,
                rows,
            )

        if dropped:
            logger.info(f"Dropped {len(dropped)} mcor files from a previous run.")

        logger.info(f"{CHECK} Added {len(rows)} {mcor_source.value} files.")

        # Reset mcor DataFrames
        self._mcor_files = None
        self.db._mcor_files = None

    @record_call
    def approve_mcor_files(self, *, exclude_acq_ids: Sequence[int] = ()) -> None:
        """
        Approve this group's motion corrected files for further processing/analysis

        **USAGE**
        ```python
            group = db.groups[some_index]
            group.approve_mcor_files()
            group.approve_mcor_files(exclude_acq_ids=[7, 12])
        ```

        **PARAMETERS**
        - `exclude_acq_ids`: acquisitions to leave out

        **IMPORTANT**
        - This call overwrites past calls, so ALL `acq_ids` in this group
        that are not in the exclusion list will be marked as approved.
        - Analysis functions reference  `group.approved_mcor_files`, so
        anything left out here is skipped from then on.
        """

        existing = self.mcor_files

        if not len(existing):
            self.add_flag(McorFlag.NOTHING_TO_APPROVE)

            logger.warning(f"{self!r} has no mcor files to approve. {CROSS}")
            self.set_output({"approved": [], "excluded": []})
            return

        acq_ids = [int(acq_id) for acq_id in existing.index]
        excluded = {int(acq_id) for acq_id in exclude_acq_ids}

        # Raises if there are exclusions that are not actually part of this
        # group mcors, since this usually implies a mistake upstream.
        missing = sorted(excluded - set(acq_ids))

        if missing:
            raise ValueError(
                f"{missing} are not acquisitions with mcor files in {self!r}. "
                f"'approved' flag was not changed for any file."
            )

        self._check_mcor_group(acq_ids)

        with self.db.con as con:
            con.executemany(
                "UPDATE mcor_files SET approved = ? WHERE acq_id = ?;",
                [(acq_id not in excluded, acq_id) for acq_id in acq_ids],
            )

        approved = [acq_id for acq_id in acq_ids if acq_id not in excluded]

        if excluded:
            self.add_flag(McorFlag.SOME_NOT_APPROVED)
            logger.warning(f"Left {len(excluded)} out: {sorted(excluded)}")

        self.set_output({"approved": approved, "excluded": sorted(excluded)})
        logger.info(f"{CHECK} Approved {len(approved)} of {len(acq_ids)} mcor files.")

        # Reset mcor DataFrames
        self._mcor_files = None
        self.db._mcor_files = None

    def _check_mcor_group(
        self, acq_ids: list[int], change_mcor_group: bool = False
    ) -> None:
        """
        Stop or warn, depending on what changing these mcor files disturbs.

        Each experiment is motion corrected by exactly one group: its own
        singleton, or the group made for a session that got split into several
        experiments. Every other group holding that experiment is for analysis.

        This method stops you from changing records from a group that is
        not the "motion correcting" one (unless you pass `change_mcor_group`).
        That parameter is private in the caller because this should be a rare
        occasion. But, either way, an ownership change is visible via flags.

        Shared by every method that writes `mcor_files`, so it says nothing
        about what the caller is about to change.
        """

        marks = ",".join("?" * len(acq_ids))

        owners = self.db.con.execute(
            f"""
            SELECT DISTINCT mc.group_id
                FROM mcor_files   AS m
                JOIN method_calls AS mc ON mc.method_call_id = m.last_updated_by
                WHERE m.acq_id    IN ({marks});
            """,
            acq_ids,
        ).fetchall()

        would_take_from = [
            row["group_id"] for row in owners if row["group_id"] != self.group_id
        ]

        # Flagged whether or not it is allowed, so "which groups had their mcor
        # files taken over?" stays answerable from method_calls afterwards.
        if would_take_from:
            self.add_flag(McorFlag.OWNED_BY_OTHER_GROUP)

            if not change_mcor_group:
                raise RuntimeError(
                    f"These files were motion corrected by group(s) {would_take_from}, "
                    f"not by {self!r}. Call this from those group(s) instead."
                )

            logger.warning(
                f"Taking these files over from group(s) {would_take_from}. {CROSS}"
            )

        users = self.db.con.execute(
            f"""
            SELECT DISTINCT ge.group_id
                FROM mcor_files        AS m
                JOIN acquisitions      AS a  ON a.acq_id = m.acq_id
                JOIN group_experiments AS ge ON ge.exp_id = a.exp_id
                WHERE m.acq_id IN ({marks}) AND ge.group_id != ?;
            """,
            [*acq_ids, self.group_id],
        ).fetchall()

        if users:
            self.add_flag(McorFlag.SHARED_WITH_OTHER_GROUPS)
            logger.warning(
                f"Groups {[row['group_id'] for row in users]} also use these "
                "files and will see this change."
            )

    # ----------------------------------------------------------------------- #
    # Data Analysis
    # ----------------------------------------------------------------------- #

    def _mcors(self, only_approved: bool) -> pd.DataFrame:
        """
        The mcor files to use when making a movie.

        `only_approved=False` is for computing before the motion correction has
        been looked at, while the files are still in the network cache.
        """
        return self.approved_mcor_files if only_approved else self.mcor_files

    def _z_score_frames(
        self, photobleach_window_s: float, only_approved: bool = True
    ) -> tuple[int, int, float]:
        """
        Range of frames that can be compared across the group.

        Returns `(first, last, frame_rate)`, where:
        - `first` and `last` are indices of frames relative to odor onsets;
        - the range `[first, last]` is contained in all acquisitions;
        - the interval above is the largest such interval;
        - `frame_rate` is the mean frame rate across acquisitions.

        Only uses acquisitions associated with an odor.
        """

        usable = self.acquisitions.loc[self._mcors(only_approved).index]

        if not len(usable):
            raise RuntimeError(
                f"{self!r} has no approved mcor files. Check the previews and "
                "run 'approve_mcor_files()', or pass only_approved=False."
                if only_approved
                else f"{self!r} has no mcor files."
            )

        # An acquisition might not be associated with an odor (no `odor_start`).
        # We drop them to avoid NaNs in the np.max and np.min below.
        acquisitions = usable[usable["odor_start"].notna()]

        if not len(acquisitions):
            raise RuntimeError("Found no mcor files with an odor onset.")

        elif len(acquisitions) < len(usable):
            no_odor = len(usable) - len(acquisitions)
            logger.warning(f"{no_odor} acquisitions don't have odor onsets!")

        experiments = self.experiments.loc[acquisitions["exp_id"]]

        frame_rates = experiments["frame_rate"].to_numpy()
        frame_counts = experiments["frame_count"].to_numpy()

        onset_frames = np.rint(
            (acquisitions["odor_start"] - acquisitions["acq_start"])
            .dt.total_seconds()
            .to_numpy()
            * frame_rates
        )

        frame_rate = float(frame_rates.mean())
        photobleach_frame = round(photobleach_window_s * frame_rate)

        first = int(np.max(photobleach_frame - onset_frames))
        last = int(np.min(frame_counts - 1 - onset_frames))

        return first, last, frame_rate

    def z_score_acquisition(
        self,
        *,
        acq_id: int,
        photobleach_window_s: float = 1.0,
        only_approved: bool = True,
    ) -> np.ndarray:
        """
        Z-score one acquisition against its own pre-odor baseline

        **USAGE**
        ```python
            group = db.groups[some_index]
            z = group.z_score_acquisition(acq_id=some_acq_id)
        ```

        Each pixel is compared to how much it varied before the odor
        presentation and after the photobleaching window. Frames indices are
        relative to the acquisition odor onset, and the same range is used
        for every acquisition of the group, so the results can be averaged.

        Frames are not smoothed over time, differently from the MATLAB script.
        (Averaging leaves far fewer frames to measure the noise from.)
        """
        usable = self._mcors(only_approved)

        # Check here because .loc would only say "KeyError: <acq_id>"
        if acq_id not in usable.index:
            added = {"approved " if only_approved else ""}
            raise KeyError(f"Acquisition {acq_id} has no {added}mcor file in {self!r}.")

        # Redoes the computation for every acquisition, but that is fast.
        first, last, frame_rate = self._z_score_frames(
            photobleach_window_s, only_approved
        )

        onset_delay = cast(
            datetime, self.acquisitions.loc[acq_id, "odor_start"]
        ) - cast(datetime, self.acquisitions.loc[acq_id, "acq_start"])

        onset_frame = int(round(onset_delay.total_seconds() * frame_rate))

        path = self.db.main_folder / cast(str, usable.loc[acq_id, "mcor_path"])

        with tifffile.TiffFile(path) as tif:
            movie = tif.asarray(
                key=slice(onset_frame + first, onset_frame + last + 1)
            ).astype(np.float32)

        # The odor starts at index -first, so everything before it is baseline
        baseline = movie[:-first]
        spread = baseline.std(axis=0, ddof=1)

        z_score = (movie - baseline.mean(axis=0)) / np.where(spread > 0, spread, 1)

        return np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)

    def z_score_average_movies(
        self, *, photobleach_window_s: float = 1.0, only_approved: bool = True
    ) -> dict[tuple[int, int, str], np.ndarray]:
        """
        Normalized average z-scores over the acquisitions of each
        program, odor and outcome

        **USAGE**
        ```python
            group = db.groups[some_index]
            z_score_movies = group.z_score_average_movies()
            z_score_movies[(program_id, odor_id, outcome)]
        ```

        Every acquisition is z-scored against its own baseline, and then we
        compute the normalized average across acquisitions (`average * sqrt(n)`).
        We use this normalization so that conditions with different number of
        acquisitions have similar noise levels (~1). Pass `only_approved=False`
        to include mcor files that have not been approved yet.

        ALERT: results holds one movie per condition, so it can be too RAM
        intensive for large experiments.
        """

        trials = self.trials[
            self.trials["acq_id"].isin(self._mcors(only_approved).index)
        ]
        conditions: dict[tuple[int, int, str], np.ndarray] = {}

        for key, rows in trials.groupby(["program_id", "odor_id", "outcome"]):
            # groupby keys come back as a tuple of Hashable
            condition = cast(tuple[int, int, str], key)
            acq_ids = rows["acq_id"].astype(int).unique()
            total = None

            for acq_id in tqdm(acq_ids, desc=f"{condition}"):
                z_score = self.z_score_acquisition(
                    acq_id=acq_id,
                    photobleach_window_s=photobleach_window_s,
                    only_approved=only_approved,
                )
                total = z_score if total is None else total + z_score

            # Sum over sqrt(n), not mean, to keep one noise scale for them all
            conditions[condition] = total / np.sqrt(len(acq_ids))

        return conditions

    @record_call
    def save_z_score_movies(
        self,
        *,
        photobleach_window_s: float = 1.0,
        smoothing_s: float = 0.2,
        only_approved: bool = True,
        save_folder: str = r"./movies",
        display_range: float = 5.0,
        codec: str = "mp4v",
        extension: str = "mp4",
    ) -> list[Path]:
        """
        Save one z-score movie per program, odor and outcome

        **USAGE**
        ```python
            group = db.groups[some_index]
            group.save_z_score_movies()
        ```

        **PARAMETERS**
        - `photobleach_window_s`: how much to drop from the start of each movie
        - `smoothing_s`: average this many seconds together, `0` for none
        - `only_approved`: `False` for all mcor files, `True` for approved ones.
        - `save_folder`: `"."` is the main_folder (r is to use \\ in the path)
        - `display_range`: z-scores shown, from `-display_range` to `+display_range`
        - `codec`, `extension`: how to encode (see ALERT below)

        The movie plays at the rate it was recorded at, so what you see takes as
        long as the experiment did. Red marks the odor presentation.

        This function is similar to `z_score_average_movies`, but only one movie
        is held at a time, so it works on experiments too large to keep in RAM.

        Averaging in time (`smoothing_s`) is to reduce white noise that varies
        in a smaller time scale than the signal responses. Try to use the
        smallest acceptable value. We use this instead of spatial averaging
        to preserve structures that are few pixels wide (e.g. neurites). Output
        is rescaled afterwards so the noise level stays around 1 and
        `display_range` keeps its meaning.

        ALERT: the range is fixed on purpose to make movies comparable. It is
        also symmetric so that saturation matches absolute value approximately.

        ALERT: H.264 cannot be written by most OpenCV builds, so the default is
        MPEG-4 part 2 ('mp4v'). Use `codec="MJPG", extension="avi"` to match
        `play_movie`. VLC opens either one on any platform.
        """
        folder = (
            (self.db.main_folder / save_folder).resolve()
            if save_folder[0] == "."
            else Path(save_folder).resolve()
        )
        folder.mkdir(parents=True, exist_ok=True)

        first_frame, _, frame_rate = self._z_score_frames(
            photobleach_window_s, only_approved
        )

        # Add a flag for the user that the video was made before approval
        provisional = "" if only_approved else "_provisional"
        exp_name = self.experiments.iloc[0]["exp_name"]

        # Odd sized window, so it has a middle frame and the odor keeps its place
        window = max(1, round(smoothing_s * frame_rate))
        window += 1 - window % 2
        onset = -first_frame - window // 2

        self.update_parameters_used(
            {"frame_rate": frame_rate, "smoothing_frames": window}
        )

        # outputs stores paths relative to main_folder so the database still
        # works from another machine, so anywhere else cannot be recorded.
        recordable = folder.is_relative_to(self.db.main_folder)

        if not recordable:
            logger.warning(
                f"'{folder}' is outside the main folder, so the movies are "
                "saved but not recorded in the database."
            )

        trials = self.trials[
            self.trials["acq_id"].isin(self._mcors(only_approved).index)
        ]
        saved = []
        written: list[Object] = []

        for key, rows in trials.groupby(["program_id", "odor_id", "outcome"]):
            # groupby keys come back as a tuple of Hashable
            program_id, odor_id, outcome = cast(tuple[int, int, str], key)

            acq_ids = rows["acq_id"].astype(int).unique()
            description = (
                f"Program ID: {program_id}, "
                f"Odor ID: {odor_id}, "
                f"Outcome: {outcome}"
            )

            total = None
            for acq_id in tqdm(acq_ids, desc=description):
                z_score = self.z_score_acquisition(
                    acq_id=acq_id,
                    photobleach_window_s=photobleach_window_s,
                    only_approved=only_approved,
                )
                total = z_score if total is None else total + z_score

            # Sum over sqrt(n), not mean, to keep one noise scale for them all
            total /= np.sqrt(len(acq_ids))

            if window > 1:
                total = _moving_mean(total, window)

                # Averaging shrank the noise; put it back on the scale
                # display_range is written in, measured rather than assumed so
                # it holds however correlated the frames turn out to be.
                total /= total[:onset].std()

            path = folder / (
                f"Group_{self.group_id}_{exp_name}_z_score"
                f"_program_{program_id}_odor_{odor_id}"
                f"_outcome_{outcome.replace(' ', '_')}{provisional}.{extension}"
            )

            odor_s = (rows["odor_end"] - rows["odor_start"]).dt.total_seconds()
            _save_movie(
                path,
                total,
                odor_frames=range(onset, onset + round(odor_s.min() * frame_rate)),
                display_range=display_range,
                codec=codec,
                fr=frame_rate,
            )

            saved.append(path)

            if recordable:
                self.add_output_file(path)

            # int() because these come from pandas as numpy scalars, which
            # json.dumps refuses -- and record_call dumps on the way out, so
            # it would fail after the movies were already written.
            written.append(
                {
                    "program_id": int(program_id),
                    "odor_id": int(odor_id),
                    "outcome": str(outcome),
                    "acquisitions": len(acq_ids),
                    "file": path.name,
                }
            )

            # Let this one go before the next average is built
            del total, z_score

        self.set_output({"movies": written})

        return saved


def _moving_mean(movie: np.ndarray, window: int) -> np.ndarray:
    """
    Centred moving mean along time, keeping only frames with a whole window.

    Returns `len(movie) - window + 1` frames, so frame `i` of the result is
    centred on frame `i + window // 2` of the input.
    """
    sums = np.cumsum(np.insert(movie, 0, 0.0, axis=0), axis=0)

    return (sums[window:] - sums[:-window]) / window


def _save_movie(
    path: Path,
    time_series: np.ndarray,
    *,
    odor_frames: range,
    fr: float,
    display_range: float = 5.0,
    codec: str = "mp4v",
) -> None:
    """
    Render a time-series into a movie (for visualization purposes only).

    Odor presentation frames are marked with a red dot in the corner. Colormap
    is fixed such that blue is negative, white is zero, and red is positive.
    Use a fixed `display_range` to make movies comparable.
    """

    # Kept deliberately separate from `play_movie` because that one reads files
    # and compares processing steps and this just saves a preprocessed time-series.
    # Costs some repetition but avoid interdependencies between processing paths.

    # An 8-bit lookup table beats mapping colours frame by frame.
    # coolwarm is the same colormap as the MATLAB scripts used.
    colors = colormaps["coolwarm"](np.linspace(0, 1, 256))
    lut = (colors[:, 2::-1] * 255).astype(np.uint8)  # RGBA -> BGR

    frames, height, width = time_series.shape
    radius = max(4, height // 60)

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fr, (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not write '{path}' with codec {codec!r}.")

    try:
        for frame in range(frames):
            # Frame rescaled to be between 0-255
            scaled = (time_series[frame] + display_range) * (255 / (2 * display_range))
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)

            picture = lut[scaled]

            if frame in odor_frames:
                cv2.circle(picture, (2 * radius, 2 * radius), radius, (0, 0, 255), -1)

            writer.write(picture)

    finally:
        writer.release()

    logger.info(f"Wrote {path}")


def _frames_to_keep(path: Path, downsample_ratio: float) -> np.ndarray:
    """
    Which frames of `path` to read to downsample it by `downsample_ratio`.

    Matches how many frames caiman's `resize` would leave, so that skipping and
    averaging produce movies of the same length. It keeps `int(ratio * frames)`
    of them (truncated, and never fewer than one).

    NOTE: this must be an array. `caiman.load` reads a *list* of subindices as
    one index per dimension, so a list would be taken as [time, y, x].
    """
    _, frames = get_file_size(path)
    frames = cast(int, frames)

    keep = max(1, min(int(frames), int(downsample_ratio * frames)))

    return np.linspace(0, frames - 1, keep).round().astype(int)


def _write_preview_movie(
    path: Path,
    movie: np.ndarray,
    *,
    fr: float,
    q_min: float = 0.0,
    q_max: float = 99.5,
    codec: str = "MJPG",
) -> None:
    """
    Write a grayscale preview of `movie` (time, height, width).

    Black and white are put at the `q_min` and `q_max` percentiles of the whole
    movie rather than at its extremes, so one bright speck cannot flatten
    everything else. The scale is therefore relative to this movie and says
    nothing across movies, which is the opposite of the z-score renderer.
    """

    # Percentiles over every frame, so brightness does not drift as it plays.
    # On the whole array at once because it is already downsampled and in RAM.
    low, high = np.percentile(movie, [q_min, q_max])
    spread = max(float(high - low), 1e-9)

    frames, height, width = movie.shape

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fr, (width, height), isColor=False
    )

    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not write '{path}' with codec {codec!r}.")

    try:
        for frame in tqdm(range(frames), desc="Writing movie"):
            # uint8 before anything is drawn on it:
            # OpenCV 5 refuses to draw on float images,
            #   (caiman does this backwards and, thus, it crashes)
            picture = np.clip((movie[frame] - low) * (255 / spread), 0, 255)

            writer.write(picture.astype(np.uint8))

    finally:
        writer.release()


def _worker_count() -> int:
    """
    How many processes to hand caiman, which caiman gets wrong on its own.

    Caiman default (`n_processes=None`) uses `os.cpu_count()`, which reports
    the whole machine: a 12-core slurm job on a 28-core node measured 28, so it
    would start 27 workers on 12 cores and split the job's memory between them.

    ASSUMES Linux is the cluster and everything else is someone's own machine.
    A cluster job owns its cores, so it uses all of them, and only Linux has
    `sched_getaffinity` to report what the scheduler actually gave. A personal
    machine keeps one core free for the user.
    """

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    # cpu_count is documented to return None when it cannot tell
    return max(1, (os.cpu_count() or 1) - 1)


def _tiff_shape(path: Path, expected_frames: int) -> tuple[tuple[int, int], int]:
    """
    Frame size and frame count of a TIFF, reading from disk as little as possible.

    Returns `((height, width), frames)`. Raises `FileNotFoundError` if the file
    is not there, which is how the caller tells missing apart from unreadable.
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]

        # Uses tags as in add_experiment
        height = int(page.tags["ImageLength"].value)
        width = int(page.tags["ImageWidth"].value)

        # Pixel data is usually most of the TIFF size, so you can cheaply
        # get the frame count by using the approximation
        #       size ~ size of frame * number of frames
        # The size comes off the open file, so it costs no extra round trip.
        frame_bytes = height * width * page.dtype.itemsize * page.samplesperpixel
        frames = tif.filehandle.size // frame_bytes

        # Fallback if the quick test fails. Counting properly walks one
        # directory per frame, which can be slow over the network (patchwarp
        # files carry a ScanImage header on every page).
        if frames != expected_frames:
            logger.info(f"  Counting pages of '{path.name}' (this is slow)...")
            frames = len(tif.pages)

    return (height, width), int(frames)


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

    def maybe_update(self, downsample_ratio, downsample_type="average") -> cm.movie:
        if self.movie is not None:
            logger.info("Using cached movie...")
            return self.movie

        logger.info("Updating movie...")

        self.movie = self.owner._load_preview_movie(
            self.types, downsample_ratio, downsample_type
        )

        return self.movie


#     def play(self) -> None:
#         # TODO: - Use Caiman function as blueprint
#         #       - Add movie type, total time, current time, and speed (e.g. x2) to label
#         #       - Remove frame number from label
#         #       - Possibly add total time to config and remove fr in [play.load]
#         return
