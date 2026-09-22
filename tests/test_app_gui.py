"""
The entry app's widgets, offscreen: the window opens on a fresh database,
the registries write through the worker, the table browser reads, and the
recorder writes a sync file where `submit` will look for it.
"""

import os
import time

import h5py
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Needs Qt and the system libraries it loads (libEGL and friends): skipped
# where either is missing, which is what leaves the rest of the suite runnable
# on a bare machine.
try:
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

except ImportError as error:  # a missing shared library is not a missing module
    pytest.skip(f"Qt cannot load here: {error}", allow_module_level=True)

from odyn import Database  # noqa: E402
from odyn.app.draft import Draft  # noqa: E402
from odyn.app.models import FrameModel  # noqa: E402
from odyn.app.pane_recorder import RecorderPane, recording_path  # noqa: E402
from odyn.app.pane_tables import filter_frame  # noqa: E402
from odyn.app.recorder import SimulatedSource  # noqa: E402
from odyn.app.window import MainWindow  # noqa: E402
from odyn.app.worker import DatabaseWorker  # noqa: E402

TIMEOUT_MS = 20_000


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(condition, timeout_ms: int = TIMEOUT_MS) -> None:
    """Spin the event loop until `condition()` holds."""
    deadline = time.monotonic() + timeout_ms / 1000

    while not condition():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)

        if time.monotonic() > deadline:
            raise TimeoutError("the app did not get there in time")


def settle(context) -> None:
    """
    Wait for every queued job, and for the jobs their results queue (a write
    announces `changed`, which reloads the panes), until nothing is pending.
    """
    for _ in range(10):
        done = []
        job = context.worker.run(lambda db: True, "settle")
        job.done.connect(lambda _: done.append(True))
        job.failed.connect(lambda m: done.append(m))
        wait_until(lambda: bool(done))
        assert done == [True], done
        QApplication.processEvents()

        if context.worker._jobs.empty():
            return

    raise AssertionError("the worker never ran out of jobs")


@pytest.fixture
def window(app, tmp_path):
    Database(tmp_path / "data", can_create=True, user="tester")  # make the file
    worker = DatabaseWorker(tmp_path / "data", None, user="tester")
    opened = []
    worker.opened.connect(lambda: opened.append(True))
    worker.open_failed.connect(lambda m: opened.append(m))
    worker.start()
    wait_until(lambda: bool(opened))
    assert opened == [True], opened

    window = MainWindow(
        worker, drafts=tmp_path / "drafts", recorder_source=SimulatedSource()
    )
    window.show()
    settle(window.context)

    yield window

    window.recorder.close_down()
    worker.stop()
    window.close()


def test_window_opens_with_every_tab(window):
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]

    assert titles == [
        "Users",
        "Session",
        "Recorder",
        "Odors",
        "Panels",
        "Mice",
        "Tables",
        "QC",
    ]
    assert window.tabs.currentWidget() is window.users
    assert "tester" in window.user_label.text()


def test_users_pane_is_a_login_screen(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.add_user(user_name="ana", display_name="Ana Silva")
    db.add_user(user_name="bo", display_name="Bo Li")
    db.add_user(user_name="old", display_name="Old Hand")
    db.retire_user(user_name="old")
    window.context.changed.emit()
    settle(window.context)

    users = window.users
    assert [b.text() for b in users.buttons.values()] == ["Ana Silva", "Bo Li"]
    assert "Click on your name" in users.hint.text()

    users.buttons["ana"].click()
    settle(window.context)
    assert window.context.user == "ana"
    assert users.buttons["ana"].isChecked() and not users.buttons["bo"].isChecked()
    assert "Ana Silva" in users.hint.text() and "ana" in window.user_label.text()
    assert window.tabs.currentWidget() is window.session

    # the worker's database now signs calls as ana
    db.set_mouse(mouse_id=1)  # a reader opens fine beside the worker
    window.mice.form.set_values({"mouse_id": "m2"})
    window.mice._save()
    settle(window.context)
    calls = Database(tmp_path / "data").latest_calls("set_mouse")
    assert calls.iloc[0]["user"] == "ana"


def test_mice_pane_registers_a_mouse(window, tmp_path):
    mice = window.mice
    mice.form.set_values({"mouse_id": "m7", "sex": "F", "dob": "2026-01-01"})
    mice._save()
    settle(window.context)

    assert 7 in Database(tmp_path / "data").mice.index


def test_tables_pane_reads_filters_and_shows_a_row(window):
    tables = window.tables
    tables.list.setCurrentRow(
        list(tables.list.item(i).text() for i in range(tables.list.count())).index(
            "odors"
        )
    )
    settle(window.context)

    model = tables.view.model()
    assert isinstance(model, FrameModel)
    assert model.rowCount() > 20  # the seeded odors

    tables.filter.setText("eugenol")
    assert 0 < tables.view.model().rowCount() < 5

    tables._row_clicked(tables.view.model().index(0, 0))
    assert tables.detail.title.text().startswith("odors: ")
    assert not tables.detail.annotate_box.isVisible()  # odors are not annotated


def test_detail_pane_annotates_a_mouse(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=9, sex="M")
    window.context.changed.emit()
    settle(window.context)

    tables = window.tables
    names = [tables.list.item(i).text() for i in range(tables.list.count())]
    tables.list.setCurrentRow(names.index("mice"))
    settle(window.context)

    tables._row_clicked(tables.view.model().index(0, 0))
    settle(window.context)
    detail = tables.detail
    assert detail.target_type == "mouse" and detail.target_id == 9
    assert detail.form is not None

    detail.form.rows["note"].set_value("lost a whisker")
    detail._save()
    settle(window.context)

    current = Database(tmp_path / "data").annotations_for("mouse")
    assert current.loc[9, "note"] == "lost a whisker"


def test_filter_frame_matches_any_cell():
    import pandas as pd

    frame = pd.DataFrame(
        {"a": ["Eugenol", "hexanal"], "b": [1.0, None]}, index=[10, 11]
    )

    assert list(filter_frame(frame, "eug").index) == [10]
    assert list(filter_frame(frame, "11").index) == [11]
    assert len(filter_frame(frame, "")) == 2


def test_recorder_test_shows_traces_and_writes_nothing(window, tmp_path):
    recorder: RecorderPane = window.recorder
    recorder.test()
    assert recorder.recording is not None and recorder.recording.path is None
    assert not recorder.start_button.isEnabled() and recorder.stop_button.isEnabled()
    wait_until(
        lambda: recorder.recording is not None and recorder.recording.total > 500
    )
    assert recorder.buffer.chunks  # the traces are being drawn
    recorder.stop()

    assert recorder.recording is None and recorder.test_button.isEnabled()
    assert "test" in recorder.status.text()
    assert list(tmp_path.rglob("*.h5")) == []


def test_recorder_records_into_the_session_folder(window, tmp_path):
    draft = Draft.start(tmp_path / "drafts", user="tester").identify(
        mouse_id=7, date="2026-09-22"
    )
    window.session.reload_drafts()
    assert window.recorder.draft is draft or window.recorder.draft.path == draft.path

    draft = window.session.current  # the pane's own copy, loaded from disk
    draft.set_experiment("e1")
    window.context.announce_draft()
    recorder: RecorderPane = window.recorder
    assert recorder.experiment.currentText() == "e1"
    recorder.start()
    wait_until(
        lambda: recorder.recording is not None and recorder.recording.total > 100
    )
    recorder.stop()

    experiment = tmp_path / "data" / "20260922" / "m7" / "e1"
    assert (experiment / "raw").is_dir() and (experiment / "processed").is_dir()
    files = list((experiment / "sync").glob("m7_e1_*.h5"))
    assert len(files) == 1

    with h5py.File(files[0]) as h5:
        assert h5.attrs["mouse"] == "m7" and h5.attrs["experiment_folder"] == "e1"
        assert h5["2pFrameSync"].shape[0] > 100
        assert h5["filter_changes"].shape == (1, 3)

    # the draft now lists the experiment the recorder made
    shown = window.session.current
    assert shown is not None and shown.experiment("e1") is not None
    assert window.session.tabs.count() == 1


def test_recording_path_layout(tmp_path):
    draft = Draft.start(tmp_path, user="x").identify(mouse_id=12, date="2026-03-04")
    path, mouse = recording_path(tmp_path / "main", draft, "e2")

    assert mouse == "m12"
    assert path.parent == tmp_path / "main" / "20260304" / "m12" / "e2" / "sync"
    assert path.name.startswith("m12_e2_")


def test_qc_pane_lists_nothing_for_a_new_user(window):
    qc = window.qc
    settle(window.context)

    assert qc.experiments_view.model().rowCount() == 0
    assert "tester" in qc.who.text() or "ana" in qc.who.text()


def test_worker_runs_jobs_in_order(app, tmp_path):
    Database(tmp_path, can_create=True)
    worker = DatabaseWorker(tmp_path, None, user="tester")
    worker.start()
    results = []

    # (lambdas, not `results.append`: PySide keeps bound methods weakly)
    for n in range(3):
        worker.run(lambda db, n=n: n).done.connect(lambda r: results.append(r))

    failed = []
    worker.run(lambda db: 1 / 0).failed.connect(lambda m: failed.append(m))
    wait_until(lambda: len(results) == 3 and failed)
    worker.stop()

    assert results == [0, 1, 2]
    assert failed[0].startswith("ZeroDivisionError")


def test_panels_table_shows_the_latest_vial_per_position():
    import pandas as pd

    from odyn.app.pane_registry import current_vials

    vials = pd.DataFrame(
        {
            "panel_id": [1, 1, 1, 2, 2],
            "panel_name": ["b", "b", "b", "a", "a"],
            "vial_position": [2, 1, 1, 1, 1],
            "made_on": pd.to_datetime(
                ["2026-07-06", "2026-07-06", "2026-08-06", "2026-08-06", None]
            ),
        },
        index=pd.Index([1, 2, 3, 4, 5], name="vial_id"),
    )

    shown = current_vials(vials)

    # one row per panel and position, the remade vial replacing the old one,
    # panels then positions in order
    assert list(shown.index) == [4, 3, 1]
    assert list(shown["panel_name"]) == ["a", "b", "b"]


def test_recipe_components_follow_the_odor(window):
    panels = window.panels
    form = panels._add_recipe()

    form.delivered.setCurrentIndex(form.delivered.findData(17))  # alpha
    assert [row.odor_id for row in form.components] == [4, 10]
    form.set_targets({4: 44.2, 10: 13.3})
    vial = form.vial()
    assert vial["odor_id"] == 17
    assert [(c["odor_id"], c["liquid_ul"]) for c in vial["components"]] == [
        (4, 150),
        (10, 100),
    ]
    assert vial["solvent_volume_ml"] == pytest.approx(4.75)

    form.delivered.setCurrentIndex(form.delivered.findData(1))  # eugenol, pure
    assert [row.odor_id for row in form.components] == [1]

    form.delivered.setCurrentIndex(form.delivered.findData(0))  # mineral oil
    assert form.components == [] and form.vial()["components"] == []


def test_odors_pane_adds_a_mix_from_its_components(window, tmp_path):
    odors = window.odors
    odors.odor_id.setValue(45)
    odors.name.setText("sigma")
    odors._add_component()
    odors._add_component()
    odors.components[0].setCurrentIndex(odors.components[0].findData(4))
    odors.components[1].setCurrentIndex(odors.components[1].findData(10))
    assert "mix of 2" in odors.kind.text()
    odors._add()
    settle(window.context)

    db = Database(tmp_path / "data")
    assert db.odors.loc[45, "mix"]
    assert sorted(db.odor_components.loc[45, "component_id"]) == [4, 10]
    assert odors.components == []  # cleared once saved

    shown = odors.table.model().frame
    assert shown.loc[45, "components"] == "4 1-butanol + 10 (-) alpha pinene"
    assert shown.loc[1, "components"] == ""


def test_tables_pane_lists_every_method_called(window):
    tables = window.tables
    names = [tables.list.item(i).text() for i in range(tables.list.count())]
    assert "method_calls:Database.set_mouse" not in names

    window.mice.form.set_values({"mouse_id": "m3"})
    window.mice._save()
    settle(window.context)

    names = [tables.list.item(i).text() for i in range(tables.list.count())]
    assert "method_calls:Database.set_mouse" in names

    tables.list.setCurrentRow(names.index("method_calls:Database.set_mouse"))
    settle(window.context)
    frame = tables.view.model().frame
    assert list(frame["input_mouse_id"]) == ["m3"]

    # the row detail lays the JSON out and wraps
    tables._row_clicked(tables.view.model().index(0, 0))
    shown = tables.detail.row_view.model().frame
    assert "\n" in shown.loc["environment", "value"]
    assert tables.detail.row_view.wordWrap()


def test_a_boolean_field_is_a_box_that_starts_unchecked(app):
    from odyn.app.fields import Field
    from odyn.app.forms import Form

    form = Form([Field(key="flag", label="Flag", value_type="boolean", description="")])
    row = form.rows["flag"]

    assert row.value() is False and not row.editor.isChecked()
    assert form.values() == {"flag": False}

    heard = []
    form.changed.connect(lambda key, value: heard.append((key, value)))
    row.editor.setChecked(True)
    assert heard == [("flag", True)] and form.values() == {"flag": True}

    row.set_value(None)  # nothing stored reads as unchecked
    assert row.value() is False and not row.editor.isChecked()


def test_session_pane_picks_a_panel_and_its_installed_vials(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.add_panel(panel_name="rack A")
    rack = [
        {
            "vial_position": p,
            "odor_id": o,
            "vial_sccm": 200,
            "total_sccm": 1500,
            "total_volume_ml": 5,
        }
        for p, o in ((2, 2), (1, 1), (3, 0))
    ]
    old = db.add_vials(panel_name="rack A", vials=rack, made_on="2026-07-01")
    new = db.add_vials(panel_name="rack A", vials=rack[:1], made_on="2026-08-01")
    window.context.changed.emit()
    settle(window.context)

    session = window.session
    session._new()
    settle(window.context)
    picker = session.vial_list
    assert picker.table.rowCount() == 0

    picker.panel.setCurrentIndex(picker.panel.findData(1))
    # ordered by position, the remade vial 2 replacing the old one, all installed
    assert [picker.table.item(r, 1).text() for r in range(3)] == ["1", "2", "3"]
    assert picker._rows == [old[1], new[0], old[2]]
    assert picker.checked() == picker._rows
    assert sorted(session.current.vial_ids) == sorted(picker._rows)

    from PySide6.QtCore import Qt

    picker.table.item(2, 0).setCheckState(Qt.CheckState.Unchecked)  # mineral oil out
    assert sorted(session.current.vial_ids) == sorted([old[1], new[0]])

    # shown again, the draft's vials bring their panel back with the box as left
    session.refresh_current()
    assert picker.panel_id() == 1
    assert picker.checked() == [old[1], new[0]]


def test_session_and_recorder_share_one_selection(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=5)
    window.context.changed.emit()
    settle(window.context)

    session, recorder, context = window.session, window.recorder, window.context
    session._new()
    settle(window.context)
    draft = session.current
    assert context.draft is draft and context.experiment is None
    assert "no mouse yet" in recorder.session_label.text()

    # picking the mouse shows in the recorder at once
    session.mouse.setCurrentText("m5")
    assert draft.mouse_id == 5
    assert draft.folder in recorder.session_label.text()

    # nothing to record into until the session lists an experiment
    assert recorder.experiment.count() == 0
    recorder.start()
    assert recorder.recording is None
    assert "Add an experiment" in context.last_report[1]

    # experiments added in the session tab are the recorder's only choices,
    # the one in front is the one picked
    draft.set_experiment("e1")
    draft.set_experiment("e2")
    context.announce_draft()
    session.refresh_current()
    assert [
        recorder.experiment.itemText(i) for i in range(recorder.experiment.count())
    ] == ["e1", "e2"]
    assert context.experiment == "e2" and recorder.experiment.currentText() == "e2"

    # switching tabs in the session pane moves the recorder; and back
    session.tabs.setCurrentIndex(0)
    assert context.experiment == "e1" and recorder.experiment.currentText() == "e1"
    recorder.experiment.setCurrentText("e2")
    assert context.experiment == "e2" and session.tabs.currentIndex() == 1

    # removing the experiment in front drops it everywhere, no phantom name
    session._remove_experiment()
    assert [str(e["name"]) for e in draft.experiments] == ["e1"]
    assert context.experiment == "e1"
    assert [
        recorder.experiment.itemText(i) for i in range(recorder.experiment.count())
    ] == ["e1"]
    assert recorder.experiment.currentText() == "e1"


def test_quick_maths_warns_when_imaging_outlasts_the_trial(app):
    from odyn.app.forms import TimingForm

    form = TimingForm()
    assert not form.overrun.isVisibleTo(form)
    assert form.outputs["frames"].text() == "258"
    assert form.outputs["camera_rate"].text() == "12.86 Hz"

    form.inputs["block_iti_s"].setValue(4)  # 5 + 4 = 9 s after the odor, wants 11
    assert form.overrun.isVisibleTo(form)
    assert "2 s too short" in form.overrun.text()

    form.inputs["block_post_odor_acq_s"].setValue(9)
    assert form.overrun.isVisibleTo(form)  # fits, but with nothing to spare
    assert form.overrun.text() == "Only 0 s between acquisitions."

    form.inputs["block_post_odor_acq_s"].setValue(8)
    assert "Only 1 s" in form.overrun.text()

    form.inputs["block_post_odor_acq_s"].setValue(7.5)
    assert not form.overrun.isVisibleTo(form)


def test_experiment_tab_has_the_objective_in_its_form(window, tmp_path):
    session = window.session
    session._new()
    settle(window.context)
    draft = session.current
    draft.set_experiment("e1", exp_type="loop")
    window.context.announce_draft()
    session.refresh_current()

    tab = session.tabs.widget(0)
    assert list(tab.form.rows)[0] == "objective"
    assert not hasattr(tab, "exp_type")

    tab.form.rows["objective"].editor.setCurrentText("20")
    assert draft.experiment("e1")["objective"] == 20
    assert draft.experiment("e1")["exp_type"] == "loop"
