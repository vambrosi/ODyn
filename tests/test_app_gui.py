"""
The entry app's widgets, offscreen: the window opens on a fresh database,
the bars switch pages and list the day's sessions, the registries write
through the worker, the table browser reads, and the recorder writes a sync
file where `submit` will look for it.
"""

import os
import time

from pathlib import Path

import h5py
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Needs Qt and the system libraries it loads (libEGL and friends): skipped
# where either is missing, which is what leaves the rest of the suite runnable
# on a bare machine.
try:
    from PySide6.QtCore import QEventLoop, Qt
    from PySide6.QtWidgets import QApplication, QFormLayout, QPlainTextEdit, QWidget

except ImportError as error:  # a missing shared library is not a missing module
    pytest.skip(f"Qt cannot load here: {error}", allow_module_level=True)

from odyn import Database  # noqa: E402
from odyn.app.draft import Draft  # noqa: E402
from odyn.app.models import FrameModel  # noqa: E402
from odyn.app.pane_recorder import RecorderPane, recording_path  # noqa: E402
from odyn.app.pane_session import ExperimentForm, ZStacksForm  # noqa: E402
from odyn.app.pane_tables import filter_frame  # noqa: E402
from odyn.app.recorder import ReplaySource, SimulatedSource  # noqa: E402
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
            wait_until(lambda: context.pending == 0)
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


def test_window_opens_on_users_with_every_page_in_the_left_bar(window):
    tips = [a.toolTip() for a in window.left.actions() if not a.isSeparator()]

    assert tips == ["Users", "Mice", "Odors", "Panels", "Tables", "Quality Control"]
    assert window.pages.currentWidget() is window.users
    assert window.page_actions[window.users].isChecked()
    assert "tester" in window.user_label.text()

    # every page's button shows it
    window.page_actions[window.tables].trigger()
    assert window.pages.currentWidget() is window.tables
    assert window.view == "page"

    # no session yet: the right bar holds only the day's actions
    assert window.today == {}
    assert [a.toolTip() for a in window.right.actions() if a.toolTip()] == [
        "Add session",
        "Add experiment",
        "Add z-stacks",
        "Record the selected experiment",
        "Nothing to undo",
        "Nothing selected to remove",
        "Check session",
        "Submit session",
    ]
    assert not window.undo_action.isEnabled()
    assert not window.remove_action.isEnabled()
    assert not window.selection.isVisibleTo(window)

    # the recorder sits right after the Add buttons, a line between it and Undo
    actions = window.right.actions()
    at = actions.index(window.page_actions[window.recorder])
    assert actions[at - 1].toolTip() == "Add z-stacks"
    assert actions[at + 1].isSeparator() and actions[at + 2] is window.undo_action


def test_users_pane_is_a_login_screen(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.add_user(user_name="alice", display_name="Alice Smith")
    db.add_user(user_name="bob", display_name="Bob Jones")
    db.add_user(user_name="old", display_name="Old Hand")
    db.retire_user(user_name="old")
    window.context.changed.emit()
    settle(window.context)

    users = window.users
    assert [b.text() for b in users.buttons.values()] == ["Alice Smith", "Bob Jones"]
    assert "Click on your name" in users.hint.text()

    users.buttons["alice"].click()
    settle(window.context)
    assert window.context.user == "alice"
    assert users.buttons["alice"].isChecked() and not users.buttons["bob"].isChecked()
    assert "Alice Smith" in users.hint.text() and "alice" in window.user_label.text()
    # the day's work comes next: with no session yet, the page says where to add one
    assert window.pages.currentWidget() is window.sessions.no_session_page
    assert "add one" in window.sessions.no_session_page.message.text()

    # the worker's database now signs calls as alice
    db.set_mouse(mouse_id=1)  # a reader opens fine beside the worker
    window.mice.form.set_values({"mouse_id": "m2"})
    window.mice._save()
    settle(window.context)
    calls = Database(tmp_path / "data").latest_calls("set_mouse")
    assert calls.iloc[0]["user"] == "alice"


def test_mice_pane_registers_a_mouse(window, tmp_path):
    mice = window.mice
    window.show_page(mice)
    mice.form.set_values({"mouse_id": "m7", "sex": "F", "dob": "2026-01-01"})
    mice._save()
    settle(window.context)

    assert 7 in Database(tmp_path / "data").mice.index


def test_saving_shows_it_is_busy_then_says_so(window, monkeypatch):
    said = []
    window.context.status.connect(said.append)
    mice = window.mice
    mice.form.set_values({"mouse_id": "m7"})

    mice._save()
    assert QApplication.overrideCursor() is not None  # busy until it is back
    settle(window.context)

    assert said[0] == "Saving m7" and "Saved m7" in said
    assert window.context.pending == 0
    assert QApplication.overrideCursor() is None


def test_after_a_change_only_the_page_on_screen_reloads(window, monkeypatch):
    from odyn.app.pane_qc import QcPane

    reloads = []
    monkeypatch.setattr(QcPane, "reload", lambda self: reloads.append("qc"))
    # (the instance's own connection holds the method it was given; rebind)
    window.context.follow_changes(window.qc, lambda: window.qc.reload())
    window.show_page(window.mice)
    settle(window.context)
    reloads.clear()

    window.mice.form.set_values({"mouse_id": "m8"})
    window.mice._save()
    settle(window.context)
    assert reloads == []  # QC is hidden: it waits

    window.show_page(window.qc)
    settle(window.context)
    assert reloads == ["qc"]  # and catches up once, when shown


def test_a_short_job_takes_the_lock_once(window, monkeypatch):
    from odyn.locking import DatabaseLock

    taken = []
    original = DatabaseLock._acquire_file
    monkeypatch.setattr(
        DatabaseLock,
        "_acquire_file",
        lambda self: taken.append(1) or original(self),
    )

    def reads(db):
        return db.mice, db.mouse_lines, db.mouse_viruses, db.lines, db.viruses

    window.context.run(reads, None, "reads", one_lock=True)
    settle(window.context)
    assert len(taken) == 1

    taken.clear()
    window.context.run(reads, None, "reads")
    settle(window.context)
    assert len(taken) > 1


def test_mice_pane_loads_a_stored_mouse_and_asks_before_overwriting(
    window, tmp_path, monkeypatch
):
    mice = window.mice
    window.show_page(mice)
    mice.form.set_values({"mouse_id": "m7", "sex": "F"})
    mice._save()
    settle(window.context)

    asked = []
    monkeypatch.setattr(
        window.context, "confirm", lambda *args: asked.append(args) or False
    )

    # typing a stored number loads the mouse, with a note that it is stored
    mice.form.clear()
    number = mice.form.rows["mouse_id"]
    number.editor.setText("m7")
    number.editor.editingFinished.emit()
    assert mice.form.values() == {"mouse_id": 7, "sex": "F"}
    assert number.note.text() == "Mouse ID already exists (saving will overwrite data)"

    # saving over it asks, and Cancel writes nothing
    mice.form.rows["sex"].set_value("M")
    mice._save()
    settle(window.context)
    assert asked[-1][1:] == (number.note.text(), "Overwrite")
    assert Database(tmp_path / "data").mice.loc[7, "mouse_sex"] == "F"

    monkeypatch.setattr(window.context, "confirm", lambda *args: True)
    mice._save()
    settle(window.context)
    assert Database(tmp_path / "data").mice.loc[7, "mouse_sex"] == "M"

    # a number typed and saved at once (no focus change) is read, and asks too
    monkeypatch.setattr(
        window.context, "confirm", lambda *args: asked.append(args) or False
    )
    asked.clear()
    mice.form.clear()
    number.editor.setText("7")
    mice._save()
    assert len(asked) == 1

    # clicking a mouse in the table loads it
    mice.form.clear()
    row = list(mice.table.model().frame.index).index(7)
    mice.table.clicked.emit(mice.table.model().index(row, 0))
    assert mice.form.values() == {"mouse_id": 7, "sex": "M"}


def test_registry_forms_hide_and_come_back(window):
    from odyn.app.pane_registry import TableAndForm
    from odyn.app.style import FORM_PX

    window.show_page(window.mice)
    split = window.mice.split
    QApplication.processEvents()
    assert split.is_open() and split.form.width() >= FORM_PX

    split.toggle.click()  # the table takes the whole width
    QApplication.processEvents()
    assert not split.is_open()
    assert split.table.width() > split.width() - 3 * split.toggle.width()

    # dragged wider, the width is kept for when it opens again
    split.toggle.click()
    total = sum(split.splitter.sizes())
    split.splitter.setSizes([total - FORM_PX - 80, FORM_PX + 80])
    split._moved()
    assert TableAndForm.form_px == FORM_PX + 80

    # and showing the page opens it, whatever it was left as
    split.set_open(False)
    window.show_page(window.odors)
    window.show_page(window.mice)
    QApplication.processEvents()
    assert split.is_open() and split.splitter.sizes()[1] == FORM_PX + 80

    # the other registry pages are built the same way, and share the width
    for page in (window.odors, window.panels):
        window.show_page(page)
        QApplication.processEvents()
        assert page.split.is_open()
        assert page.split.splitter.sizes()[1] == FORM_PX + 80

    TableAndForm.form_px = FORM_PX


def test_clear_starts_a_new_entry_and_the_id_is_checked_as_typed(window):
    mice = window.mice
    window.show_page(mice)
    mice.form.set_values({"mouse_id": "m7", "sex": "F"})
    mice._save()
    settle(window.context)

    # a stored number says so while it is typed, before the field is left
    number = mice.form.rows["mouse_id"]
    number.editor.setText("m")
    number.editor.setText("m7")
    assert not number.note.isHidden() and "already exists" in number.note.text()
    number.editor.setText("m71")
    assert number.note.isHidden()

    # Clear empties everything, lists included
    number.editor.setText("m7")
    number.editor.editingFinished.emit()
    mice.line_list.add()
    assert mice.form.values() == {"mouse_id": 7, "sex": "F"}
    mice.clear()
    assert mice.form.values() == {} and not mice.line_list.rows
    assert number.note.isHidden()

    # on the odors page it also unlocks what a stored odor fixes
    odors = window.odors
    odor_number = odors.form.rows["odor_id"]
    odor_number.editor.setText("1")
    assert "already exists" in odor_number.note.text()
    odor_number.editor.editingFinished.emit()
    assert not odors.form.rows["odor_name"].editor.isEnabled()
    odors.clear()
    assert odors.form.values() == {}
    assert odors.form.rows["odor_name"].editor.isEnabled()
    assert odors.add_component.isEnabled()

    # and on the panels page it empties the list of vials
    panels = window.panels
    panels.add_card()
    panels.form.set_values({"made_on": "2026-10-06"})
    panels.clear()
    assert panels.cards == [] and panels.form.values() == {}


def test_mice_pane_lists_lines_and_viruses_as_rows(window, tmp_path, monkeypatch):
    # (the lines and viruses are seeded when the database is made)
    mice = window.mice
    window.show_page(mice)
    lines, viruses = mice.line_list, mice.virus_list
    mice.form.set_values({"mouse_id": "m7"})
    lines.add()
    lines.add()
    lines.rows[0].set_value("TH-Cre", "het")
    assert lines.rows[1].value()[1] == "na"  # a new row starts not genotyped
    lines.rows[1].set_value("Thy1-GCaMP", "na")
    viruses.add()
    viruses.rows[0].set_value("pAAVrg-EF1a-Flpo")
    assert lines.rows[0].genotype is not None and viruses.rows[0].genotype is None
    assert dict(lines.values()) == {"TH-Cre": "het", "Thy1-GCaMP": "na"}
    assert viruses.values() == ["pAAVrg-EF1a-Flpo"]

    mice._save()
    settle(window.context)

    db = Database(tmp_path / "data")
    stored = db.mouse_lines.loc[[7]]
    assert dict(zip(stored["line_name"], stored["genotype"])) == {
        "TH-Cre": "het",
        "Thy1-GCaMP": "na",
    }
    assert list(db.mouse_viruses.loc[[7], "virus_name"]) == ["pAAVrg-EF1a-Flpo"]

    # picking the mouse brings its rows back; the table shows them as text
    frame = mice.table.model().frame
    assert frame.loc[7, "lines"] == "TH-Cre (het), Thy1-GCaMP (na)"
    assert frame.loc[7, "viruses"] == "pAAVrg-EF1a-Flpo"

    row = list(frame.index).index(7)
    mice._pick(mice.table.model().index(row, 0))
    assert dict(lines.values()) == {"TH-Cre": "het", "Thy1-GCaMP": "na"}
    assert viruses.values() == ["pAAVrg-EF1a-Flpo"]

    # wild type has nothing to genotype: its choice is fixed to na
    row = lines.rows[0]
    row.name.setCurrentText("Wild Type")
    assert row.value() == ("Wild Type", "na") and not row.genotype.isEnabled()
    row.name.setCurrentText("TH-Cre")
    assert row.genotype.isEnabled()

    # the genotype box is as wide on every row, and shows its choices whole
    widths = {r.genotype.width() for r in lines.rows}
    assert len(widths) == 1
    advance = row.genotype.fontMetrics().horizontalAdvance
    assert widths.pop() > max(advance(c) for c in ("na", "het", "hom", "wt")) + 40

    # each entry is a row of the form: the list's title labels its first
    layout = mice.form.layout()
    assert [layout.labelForField(r).text() for r in lines.rows] == ["Lines", ""]
    assert layout.labelForField(viruses.rows[0]).text() == "Viruses"

    # a number typed without picking leaves its stored lines and viruses alone
    for listed in (lines, viruses):
        listed.load(None, [])

    monkeypatch.setattr(window.context, "confirm", lambda *args: True)

    mice.form.set_values({"mouse_id": "m7", "sex": "M"})
    mice._save()
    settle(window.context)
    db = Database(tmp_path / "data")
    assert len(db.mouse_lines.loc[[7]]) == 2 and len(db.mouse_viruses.loc[[7]]) == 1

    # a column is never narrower than its name
    header = mice.table.horizontalHeader()
    for column in range(header.count()):
        name = str(mice.table.model().headerData(column, 1))
        assert header.sectionSize(column) >= header.fontMetrics().horizontalAdvance(
            name
        )


def test_mice_form_rows_keep_one_spacing(window, app):
    # empty or filled, the lists' rows are spaced like the fields above them
    window.show_page(window.mice)
    mice = window.mice
    form = mice.form.layout()

    def shown() -> list[tuple[QWidget, QWidget]]:
        """(label, field) of each row on screen, top to bottom."""
        app.processEvents()
        pairs = [
            (
                form.itemAt(r, QFormLayout.ItemRole.LabelRole).widget(),
                form.itemAt(r, QFormLayout.ItemRole.FieldRole).widget(),
            )
            for r in range(form.rowCount())
        ]
        return [(label, field) for label, field in pairs if field.isVisible()]

    def check(count: int) -> None:
        pairs = shown()
        tops = [field.geometry().top() for _, field in pairs]
        assert len(pairs) == count
        assert len({b - a for a, b in zip(tops, tops[1:])}) == 1  # one step
        # every label sits the same way against its field
        assert len({f.geometry().top() - l.geometry().top() for l, f in pairs}) == 1

    check(5)
    mice.line_list.add()
    mice.line_list.add()
    mice.virus_list.add()
    check(6)
    mice.line_list.remove(mice.line_list.rows[0])
    mice.virus_list.remove(mice.virus_list.rows[0])
    check(5)
    assert mice.virus_list.none.isVisible()
    assert form.labelForField(mice.line_list.rows[0]).text() == "Lines"


def test_tables_pane_reads_filters_and_shows_a_row(window):
    tables = window.tables
    tables.picker.setCurrentText("odors")
    settle(window.context)

    model = tables.view.model()
    assert isinstance(model, FrameModel)
    assert model.rowCount() > 20  # the seeded odors

    tables.filter.setText("eugenol")
    assert 0 < tables.view.model().rowCount() < 5

    tables._row_clicked(tables.view.model().index(0, 0))
    assert tables.detail.title.text().startswith("odors: ")
    assert not tables.detail.annotate_box.isVisible()  # odors are not annotated
    assert not tables.detail.save.isVisible()

    # Every column is shown locked, as the registry forms show what is fixed
    name = tables.detail.values["odor_name"]
    assert name.isReadOnly() and "eugenol" in name.text().lower()


def test_detail_pane_annotates_a_mouse(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=9, sex="M")
    window.show_page(window.tables)
    window.context.changed.emit()
    settle(window.context)

    tables = window.tables
    tables.picker.setCurrentText("mice")
    settle(window.context)

    tables._row_clicked(tables.view.model().index(0, 0))
    settle(window.context)
    detail = tables.detail
    assert detail.target_type == "mouse" and detail.target_id == 9
    assert detail.form is not None and detail.save.isVisible()

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
    assert "stopped after" in recorder.status.text()
    assert list(tmp_path.rglob("*.h5")) == []


def test_recorder_records_into_the_session_folder(window, tmp_path):
    draft = Draft.start(tmp_path / "drafts", user="tester").identify(
        mouse_id=7, date="2026-09-22"
    )
    window.sessions.reload_drafts()
    assert window.recorder.draft is draft or window.recorder.draft.path == draft.path

    draft = window.sessions.current  # the window's own copy, loaded from disk
    draft.set_part("e1").set_part("e2")
    window.context.announce_draft()
    recorder: RecorderPane = window.recorder

    # only a part selected in the right bar is recorded
    window.open_session(draft)
    assert recorder.picked is None
    recorder.start()
    assert recorder.recording is None
    assert "Select an experiment" in window.context.last_report[1]

    window.open_part(draft, "e2")
    assert recorder.picked == "e2"
    window.open_part(draft, "e1")
    window.page_actions[window.recorder].trigger()
    assert recorder.part.text() == "e1"
    recorder.start()
    wait_until(
        lambda: recorder.recording is not None and recorder.recording.total > 100
    )

    # written on this computer, beside the drafts, in the share's layout
    record = draft.recordings("e1")[0]
    local = Path(record["local"])
    assert local == tmp_path / "sync" / "20260922" / "m7" / "e1" / record["file"]
    assert local.is_file() and window.context.recording_now == str(local)
    assert record["share"] == f"20260922/m7/e1/sync/{record['file']}"

    # the file is named for the mouse and day, so those are fixed now
    assert not window.sessions.session_page.mouse.isEnabled()
    with pytest.raises(ValueError, match="fixed"):
        draft.identify(mouse_id=8)

    # selecting another part meanwhile does not move the recording
    window.open_part(draft, "e2")
    assert recorder.part.text() == "e1"

    recorder.stop()
    wait_until(lambda: not recorder.copier.running)  # copied in the background
    assert recorder.part.text() == "e2"  # and then it follows again

    part = tmp_path / "data" / "20260922" / "m7" / "e1"
    assert (part / "raw").is_dir() and (part / "processed").is_dir()
    files = list((part / "sync").glob("*"))
    assert [f.name for f in files] == [record["file"]]  # no partial left
    assert local.is_file()  # and the local copy is kept
    assert record["stopped"] and record["copied"]

    with h5py.File(files[0]) as h5:
        assert h5.attrs["mouse"] == "m7" and h5.attrs["experiment_folder"] == "e1"
        assert h5["2pFrameSync"].shape[0] > 100
        assert h5["filter_changes"].shape == (1, 3)

    # the draft now lists the part the recorder made
    shown = window.sessions.current
    assert shown is not None and shown.part("e1") is not None
    assert list(window.today) == [
        (id(shown), None),
        (id(shown), "e1"),
        (id(shown), "e2"),
    ]


def test_a_replay_records_its_file_and_stops_at_its_end(window, tmp_path, monkeypatch):
    import generate_sync

    from odyn.app.recorder import RATE

    monkeypatch.setattr(generate_sync, "RATE_HZ", RATE)
    original = tmp_path / "m7_e1_rig.h5"
    generate_sync.generate_sync(original, acquisitions=2, frames=20, mouse="m7")

    draft = Draft.start(tmp_path / "drafts", user="tester").identify(
        mouse_id=7, date="2026-01-01"
    )
    window.sessions.reload_drafts()
    draft = window.sessions.current
    draft.set_part("e1")
    window.context.announce_draft()
    window.open_part(draft, "e1")

    recorder: RecorderPane = window.recorder
    recorder.source = ReplaySource(original, speed=50)
    said = []
    window.context.status.connect(said.append)
    recorder.start()
    assert "(replay of m7_e1_rig.h5, x50)" in said[0]

    # nobody presses Stop: the end of the file does, and the copy follows
    wait_until(lambda: recorder.recording is None)
    wait_until(lambda: not recorder.copier.running)
    record = draft.recordings("e1")[0]
    assert record["stopped"] and record["copied"]

    with h5py.File(original) as rig, h5py.File(Path(record["local"])) as replayed:
        assert replayed.attrs["n_samples"] == rig["2pFrameSync"].shape[0]
        assert replayed.attrs["start_time"] == rig.attrs["start_time"]

    # played again from the start for the next recording
    recorder.test()
    wait_until(lambda: recorder.recording is None)
    assert "stopped after" in recorder.status.text()


def test_a_failed_read_stops_the_recording_once(window):
    class Broken(SimulatedSource):
        def read(self):
            raise RuntimeError("buffer overflow")

    recorder: RecorderPane = window.recorder
    recorder.source = Broken()
    recorder.test()
    recorder._tick()  # used to stop, read again, fail again... to RecursionError

    assert recorder.recording is None and recorder.start_button.isEnabled()
    assert window.context.last_report == (
        "Recorder",
        "Recording stopped:\n\nbuffer overflow",
    )


def test_without_a_board_nothing_records(window, monkeypatch):
    def no_board():
        raise RuntimeError("no device Dev1")

    recorder: RecorderPane = window.recorder
    monkeypatch.setattr(recorder, "source", None)
    monkeypatch.setattr("odyn.app.pane_recorder.DaqSource", no_board)
    recorder.test()

    assert recorder.recording is None  # no made-up traces
    assert "no device Dev1" in window.context.last_report[1]


def test_the_daq_gets_a_new_task_for_every_start(monkeypatch):
    import sys
    import types

    from odyn.app.recorder import DaqSource

    made = []

    class Task:
        def __init__(self):
            self.state = "made"
            self.ai_channels = types.SimpleNamespace(
                add_ai_voltage_chan=lambda *a, **k: None
            )
            self.timing = types.SimpleNamespace(
                cfg_samp_clk_timing=lambda *a, **k: None
            )
            made.append(self)

        def start(self):
            assert self.state == "made", "a closed task cannot start again"
            self.state = "running"

        def stop(self):
            pass

        def close(self):
            self.state = "closed"

    nidaqmx = types.ModuleType("nidaqmx")
    nidaqmx.Task = Task
    constants = types.ModuleType("nidaqmx.constants")
    constants.AcquisitionType = types.SimpleNamespace(CONTINUOUS=1)
    constants.TerminalConfiguration = types.SimpleNamespace(RSE=1)
    monkeypatch.setitem(sys.modules, "nidaqmx", nidaqmx)
    monkeypatch.setitem(sys.modules, "nidaqmx.constants", constants)

    source = DaqSource()  # a Test, then a recording
    source.start()
    source.stop()
    source.start()

    assert [task.state for task in made] == ["closed", "running"]


def test_recording_again_asks_and_hides_the_earlier_copy(window, tmp_path, monkeypatch):
    Draft.start(tmp_path / "drafts", user="tester").identify(
        mouse_id=7, date="2026-09-22"
    )
    window.sessions.reload_drafts()
    draft = window.sessions.current
    draft.set_part("e1")
    window.open_part(draft, "e1")
    recorder: RecorderPane = window.recorder

    def record():
        recorder.start()
        wait_until(lambda: recorder.recording is not None)
        time.sleep(1.1)  # file names are by the second
        recorder.stop()
        wait_until(lambda: not recorder.copier.running)

    record()
    asked = []
    monkeypatch.setattr(
        window.context, "confirm", lambda *args: asked.append(args) or False
    )
    recorder.start()  # declined: nothing starts
    assert asked and recorder.recording is None

    monkeypatch.setattr(window.context, "confirm", lambda *args: True)
    record()
    first, second = draft.recordings("e1")
    assert first["replaced"] and first["hidden"] and not second.get("replaced")

    # only the new one is in `process_sync`'s sight; the old one is kept, hidden
    sync = tmp_path / "data" / "20260922" / "m7" / "e1" / "sync"
    assert [p.name for p in sync.glob("[!.]*.h5")] == [second["file"]]
    assert (sync / f".replaced-{first['file']}").is_file()


def test_a_recording_stays_local_when_the_share_is_away(window, tmp_path):
    Draft.start(tmp_path / "drafts", user="tester").identify(
        mouse_id=7, date="2026-09-22"
    )
    window.sessions.reload_drafts()
    draft = window.sessions.current
    draft.set_part("e1")
    window.open_part(draft, "e1")

    # the session folder is a file: nothing can be made or copied under it
    (tmp_path / "data" / "20260922").mkdir(parents=True)
    (tmp_path / "data" / "20260922" / "m7").write_text("in the way")

    recorder: RecorderPane = window.recorder
    recorder.start()
    wait_until(lambda: recorder.recording is not None)
    recorder.stop()
    wait_until(lambda: not recorder.copier.running)

    record = draft.recordings("e1")[0]
    assert Path(record["local"]).is_file() and not record.get("copied")
    assert "will try to copy it again" in recorder.status.text()


def test_recording_path_layout(tmp_path):
    draft = Draft.start(tmp_path, user="x").identify(mouse_id=12, date="2026-03-04")
    local, share, mouse = recording_path(tmp_path / "sync", draft, "e2")

    assert mouse == "m12"
    assert local.parent == tmp_path / "sync" / "20260304" / "m12" / "e2"
    assert share == f"20260304/m12/e2/sync/{local.name}"
    assert local.name.startswith("m12_e2_")


def test_qc_pane_lists_nothing_for_a_new_user(window):
    qc = window.qc
    settle(window.context)

    assert qc.recordings_view.model().rowCount() == 0
    assert "tester" in qc.who.text() or "alice" in qc.who.text()

    # One tab per list, no counts when empty, the first one shown
    names = [qc.tabs.tabText(i) for i in range(qc.tabs.count())]
    assert names == ["Waiting", "Added", "Annotations", "Motion correction"]
    assert qc.tabs.currentIndex() == 0


def test_qc_lists_the_drafts_waiting_for_the_cluster(window, tmp_path, monkeypatch):
    from odyn.app.submissions import enqueue

    main = tmp_path / "data"
    draft = Draft.start(tmp_path / "elsewhere", mouse_id=3, date="2026-10-07")
    draft.set_part("e1", objective=20)
    path = enqueue(draft, main, user="tester")
    enqueue(Draft.start(tmp_path / "x", mouse_id=4), main, user="someone")

    window.context.set_user("tester")
    qc = window.qc
    qc.reload()
    settle(window.context)

    # Only this user's, and nothing to do before one is picked
    assert qc.queue[["session", "state", "progress"]].values.tolist() == [
        ["20261007/m3", "waiting", "not looked at yet"]
    ]
    assert not qc.add_now.isEnabled() and not qc.edit_draft.isEnabled()
    assert qc.tabs.tabText(0) == "Waiting (1)" and qc.tabs.currentIndex() == 0

    # Every user's with Show all users, each with whose it is
    qc.everyone.setChecked(True)
    settle(window.context)
    assert sorted(qc.queue["user"]) == ["someone", "tester"]
    assert list(qc._model(qc.queue_view)._frame.columns)[:2] == ["submitted_at", "user"]
    assert qc.who.text() == "To-do list of all users"
    qc.everyone.setChecked(False)
    settle(window.context)
    assert len(qc.queue) == 1

    # A tab the user picked stays picked through reloads
    qc.tabs.tabBarClicked.emit(3)
    qc.tabs.setCurrentIndex(3)
    qc.reload()
    settle(window.context)
    assert qc.tabs.currentIndex() == 3
    qc.tabs.setCurrentIndex(0)

    qc.queue_view.selectRow(0)
    assert qc.add_now.isEnabled() and qc.edit_draft.isEnabled()
    assert not qc.submit_again.isEnabled()  # only a failed one

    qc.add_now.click()
    settle(window.context)
    assert path.with_name(f"{path.stem}.now").exists()

    # Edit draft takes it out of the queue and opens it among the drafts
    monkeypatch.setattr(window.context, "confirm", lambda *_: True)
    qc.queue_view.selectRow(0)
    qc.edit_draft.click()
    settle(window.context)

    assert qc.queue.empty and not path.exists()
    assert window.context.draft.where == "20261007/m3"
    assert window.pages.currentWidget() is window.sessions.session_page


def test_qc_asks_the_cluster_to_process_a_sync_file_again(window, tmp_path):
    import json
    from datetime import datetime

    from generate_data import generate

    from odyn.cluster import pending_requests

    main = tmp_path / "data"
    generate(
        main,
        acquisitions=2,
        frames=56,
        height=32,
        width=40,
        start=datetime(2026, 3, 4, 9),
    )
    db = Database(main, user="tester")
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )

    window.context.set_user("tester")
    qc = window.qc
    qc.reload()
    settle(window.context)
    assert not qc.rerun_sync.isEnabled()

    qc.recordings_view.selectRow(0)
    qc.rerun_sync.click()
    settle(window.context)

    [request] = pending_requests(main)
    sent = json.loads(request.read_text())
    assert sent["method"] == "process_sync" and sent["kwargs"] == {"exp_id": exp_id}
    assert sent["user"] == "tester"


def test_qc_lists_every_kind_of_recording_this_user_submitted(tmp_path):
    from datetime import datetime

    from generate_data import generate

    from odyn.app.pane_qc import my_missing, my_recordings

    size = dict(frames=56, height=32, width=40)
    generate(tmp_path, acquisitions=2, start=datetime(2026, 3, 4, 9), **size)
    generate(
        tmp_path,
        acquisitions=1,
        exp="e2",
        acq_state="grab",
        start=datetime(2026, 3, 4, 10),
        **size,
    )
    generate(
        tmp_path,
        acquisitions=1,
        exp="zstacks",
        acq_state="grab",
        start=datetime(2026, 3, 4, 11),
        **size,
        zstack=dict(
            slices=3, step_um=5, top_um=-10, power=30, length_um=-300, reference_um=-15
        ),
    )
    db = Database(tmp_path, can_create=True, user="alice")
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )
    grab_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e2", objective=10, exp_type="grab"
    )
    (stack_id,) = db.add_zstacks(
        session_id=session_id, folder="20260304/m1/zstacks", objective=10
    )

    mine = my_recordings(db, "alice")
    assert list(mine.index) == [
        ("zstack", stack_id),
        ("experiment", grab_id),
        ("experiment", exp_id),
    ]  # newest first
    assert mine.loc[("zstack", stack_id), "path"].endswith("_zstacks_00001.tif")
    assert mine.loc[("experiment", grab_id), "path"] == "20260304/m1/e2"
    assert mine.loc[("experiment", exp_id), "path"] == "20260304/m1/e1"
    assert mine.loc[("zstack", stack_id), "sync"] == ""
    assert my_recordings(db, "someone else").empty

    # A key made required afterwards asks for it on what is already in, in
    # the order of the submissions
    db.update_annotation_key(applies_to="experiment", key="goal", required=True)
    missing = my_missing(db, mine)
    assert list(zip(missing["target_type"], missing["target_id"])) == [
        ("experiment", grab_id),
        ("experiment", exp_id),
    ]
    assert missing.columns[0] == "waiting_since"
    assert missing["waiting_since"].tolist() == [
        mine.loc[("experiment", grab_id), "submitted_at"],
        mine.loc[("experiment", exp_id), "submitted_at"],
    ]
    assert my_missing(db, mine.iloc[0:0]).columns[0] == "waiting_since"


def test_qc_says_when_the_cluster_last_ran(tmp_path):
    from datetime import datetime, timedelta

    from odyn.app.pane_qc import cluster_line
    from odyn.cluster import send_request, tick

    db = Database(tmp_path, can_create=True, user="alice")
    assert cluster_line(db) == ("The cluster has not run yet", "computed")

    tick(tmp_path)
    send_request(tmp_path, "process_sync", exp_id=1)
    text, kind = cluster_line(db)
    assert text.startswith("The cluster last ran at") and kind == "computed"
    assert text.endswith("(0 min ago), 1 request waiting")

    text, kind = cluster_line(db, now=datetime.now() + timedelta(hours=3))
    assert "(3 h ago)" in text and "may have stopped" in text and kind == "warning"


def a_tested_experiment(main, user="tester") -> tuple[Database, int]:
    """
    An experiment of `user`'s with the cluster's test finished, and a sweep
    whose baseline finished and whose one piecewise run did not.
    """
    import json
    from datetime import datetime

    import pandas as pd
    from generate_data import generate

    from odyn.cluster import cluster_folder

    generate(
        main,
        acquisitions=2,
        frames=56,
        height=32,
        width=40,
        start=datetime(2026, 3, 4, 9),
    )
    db = Database(main, can_create=True, user=user)
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )
    links = db.group_experiments
    group_id = int(links.index[links["exp_id"] == exp_id][0])
    group = db.groups[group_id]
    cluster = cluster_folder(main)
    jobs = cluster / "jobs"
    jobs.mkdir(parents=True)

    def report(folder, corr, **metrics):
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(
            json.dumps({"frames_at_limit": 0.0, **metrics})
        )
        pd.DataFrame(
            {
                "acq_id": [1, 2],
                "test_corr_with_ref": [corr, corr],
                "test_flow_median_px": [0.5, 0.7],
                "test_flow_p90_px": [1.5, 2.5],
            }
        ).to_csv(folder / "summary.csv", index=False)
        (folder / "shifts.png").write_bytes(b"png")
        (folder / "Group_1_e1_raw_test.avi").write_bytes(b"avi")

    # The cluster's test, and someone else's (not listed)
    (cluster / "mcor").mkdir()
    for group, job_id, who in [(group_id, "11", user), (999, "12", "someone")]:
        (cluster / "mcor" / f"{group}.json").write_text(
            json.dumps(
                {
                    "group_id": group,
                    "job_id": job_id,
                    "user": who,
                    "submitted_at": "2026-10-13 09:00:00",
                }
            )
        )
    (jobs / f"odyn-mcor-main-g{group_id}-11.out").write_text("...\nfinished now\n")
    report(db.groups[group_id].mcor_test_folder(), 0.9)

    # A sweep, submitted later
    (cluster / "sweeps").mkdir()
    (cluster / "sweeps" / f"g{group_id}.json").write_text(
        json.dumps(
            {
                "group_id": group_id,
                "user": user,
                "baseline_job": "20",
                "array_job": "21",
                "submitted_at": "2026-10-13 10:00:00",
                "runs": [{"run": "s128_o64_d6"}],
            }
        )
    )
    name = f"odyn-sweep-main-g{group_id}"
    (jobs / f"{name}-20.out").write_text("finished now\n")
    (jobs / f"{name}-21_1.out").write_text("started  ...\n")
    report(db.groups[group_id].mcor_test_folder("rigid"), 0.8)

    return db, group_id


def test_qc_lists_the_tests_of_this_users_experiments(tmp_path, monkeypatch):
    from odyn.app.pane_qc import my_tests

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path)

    from odyn.app.pane_qc import my_recordings

    tests = my_tests(db, "tester", my_recordings(db, "tester"))

    # Newest first, the sweep's baseline before its runs
    assert tests[["path", "run", "state"]].values.tolist() == [
        ["20260304/m1/e1", "rigid", "finished"],
        ["20260304/m1/e1", "s128_o64_d6", "not finished"],
        ["20260304/m1/e1", "test", "finished"],
    ]
    assert tests["corr_with_ref"].tolist()[0] == 0.8
    assert tests["flow_px"].tolist()[2] == 0.6
    assert pd_isna(tests["corr_with_ref"].tolist()[1])
    assert tests["finished_at"].tolist()[1] == ""
    assert my_tests(db, "someone else", my_recordings(db, "someone else")).empty

    # Each test shows when its experiment was submitted, as the list above it
    submitted = my_recordings(db, "tester")["submitted_at"].iloc[0]
    assert (tests["submitted_at"] == submitted).all()

    # The cluster's whole correction (every acquisition saved) says so
    import json

    from odyn.cluster import cluster_folder

    sent = cluster_folder(tmp_path) / "mcor" / f"{group_id}.json"
    sent.write_text(json.dumps({**json.loads(sent.read_text()), "whole": True}))
    runs = my_tests(db, "tester", my_recordings(db, "tester"))["run"].tolist()
    assert runs == ["rigid", "s128_o64_d6", "rigid (all)"]

    # One row per experiment, its files counted once, the cluster's test
    # first among the runs to compare
    from odyn.app.pane_qc import my_experiments

    experiments = my_experiments(tests, db.main_folder, tmp_path / "base")
    assert len(experiments) == 1

    # Approved after the latest pick: approved; a pick after it: not
    pick = {"call_id": 5, "run": "s128", "correct": False}
    status = [
        my_experiments(
            tests, db.main_folder, tmp_path / "base", {group_id: pick}, None, approved
        )["status"].iloc[0]
        for approved in ({group_id: {"call_id": 7}}, {group_id: {"call_id": 3}}, None)
    ]
    assert status == ["approved", "ready for final check", "ready for final check"]
    row = experiments.iloc[0]
    top = db.groups[group_id].mcor_test_folder()
    assert row["folder"] == str(top)
    assert row["folders"] == [str(top), str(top / "rigid")]
    assert [run.name for run in row["finished"]] == ["test", "rigid"]
    assert row["finished"][0].corr_with_ref == 0.9
    assert row["finished"][0].flow_worst_px == 2.5
    assert row["finished"][0].flow_worst_acq == 2
    assert row["bytes"] == sum(
        path.stat().st_size
        for f in (top, top / "rigid")
        for path in f.iterdir()
        if path.is_file()
    )

    # A whole correction is no run to compare: rigid (all) is rigid
    whole = my_tests(db, "tester", my_recordings(db, "tester"))
    row = my_experiments(whole, db.main_folder, tmp_path / "base").iloc[0]
    assert whole["whole"].tolist() == [False, False, True]
    assert "rigid (all)" not in [run.name for run in row["finished"]]
    assert str(top) in row["folders"]  # its files: the full run's


def pd_isna(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def a_video(path, frames=4, value=100) -> None:
    """A small grayscale video at `path`."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 32), isColor=False
    )

    for index in range(frames):
        writer.write(np.full((32, 40), value + index, np.uint8))

    writer.release()


def test_qc_opens_an_experiments_files(window, tmp_path, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    import gc

    db, group_id = a_tested_experiment(tmp_path / "data")
    top = db.groups[group_id].mcor_test_folder()
    del db
    gc.collect()  # its database (a cycle with its groups) closes in this thread

    # Real videos: the raw one apart, and each finished run's test alone
    for folder in (top, top / "rigid"):
        (folder / "Group_1_e1_raw_test.avi").unlink()
        a_video(folder / "Group_1_e1_test.avi")

    a_video(top / "raw" / "Group_1_e1_raw.avi")
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )

    window.context.set_user("tester")
    qc = window.qc
    qc.reload()
    settle(window.context)

    # One row for the experiment, whatever its tests
    rows = qc.experiments[["path", "status", "runs", "local copy"]].values.tolist()
    assert rows == [["20260304/m1/e1", "ready to compare", "2 of 3 finished", "no"]]
    assert qc.experiments["size"].iloc[0].endswith("B")
    assert qc.download_all.text().startswith("Download all (1 experiment, ")

    # Nothing opens from the share: an experiment is downloaded first
    qc.tests_view.selectRow(0)
    assert not qc.compare.isEnabled() and not qc.show_folder.isEnabled()
    assert qc.download.isEnabled() and not qc.delete_local.isEnabled()
    qc.download.click()
    wait_until(lambda: qc.downloads is None)
    settle(window.context)
    assert qc.experiments["local copy"].tolist() == ["yes"]
    assert not qc.download_bar.isVisibleTo(qc)  # only while downloading
    assert window.cache_label.text() != "Local cache: 0 B"
    local = window.context.cache_base / "cache"
    assert len(list(local.rglob("*.avi"))) == 3

    qc.tests_view.selectRow(0)
    qc.show_folder.click()
    assert Path(opened[-1].toLocalFile()).name == top.name

    # The window: raw first, then the cluster's test, then the sweep's runs
    qc.compare.click()
    assert len(qc.windows) == 1
    mcor = next(iter(qc.windows.values()))
    grid = mcor.grid
    assert [tile.video.name for tile in grid.tiles] == ["raw", "test", "rigid"]
    assert grid.tiles[1].video.numbers.startswith("corr 0.900  flow 0.60 / 2.50 px")

    # A focused run's numbers in words, under the grid
    grid.tiles[1].clicked.emit(False)
    assert mcor.details.text() == (
        "test: correlation with the reference 0.900, residual flow 0.60 px "
        "typical, 2.50 px worst (acquisition 2)"
    )
    grid.clear_focus()
    assert all(tile.video.path.is_relative_to(local) for tile in grid.tiles)
    wait_until(lambda: grid.frame_count == 4)
    assert mcor.missing.text() == (  # these reports are from before
        "No per-frame numbers to plot (older tests): test, rigid"
    )

    # A run's plot, from its tile
    assert not mcor.report_action.isEnabled()
    grid.tiles[2].clicked.emit(False)
    assert mcor.report_action.isEnabled()
    mcor.report_action.trigger()
    report = Path(opened[-1].toLocalFile())
    assert report.name == "shifts.png" and report.parent.name == "rigid"
    assert report.is_relative_to(local)
    grid.tiles[0].clicked.emit(False)  # the raw video has no plot
    assert not mcor.report_action.isEnabled()

    # Opened again, the same window comes up
    qc.compare.click()
    assert list(qc.windows.values()) == [mcor]

    # The local copy goes (its window with it), the share's stays
    qc.delete_local.click()
    settle(window.context)
    assert qc.windows == {}
    assert qc.experiments["local copy"].tolist() == ["no"]
    assert window.cache_label.text() == "Local cache: 0 B"
    assert (top / "metrics.json").is_file()


def test_a_test_is_picked_from_the_window(window, tmp_path, monkeypatch):
    import gc
    import json

    from PySide6.QtWidgets import QMessageBox

    from odyn.cluster import cluster_folder

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path / "data")
    del db
    gc.collect()
    asked = []
    monkeypatch.setattr(
        window.context,
        "confirm",
        lambda title, message, action, parent=None: asked.append(
            (message, action, parent)
        )
        or True,
    )

    window.context.set_user("tester")
    qc = window.qc
    window.show_page(qc)  # a change reloads the page on screen
    qc.reload()
    settle(window.context)
    qc.tests_view.selectRow(0)
    qc.download.click()
    wait_until(lambda: qc.downloads is None)
    settle(window.context)
    qc.tests_view.selectRow(0)
    qc.compare.click()
    mcor = next(iter(qc.windows.values()))
    assert mcor.state.text() == "Status: ready to compare; raw check: not yet"

    # One run picked, never raw; the cluster test set aside first
    assert not mcor.pick_action.isEnabled()
    names = [tile.video.name for tile in mcor.grid.tiles]
    mcor.grid.tiles[names.index("test")].clicked.emit(False)
    mcor.aside_action.trigger()
    mcor.grid.tiles[names.index("rigid")].clicked.emit(False)
    assert mcor.pick_action.isEnabled()
    mcor.pick_action.trigger()
    settle(window.context)

    message, action, parent = asked[-1]
    assert message == (
        "Are you sure you want to pick rigid?\n\n"
        "This will overwrite the rigid run files."
    )
    assert action == "Pick" and parent is mcor
    assert qc.experiments["status"].tolist() == ["correcting with rigid"]

    # Sent to the cluster: said over the window
    boxes = [box for box in mcor.findChildren(QMessageBox) if box.isVisible()]
    assert len(boxes) == 1 and boxes[0].text().startswith("Sent to the cluster")
    boxes[0].close()

    # What was set aside goes with the pick
    from odyn import Database

    calls = Database(tmp_path / "data").latest_calls("pick_mcor_candidate")
    assert calls["input_set_aside"].iloc[0] == ["test"]
    assert mcor.state.text().startswith("Status: correcting with rigid;")

    # Refused while a whole correction has not ended
    sent = cluster_folder(tmp_path / "data") / "mcor" / f"{group_id}.json"
    sent.write_text(json.dumps({**json.loads(sent.read_text()), "whole": True}))
    log = sent.parent.parent / "jobs" / f"odyn-mcor-main-g{group_id}-11.out"
    log.write_text("started  ...\n")
    mcor.pick_action.trigger()
    settle(window.context)
    boxes = [box for box in mcor.findChildren(QMessageBox) if box.isVisible()]
    assert len(boxes) == 1 and "has not finished yet" in boxes[0].text()
    boxes[0].close()
    mcor.close()


def test_an_experiment_with_no_finished_test_is_not_compared(window, tmp_path):
    import gc

    db, group_id = a_tested_experiment(tmp_path / "data")
    top = db.groups[group_id].mcor_test_folder()
    del db
    gc.collect()

    # Only the sweep's run left, unfinished, and nothing to download
    import shutil

    from odyn.cluster import cluster_folder

    cluster = cluster_folder(tmp_path / "data")
    (cluster / "mcor" / f"{group_id}.json").unlink()
    (cluster / "jobs" / f"odyn-sweep-main-g{group_id}-20.out").unlink()
    shutil.rmtree(top)

    window.context.set_user("tester")
    qc = window.qc
    qc.reload()
    settle(window.context)
    row = qc.experiments.iloc[0]
    assert [row["status"], row["runs"], row["local copy"]] == [
        "waiting for the cluster",
        "0 of 2 finished",
        "",
    ]
    qc.tests_view.selectRow(0)
    assert not qc.download.isEnabled() and not qc.compare.isEnabled()
    assert qc.download_all.text() == "Download all"


def test_another_user_never_sees_the_rows_of_the_one_before(
    window, tmp_path, monkeypatch
):
    import gc

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, _ = a_tested_experiment(tmp_path / "data", user="tester")
    del db
    gc.collect()

    window.context.set_user("tester")
    qc = window.qc
    window.show_page(qc)
    qc.reload()
    settle(window.context)
    assert len(qc.experiments) == 1

    # Emptied at once, before the next user's rows are loaded
    window.context.set_user("carol")
    assert qc.experiments.empty
    assert qc._model(qc.tests_view).rowCount() == 0
    assert not qc.compare.isEnabled()
    assert qc.who.text() == "Loading the to-do list of carol"
    settle(window.context)
    assert qc.experiments.empty

    # A reload overtaken by a change of user drops its rows: none shown
    model = qc._model(qc.tests_view)
    shown = []
    set_frame = model.set_frame
    monkeypatch.setattr(
        model, "set_frame", lambda frame: shown.append(len(frame)) or set_frame(frame)
    )
    window.context.set_user("tester")
    window.context.set_user("carol")
    settle(window.context)
    assert qc.experiments.empty and not any(shown)


def test_everyones_experiments_are_shown_to_check(window, tmp_path, monkeypatch):
    import gc

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path / "data", user="tester")
    del db
    gc.collect()

    # None of theirs, until they ask for all users experiments
    window.context.set_user("carol")
    qc = window.qc
    window.show_page(qc)
    qc.reload()
    settle(window.context)
    assert qc.experiments.empty

    qc.everyone.setChecked(True)
    settle(window.context)
    assert qc.experiments["path"].tolist() == ["20260304/m1/e1"]
    assert qc.experiments["user"].tolist() == ["tester"]
    model = qc._model(qc.tests_view)
    assert list(model._frame.columns)[:3] == ["submitted_at", "path", "user"]

    # The other tabs too: what was added, and its annotations to fill
    assert qc.mine[["path", "user"]].values.tolist() == [["20260304/m1/e1", "tester"]]
    added = qc._model(qc.recordings_view)._frame
    assert list(added.columns)[:2] == ["submitted_at", "user"]
    annotations = qc._model(qc.missing_view)._frame
    assert set(annotations["user"]) <= {"tester"}

    qc.everyone.setChecked(False)
    settle(window.context)
    assert qc.experiments.empty

    # A job run for the owner (a pick's whole correction) leaves it theirs
    import json

    from odyn.cluster import cluster_folder

    sent = cluster_folder(tmp_path / "data") / "mcor" / f"{group_id}.json"
    sent.write_text(json.dumps({**json.loads(sent.read_text()), "user": "carol"}))
    qc.reload()
    settle(window.context)
    assert qc.experiments.empty  # carol's own list: still none

    window.context.set_user("tester")
    settle(window.context)
    assert qc.experiments["user"].tolist() == ["tester"]
    assert "test" in [run.name for run in qc.experiments.iloc[0]["finished"]]


def test_an_experiment_with_only_its_full_run_is_checked(window, tmp_path, monkeypatch):
    import gc
    import json

    from odyn.cluster import cluster_folder

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path / "data")
    top = db.groups[group_id].mcor_test_folder()
    del db
    gc.collect()

    # The whole rigid correction finished, with its full run; no sweep yet
    cluster = cluster_folder(tmp_path / "data")
    sent = cluster / "mcor" / f"{group_id}.json"
    sent.write_text(json.dumps({**json.loads(sent.read_text()), "whole": True}))
    (cluster / "jobs" / f"odyn-sweep-main-g{group_id}-20.out").unlink()
    a_video(top / "full" / "G_e1_raw.avi")
    a_video(top / "full" / "G_e1_mcor.avi")

    window.context.set_user("tester")
    qc = window.qc
    window.show_page(qc)
    qc.reload()
    settle(window.context)
    assert qc.experiments.iloc[0]["finished"] == []
    qc.tests_view.selectRow(0)
    qc.download.click()
    wait_until(lambda: qc.downloads is None)
    settle(window.context)
    qc.tests_view.selectRow(0)
    assert qc.compare.isEnabled()

    # Opened on the raw check; the tests say none finished
    qc.compare.click()
    mcor = next(iter(qc.windows.values()))
    assert mcor.raw_action.isChecked()
    assert "No test run finished yet" in mcor.missing.text()
    mcor.close()


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
    form = panels.add_card()

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


def test_panels_pane_adds_a_panel_then_batches_of_its_vials(
    window, tmp_path, monkeypatch
):
    panels = window.panels
    window.show_page(panels)
    asked = []
    monkeypatch.setattr(
        window.context, "confirm", lambda *args: asked.append(args) or True
    )

    # a new panel: its vials are set up freely, positions in order
    panels.form.set_values({"panel_name": "3 Odors", "made_on": "2026-08-06"})
    for odor_id in (1, 17, 0):
        card = panels.add_card()
        card.delivered.setCurrentIndex(card.delivered.findData(odor_id))
    panels.cards[1].set_targets({4: 44.2, 10: 13.3})
    assert [c.position.value() for c in panels.cards] == [1, 2, 3]
    panels._save()
    settle(window.context)

    db = Database(tmp_path / "data")
    positions = db.panel_positions.set_index("vial_position")["odor_id"]
    assert positions.to_dict() == {1: 1, 2: 17, 3: 0}
    first = db.vials
    assert len(first) == 3 and not asked  # a new panel asks nothing

    # once saved it is loaded: one locked, checked box per position, with the
    # last recipe, and no adding or removing vials
    assert [c.title() for c in panels.cards][1] == "Position 2: Odor 17 - alpha"
    assert all(c.isChecked() and not c.delivered.isEnabled() for c in panels.cards)
    assert panels.cards[1].components[0].target.value() == pytest.approx(44.2)
    assert not panels.add_vial.isEnabled()
    assert "already exists" in panels.form.rows["panel_name"].note.text()

    # a batch remakes only the checked vials
    panels.cards[0].setChecked(False)
    panels.form.rows["made_on"].set_value("2026-10-06")
    panels._save()
    settle(window.context)
    assert "2 of its 3 vials were included (made on 2026-10-06)" in asked[-1][1]
    later = Database(tmp_path / "data").vials
    assert len(later) == 5
    assert sorted(later.loc[later.index.difference(first.index), "vial_position"]) == [
        2,
        3,
    ]

    # the table shows each position's newest vial
    shown = panels.table.model().frame
    assert list(shown["vial_position"]) == [1, 2, 3]

    # What goes into the olfactometer first, no database ids
    model = panels.table.model()
    headers = [
        model.headerData(i, Qt.Orientation.Horizontal)
        for i in range(model.columnCount())
    ]
    assert headers[:4] == ["vial_position", "odor_id", "odor_name", "panel_name"]
    assert "vial_id" not in headers and "panel_id" not in headers
    assert [str(d)[:10] for d in shown["made_on"]] == [
        "2026-08-06",
        "2026-10-06",
        "2026-10-06",
    ]

    # Clear and a new name unlock it; a stored name typed and saved at once
    # (never loaded) is refused rather than guessed at
    panels.clear()
    assert panels.add_vial.isEnabled() and panels.cards == []
    panels.form.set_values({"panel_name": "3 Odors"})
    panels._save()
    assert "already exists" in window.context.last_report[1]

    # clicking a vial in the table loads its panel
    panels.table.clicked.emit(panels.table.model().index(0, 0))
    assert len(panels.cards) == 3 and panels.locked


def test_the_wheel_scrolls_the_form_and_never_changes_a_value(window):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    panels = window.panels
    window.show_page(panels)
    for _ in range(4):
        panels.add_card()
    QApplication.processEvents()

    spin = panels.cards[0].vial_sccm
    bar = panels.split.form.verticalScrollBar()
    wait_until(lambda: bar.maximum() > 0)  # four vials do not fit: it scrolls
    before, value = bar.value(), spin.value()

    down = QWheelEvent(
        QPointF(5, 5),
        QPointF(spin.mapToGlobal(QPoint(5, 5))),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(spin, down)

    assert spin.value() == value and bar.value() > before


def test_panel_names_are_the_same_whatever_their_case(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.add_panel(panel_name="16 Odors", positions={1: 1})

    with pytest.raises(ValueError, match="already a panel named '16 Odors'"):
        db.add_panel(panel_name="16 odors", positions={1: 1})

    db.add_vials(  # and a batch finds it however it is typed
        panel_name="16 ODORS",
        vials=[
            {
                "vial_position": 1,
                "vial_sccm": 200,
                "total_sccm": 1500,
                "total_volume_ml": 5,
            }
        ],
    )
    window.show_page(window.panels)
    window.context.changed.emit()
    settle(window.context)

    panels = window.panels
    name = panels.form.rows["panel_name"]
    name.editor.setText("16 odors")
    assert "already exists" in name.note.text()
    name.editor.editingFinished.emit()
    assert panels.locked and panels.form.values()["panel_name"] == "16 Odors"


def test_odors_pane_adds_a_mix_from_its_components(window, tmp_path):
    odors = window.odors
    window.show_page(odors)
    composition = odors.composition
    assert composition.none.placeholderText() == "Not a mix of odors"

    labels = {number: label for label, number in odors.chemicals.items()}
    odors.form.set_values({"odor_id": 45, "odor_name": "sigma"})
    composition.add()
    composition.add()
    composition.rows[0].set_value(labels[4])
    composition.rows[1].set_value(labels[10])
    odors._save()
    settle(window.context)

    db = Database(tmp_path / "data")
    assert db.odors.loc[45, "mix"]
    assert sorted(db.odor_components.loc[45, "component_id"]) == [4, 10]

    shown = odors.table.model().frame
    assert shown.loc[45, "components"] == "4 1-butanol + 10 (-) alpha pinene"
    assert shown.loc[1, "components"] == ""

    # once saved it is loaded: its name and what it is made of are locked
    assert not odors.form.rows["odor_name"].editor.isEnabled()
    assert not any(row.isEnabled() for row in composition.rows)
    assert not odors.add_component.isEnabled()


def test_odors_pane_never_renames_a_stored_odor(window, tmp_path, monkeypatch):
    odors = window.odors
    monkeypatch.setattr(window.context, "confirm", lambda *args: True)
    stored = Database(tmp_path / "data").odors.loc[1, "odor_name"]

    # typing a stored number loads it, name locked, with a note
    number = odors.form.rows["odor_id"]
    number.editor.setText("1")
    number.editor.editingFinished.emit()
    assert odors.form.values()["odor_name"] == stored
    assert not odors.form.rows["odor_name"].editor.isEnabled()
    assert number.note.text() == "Odor ID already exists (saving will overwrite data)"

    # its vapor pressure can be overwritten
    odors.form.rows["vapor_pressure_mmhg"].set_value(12.5)
    odors._save()
    settle(window.context)
    db = Database(tmp_path / "data")
    assert db.odors.loc[1, "vapor_pressure_mmhg"] == pytest.approx(12.5)
    assert db.odors.loc[1, "odor_name"] == stored

    # a new number unlocks the form, without the other odor's name
    number.editor.setText("46")
    number.editor.editingFinished.emit()
    assert odors.form.rows["odor_name"].editor.isEnabled()
    assert "odor_name" not in odors.form.values()

    # a stored number with another name, saved at once, is refused
    odors.form.set_values({"odor_id": 1, "odor_name": "renamed"})
    odors._save()
    settle(window.context)
    assert window.context.last_report == (
        "Odor",
        "Cannot change the name and composition of an odor",
    )
    assert Database(tmp_path / "data").odors.loc[1, "odor_name"] == stored


def test_tables_pane_lists_every_method_called(window):
    tables = window.tables
    names = [tables.picker.itemText(i) for i in range(tables.picker.count())]
    assert "method_calls:Database.set_mouse" not in names

    window.show_page(window.mice)
    window.mice.form.set_values({"mouse_id": "m3"})
    window.mice._save()
    settle(window.context)

    # The Tables page catches up when it is shown
    window.show_page(tables)
    settle(window.context)
    names = [tables.picker.itemText(i) for i in range(tables.picker.count())]
    assert "method_calls:Database.set_mouse" in names

    tables.picker.setCurrentText("method_calls:Database.set_mouse")
    settle(window.context)
    frame = tables.view.model().frame
    assert list(frame["input_mouse_id"]) == [3]  # read as typed: 'm3' is 3

    # the row detail lays the JSON out in a box that wraps
    tables._row_clicked(tables.view.model().index(0, 0))
    shown = tables.detail.values["environment"]
    assert isinstance(shown, QPlainTextEdit) and shown.isReadOnly()
    assert "\n" in shown.toPlainText()


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
    db.add_panel(panel_name="rack A", positions={1: 1, 2: 2, 3: 0})
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

    sessions = window.sessions
    sessions.new()
    settle(window.context)
    picker = sessions.session_page.vial_list
    assert picker.table.rowCount() == 0

    picker.panel.setCurrentIndex(picker.panel.findData(1))
    # ordered by position, the remade vial 2 replacing the old one, all installed
    assert [picker.table.item(r, 1).text() for r in range(3)] == ["1", "2", "3"]
    assert picker._rows == [old[1], new[0], old[2]]
    assert picker.checked() == picker._rows
    assert sorted(sessions.current.vial_ids) == sorted(picker._rows)

    from PySide6.QtCore import Qt

    picker.table.item(2, 0).setCheckState(Qt.CheckState.Unchecked)  # mineral oil out
    assert sorted(sessions.current.vial_ids) == sorted([old[1], new[0]])

    # shown again, the draft's vials bring their panel back with the box as left
    sessions.refresh_current()
    assert picker.panel_id() == 1
    assert picker.checked() == [old[1], new[0]]


def test_session_and_recorder_share_one_selection(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=5)
    window.context.changed.emit()
    settle(window.context)

    sessions, recorder, context = window.sessions, window.recorder, window.context
    window._add_session()
    settle(window.context)
    draft = sessions.current
    assert context.draft is draft and context.part is None
    assert "m? " in recorder.session_label.text()

    # picking the mouse shows in the recorder at once, and on the right bar
    sessions.session_page.mouse.setCurrentText("m5")
    assert draft.mouse_id == 5
    assert recorder.session_label.text() == draft.label
    assert window.today[(id(draft), None)].toolTip() == draft.where
    assert window.selection.text() == draft.where

    # nothing to record into until a part is selected
    assert recorder.part.text() == ""
    recorder.start()
    assert recorder.recording is None
    assert "Select an experiment" in context.last_report[1]

    # a part added is selected, and is what the recorder records
    sessions.add_experiment()
    sessions.add_experiment()
    assert context.part == "e2" and recorder.part.text() == "e2"

    assert window.view == "part"
    assert window.group.checkedAction() is window.today[(id(draft), "e2")]

    # opening another part moves the recorder; and back (the session
    # page alone selects no part to record)
    window.open_session(draft)
    assert recorder.part.text() == ""
    window.open_part(draft, "e1")
    assert context.part == "e1" and recorder.part.text() == "e1"
    window.open_part(draft, "e2")
    assert recorder.part.text() == "e2"
    assert window.group.checkedAction() is window.today[(id(draft), "e2")]
    assert sessions.part_page.name == "e2"

    # removing the part shown drops it everywhere, no phantom name
    sessions.remove_part()
    assert [str(e["name"]) for e in draft.parts] == ["e1"]
    assert context.part == "e1" and recorder.part.text() == "e1"
    assert list(window.today) == [(id(draft), None), (id(draft), "e1")]
    assert sessions.part_page.name == "e1"

    # and removing the last one falls back to the session page
    sessions.remove_part()
    assert window.view == "session" and context.part is None
    assert recorder.part.text() == ""
    assert window.pages.currentWidget() is sessions.session_page
    assert window.group.checkedAction() is window.today[(id(draft), None)]


def with_mouse(window, mouse: int = 7):
    """A new session, its mouse picked (folders need it)."""
    window._add_session()
    draft = window.sessions.current
    draft.identify(mouse_id=mouse)
    window.context.announce_draft()

    return draft


def test_an_experiment_page_notes_conditions_by_acquisition(window):
    from odyn.app.forms import GrabTimingForm

    draft = with_mouse(window)
    window.sessions.add_experiment()
    shown = window.sessions.part_page.shown
    conditions = shown.conditions
    assert conditions.choices == ["baseline", "ket/xyl"]
    assert conditions.none.placeholderText() == "none noted"

    # each one follows the last: the next acquisitions, the next condition
    conditions.add()
    conditions.add()
    conditions.rows[1].end.setValue(12)
    assert draft.part("e1")["conditions"] == [
        {"start": 1, "end": 1, "condition": "baseline"},
        {"start": 2, "end": 12, "condition": "ket/xyl"},
    ]

    # rows of the part's form, below its fields, under one titled
    # header naming the columns
    layout = shown.form.layout()
    assert layout.labelForField(conditions.header).text() == "Conditions"
    assert [layout.labelForField(r).text() for r in conditions.rows] == ["", ""]
    assert conditions.header.isVisibleTo(shown)

    conditions.remove(conditions.rows[0])
    assert draft.part("e1")["conditions"] == [
        {"start": 2, "end": 12, "condition": "ket/xyl"}
    ]
    conditions.remove(conditions.rows[0])
    assert "conditions" not in draft.part("e1")

    # a grab has one acquisition, so one condition, and the calculator for a
    # grab; the page is rebuilt for the type
    shown.type.setCurrentText("grab")
    shown = window.sessions.part_page.shown
    assert shown.conditions is None and shown.condition is not None
    assert isinstance(shown.timing, GrabTimingForm)
    assert window.sessions.part_page.box.title() == f"Grab {draft.where}/e1"
    shown.condition.setCurrentText("baseline")
    assert draft.part("e1")["exp_type"] == "grab"
    assert draft.part("e1")["conditions"] == [
        {"start": 1, "end": 1, "condition": "baseline"}
    ]

    # a z-stacks folder has no conditions to note
    window.sessions.add_zstacks()
    assert not isinstance(window.sessions.part_page.shown, ExperimentForm)


def test_add_experiment_numbers_and_makes_the_folders_in_one_click(window, tmp_path):
    sessions, share = window.sessions, tmp_path / "data"

    window._add_session()
    sessions.add_experiment()
    assert "Pick a mouse for the session first" in window.context.last_report[1]
    assert window.sessions.current.parts == []

    draft = window.sessions.current
    draft.identify(mouse_id=7, date="2026-03-04")
    window.context.announce_draft()
    session = share / "20260304" / "m7"

    sessions.add_experiment()
    assert window.context.part == "e1" and window.view == "part"
    assert sorted(p.name for p in (session / "e1").iterdir()) == [
        "processed",
        "raw",
        "sync",
    ]

    # a loop folder already on the share (an earlier session that day) is
    # skipped, and so is a removed loop's name
    (session / "e4").mkdir()
    sessions.add_experiment()
    assert window.context.part == "e5"
    sessions.remove_part("e5")
    sessions.add_experiment()
    assert window.context.part == "e6"

    # z-stacks: one folder, shown with a picture
    sessions.add_zstacks()
    assert sorted(p.name for p in (session / "zstacks").iterdir()) == [
        "processed",
        "raw",
    ]
    assert [p["name"] for p in draft.parts] == ["e1", "e6", "zstacks"]

    sessions.add_experiment()
    sessions.add_zstacks()  # again: selects them, adds nothing
    assert window.context.part == "zstacks" and len(draft.parts) == 4
    assert isinstance(sessions.part_page.shown, ZStacksForm)
    assert sessions.part_page.box.title() == f"Z-stacks {draft.where}/zstacks"

    # the folders are named for the mouse and the day, which are now fixed
    assert not sessions.session_page.mouse.isEnabled()

    with pytest.raises(ValueError, match="fixed"):
        draft.identify(mouse_id=8)


def test_zstacks_take_notes_before_any_tiff_is_copied(window, monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from odyn.app.forms import Form

    sessions = window.sessions
    window._add_session()
    draft = sessions.current
    draft.identify(mouse_id=1, date="2026-03-04")
    window.context.announce_draft()
    sessions.add_experiment()
    sessions.add_zstacks()

    # One z-stack to start with, nothing on the share needed
    form = sessions.part_page.shown
    assert isinstance(form, ZStacksForm)
    assert [box.title() for box in form.boxes] == ["Z-stack 1"]
    form.form.rows["objective"].editor.setCurrentText("10")
    assert draft.part("zstacks")["objective"] == 10

    # What is typed goes to the draft; a z-stack names its experiment too
    notes = form.boxes[0].findChild(Form)
    notes.rows["z_landmark"].editor.setText("surface")
    notes.rows["z_landmark"].editor.editingFinished.emit()
    form.boxes[0].findChild(QComboBox).setCurrentText("e1")
    assert draft.stacks("zstacks") == [{"z_landmark": "surface", "experiment": "e1"}]

    # Add z-stack lists another; removing the first moves it up
    form.add.click()
    assert [box.title() for box in form.boxes] == ["Z-stack 1", "Z-stack 2"]
    form.boxes[1].findChild(Form).rows["z_landmark"].editor.setText("deeper")
    form.boxes[1].findChild(Form).rows["z_landmark"].editor.editingFinished.emit()

    monkeypatch.setattr(window.context, "confirm", lambda *args: True)
    form._remove(0)
    assert [box.title() for box in form.boxes] == ["Z-stack 1"]
    assert draft.stacks("zstacks") == [{"z_landmark": "deeper"}]


def test_a_grab_is_recorded_like_a_loop_and_zstacks_are_not(window):
    sessions, recorder = window.sessions, window.recorder
    draft = with_mouse(window)
    sessions.add_experiment()
    draft.set_part("e1", exp_type="grab")
    window.page_actions[window.recorder].trigger()
    assert recorder.part.text() == "e1"

    recorder.start()
    wait_until(lambda: recorder.recording is not None and recorder.recording.total > 50)
    recorder.stop()
    wait_until(lambda: not recorder.copier.running)

    (record,) = draft.recordings("e1")
    assert record["share"].startswith(f"{draft.folder}/e1/sync/")

    sessions.add_zstacks()
    window.open_part(draft, "zstacks")
    recorder.start()
    assert recorder.recording is None
    assert "Z-stacks are not recorded" in window.context.last_report[1]


def test_quick_maths_warns_when_imaging_outlasts_the_trial(app):
    from odyn.app.forms import TimingForm

    form = TimingForm()
    assert not form.overrun.isVisibleTo(form)
    assert form.outputs["frames"].text() == "258"
    assert form.outputs["camera_rate"].text() == "12.86 Hz"
    # One program (56 min) and 15 s: the camera is stopped between them
    assert form.outputs["camera_frames"].text() == "43403"
    form.camera_extra.setValue(60)
    assert form.outputs["camera_frames"].text() == "43982"
    assert "camera_extra" not in str(form.values())  # not one of the loop's

    # a shown value is as tall as the boxes it follows from
    heights = {line.minimumHeight() for line in form.outputs.values()}
    assert heights == {form.frame_rate.sizeHint().height()}

    form.inputs["block_iti_s"].setValue(4)  # 5 + 4 = 9 s after the odor, wants 11
    assert form.overrun.isVisibleTo(form)
    assert "2 s too short" in form.overrun.text()

    form.inputs["block_post_odor_acq_s"].setValue(9)
    assert form.overrun.isVisibleTo(form)  # fits, but with nothing to spare
    assert form.overrun.text() == "Only 0 s between acquisitions"

    form.inputs["block_post_odor_acq_s"].setValue(8)
    assert "Only 1 s" in form.overrun.text()

    form.inputs["block_post_odor_acq_s"].setValue(7.5)
    assert not form.overrun.isVisibleTo(form)


def test_an_experiment_page_has_the_objective_in_its_form(window, tmp_path):
    sessions = window.sessions
    draft = with_mouse(window)
    settle(window.context)
    sessions.add_experiment()
    assert window.pages.currentWidget() is sessions.part_page

    tab = sessions.part_page.shown
    assert list(tab.form.rows)[0] == "objective"

    # a required field is marked by its label's color, not a star
    label = tab.form.layout().labelForField(tab.form.rows["objective"])
    assert label.text() == "Objective" and "color" in label.styleSheet()
    assert not hasattr(tab, "kind")

    tab.form.rows["objective"].editor.setCurrentText("20")
    assert draft.part("e1")["objective"] == 20
    assert draft.part("e1")["kind"] == "experiment"


def test_right_bar_lists_sessions_in_the_order_started(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=442)
    window.context.changed.emit()
    settle(window.context)

    window._add_session()
    first = window.sessions.current
    window.sessions.session_page.mouse.setCurrentText("m442")
    window.sessions.add_experiment()
    window._add_session()
    second = window.sessions.current
    settle(window.context)

    # a session is named by its mouse, or by its place before it has one
    assert list(window.today) == [
        (id(first), None),
        (id(first), "e1"),
        (id(second), None),
    ]
    assert window.sessions.label(first) == "m442"
    assert window.sessions.label(second) == "s2"
    assert window.today[(id(first), "e1")].toolTip() == f"{first.where}/e1"
    assert second.where.endswith("/m?")
    assert window.group.checkedAction() is window.today[(id(second), None)]

    # Remove acts on what is selected, named on the bottom bar
    window.open_part(first, "e1")
    assert window.selection.text() == f"{first.where}/e1"
    assert window.remove_action.statusTip() == f"Remove loop {first.where}/e1"
    window.remove_action.trigger()

    # its session is selected next, so a second click removes one thing only
    assert window.view == "session" and window.selection.text() == first.where
    assert list(window.today)[:2] == [(id(first), None), (id(second), None)]
    window.remove_action.trigger()
    settle(window.context)
    assert [d is second for d in window.sessions.drafts] == [True]
    assert window.context.draft is None and window.group.checkedAction() is None
    assert window.pages.currentWidget() is window.sessions.no_session_page
    assert not window.remove_action.isEnabled()
    assert not window.selection.isVisibleTo(window)


def test_check_opens_the_panel_and_the_log_follows_the_database(window):
    panel = window.panel
    assert not panel.is_open()

    window._add_session()
    window.sessions.check()
    settle(window.context)
    assert panel.is_open() and panel.current == panel.NAMES.index("Check")
    draft = window.sessions.current
    check_tab = panel.texts["Check"]
    assert check_tab.toPlainText().split("\n") == [
        f"CHECKED  {draft.label} (2 errors, 0 warnings)",
        "",
        "   user  didn't pick a user yet (Users page)",
        "",
        "session  didn't pick a mouse yet",
        "",
        " RESULT  fix errors to submit ✘",
    ]
    assert "font-weight" in check_tab.document().toHtml()

    # a write logs on the worker's thread; the line arrives in the Log tab,
    # its level and check mark in color rather than as terminal codes
    window.mice.form.set_values({"mouse_id": "m8"})
    window.mice._save()
    settle(window.context)
    log = panel.texts["Log"]
    assert "INFO Mouse 8 recorded" in log.toPlainText()
    assert "\033" not in log.toPlainText() and "[1;" not in log.toPlainText()
    assert "font-weight" in log.document().toHtml()

    # the arrow closes the panel; a tab opens it on that tab
    panel.toggle.click()
    assert not panel.is_open()
    panel.buttons.button(panel.NAMES.index("Log")).click()
    assert panel.is_open() and panel.current == panel.NAMES.index("Log")


def test_the_submit_report_ends_with_where_the_draft_went(window, tmp_path):
    from odyn.app.submit import Problem, SubmitResult

    window._add_session()
    draft = window.sessions.current
    draft.identify(mouse_id=3, date="2026-10-07")
    window.sessions.add_experiment()
    warning = Problem("e1", "no olfactometer Events file found", blocking=False)
    result = SubmitResult(draft.label, [warning], queued=tmp_path / "queued.json")
    window.sessions._submitted(draft, result)

    assert window.panel.texts["Check"].toPlainText().split("\n") == [
        f"CHECKED  {draft.label} (0 errors, 1 warning)",
        "",
        "     e1  no olfactometer Events file found",
        "",
        " RESULT  sent to the cluster (the QC page shows its progress) ✔",
        "",
        "     e1  copy TIFF files to 20261007/m3/e1/raw",
    ]

    # A queue that cannot be written stops it
    stopped = SubmitResult(draft.label, error="could not put it in the queue: full")
    window.sessions._submitted(draft, stopped)

    assert window.panel.texts["Check"].toPlainText().split("\n")[-1] == (
        " RESULT  stopped: could not put it in the queue: full (submit again after "
        "fixing it) ✘"
    )


def test_the_panel_can_take_the_height_and_double_click_maximizes(window):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    panel, splitter = window.panel, window.panel.splitter
    window.resize(1200, 700)
    window.show_page(window.qc)  # the page that used to need the most height
    panel.set_open(True)
    QApplication.processEvents()

    # Any page gives way: it scrolls instead
    total = sum(splitter.sizes())
    splitter.setSizes([0, total])
    QApplication.processEvents()
    assert splitter.sizes()[0] < 100
    assert window.pages.currentWidget() is window.qc

    splitter.setSizes([total - 200, 200])
    QApplication.processEvents()
    check = panel.buttons.button(panel.NAMES.index("Check"))

    def double_click():
        check.click()  # a double-click starts with a click
        point = QPointF(5, 5)
        QApplication.sendEvent(
            check,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonDblClick,
                point,
                check.mapToGlobal(point),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.processEvents()

    double_click()
    assert panel.maximized and panel.is_open()
    assert splitter.sizes()[0] < 100

    # A report keeps it as it is
    panel.show_text("Check", "RESULT  ready to submit")
    QApplication.processEvents()
    assert panel.maximized and splitter.sizes()[0] < 100

    double_click()
    assert not panel.maximized and panel.is_open()
    assert abs(splitter.sizes()[1] - 200) < 10

    # A click on the open tab leaves it as it is; another tab is shown
    check.click()
    assert panel.is_open() and panel.current == panel.NAMES.index("Check")
    panel.buttons.button(panel.NAMES.index("Log")).click()
    assert panel.is_open() and panel.current == panel.NAMES.index("Log")

    # The button maximizes and restores; closed, a tab opens the panel
    panel.maximize.click()
    QApplication.processEvents()
    assert panel.maximized and splitter.sizes()[0] < 100
    assert panel.maximize.toolTip() == "Restore panel"
    panel.maximize.click()
    QApplication.processEvents()
    assert not panel.maximized and abs(splitter.sizes()[1] - 200) < 10

    panel.toggle.click()
    assert not panel.is_open() and panel.property("open") is False
    check.click()
    assert panel.is_open() and panel.current == panel.NAMES.index("Check")


def test_bottom_bar_shows_what_is_hovered_and_what_the_app_says(window):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent

    button = window.left.widgetForAction(window.page_actions[window.mice])
    QApplication.sendEvent(
        button, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5))
    )
    assert window.tip.text() == "Mice"
    QApplication.sendEvent(button, QEvent(QEvent.Type.Leave))
    assert window.tip.text() == ""

    window.context.status.emit("Saved 3 annotations")
    assert window.tip.text() == "Saved 3 annotations"


def test_remove_marks_and_undo_brings_back(window, tmp_path):
    db = Database(tmp_path / "data", user="tester")
    db.set_mouse(mouse_id=442)
    window.context.changed.emit()
    settle(window.context)
    sessions, undo = window.sessions, window.undo_action

    window._add_session()
    draft = sessions.current
    date = draft.date.replace("-", "")
    sessions.session_page.mouse.setCurrentText("m442")
    sessions.add_experiment()
    draft.set_part("e1", goal="typed by hand")
    sessions.add_experiment()
    assert not undo.isEnabled()

    # removing only hides; the button says what would come back
    window.open_part(draft, "e1")
    sessions.remove_part()
    assert list(window.today) == [(id(draft), None), (id(draft), "e2")]
    assert undo.isEnabled()
    assert undo.statusTip() == f"Undo remove (loop {date}/m442/e1)"

    sessions.remove()
    assert sessions.drafts == [] and window.today == {}
    assert undo.statusTip() == f"Undo remove (session {date}/m442)"

    # undo in reverse order, each shown as it comes back
    undo.trigger()
    assert window.view == "session" and window.context.draft.path == draft.path
    draft = sessions.current
    assert list(window.today) == [(id(draft), None), (id(draft), "e2")]

    undo.trigger()
    assert window.view == "part" and window.context.part == "e1"
    assert draft.part("e1")["goal"] == "typed by hand"
    assert (id(draft), "e1") in window.today
    assert not undo.isEnabled()


def test_adding_removed_zstacks_brings_them_back_as_they_were(window):
    sessions, undo = window.sessions, window.undo_action
    window._add_session()
    draft = sessions.current
    draft.identify(mouse_id=1, date="2026-03-04")
    window.context.announce_draft()
    sessions.add_experiment()
    sessions.add_zstacks()
    draft.set_stack("zstacks", 0, note="first one")

    sessions.remove_part("e1")
    sessions.remove_part("zstacks")
    assert undo.statusTip().startswith("Undo remove (z-stacks")

    # Add brings them back, notes included, with no question asked
    sessions.add_zstacks()
    assert window.context.last_report is None
    assert window.context.part == "zstacks"
    assert draft.stacks("zstacks") == [{"note": "first one"}]

    # Undo skips them and goes on to the loop removed before
    assert undo.statusTip().startswith("Undo remove (loop")
    undo.trigger()
    assert draft.part("e1") is not None and not undo.isEnabled()


def test_a_folder_that_cannot_be_made_is_said_in_a_box(window, monkeypatch):
    import odyn.app.pane_session as pane_session

    def refuse(folder, kind):
        raise PermissionError(f"no access to {folder}")

    monkeypatch.setattr(pane_session, "make_part_folders", refuse)
    sessions = window.sessions
    window._add_session()
    sessions.current.identify(mouse_id=1, date="2026-03-04")
    window.context.announce_draft()
    sessions.add_experiment()

    title, message = window.context.last_report
    assert title == "Experiment" and "no access to" in message
    assert message.startswith("Could not create 20260304/m1/e1 on the server")
    assert sessions.current.part("e1") is not None  # listed all the same


def test_guide_points_at_every_bar_button_with_no_session(window):
    guide = window.guide
    assert not guide.isVisible()

    window._after_login()  # no session yet
    QApplication.processEvents()
    assert guide.isVisible()

    pointers = guide.pointers()
    texts = [p.text for p in pointers]
    assert texts[:6] == [
        "Users",
        "Mice",
        "Odors",
        "Panels",
        "Tables",
        "Quality Control",
    ]
    assert "Add session" in texts and "Undo the last remove" in texts
    assert "Record the selected experiment" in texts

    # each arrow ends right beside its own button, and no two labels overlap
    buttons = {
        a.toolTip(): window.left.widgetForAction(a)
        for a in window.page_actions.values()
    }
    users = buttons["Users"]
    corner = users.mapTo(window.centralWidget(), users.rect().topLeft())

    from odyn.app.guide import TIP_GAP_PX

    assert abs(pointers[0].end.x() - (corner.x() + users.width())) <= TIP_GAP_PX + 1
    assert corner.y() <= pointers[0].end.y() <= corner.y() + users.height()
    labels = [p.label for p in pointers]
    assert not any(
        a.intersects(b) for i, a in enumerate(labels) for b in labels[i + 1 :]
    )

    # a short window drops labels rather than piling them up
    window.resize(1000, 380)
    QApplication.processEvents()
    labels = [p.label for p in guide.pointers()]
    assert not any(
        a.intersects(b) for i, a in enumerate(labels) for b in labels[i + 1 :]
    )

    # any other page hides it
    window.page_actions[window.mice].trigger()
    assert not guide.isVisible()


def test_grab_quick_maths(app):
    from odyn.app.forms import GrabTimingForm

    form = GrabTimingForm()
    form.duration.setValue(30)
    form.frame_rate.setValue(15)

    assert form.outputs["frames"].text() == "450"
    assert form.outputs["camera_rate"].text() == "15 Hz"
    assert form.outputs["camera_frames"].text() == "675"  # and 15 s more
    form.camera_extra.setValue(60)
    assert form.outputs["camera_frames"].text() == "1350"


def test_expired_odors_are_named_by_id():
    from datetime import date

    import pandas as pd

    from odyn.app.forms import expired_note

    vials = pd.DataFrame(
        {
            "panel_id": 1,
            "vial_position": [1, 2, 3, 4, 5],
            "odor_id": [3, 4, 5, 9, 12],
            "odor_name": list("abcde"),
            "made_on": ["2026-09-01", "2026-09-01", "2026-09-01", "2026-10-10", None],
        },
        index=pd.Index([11, 12, 13, 14, 15], name="vial_id"),
    )
    today = date(2026, 10, 14)

    assert expired_note(vials, [11, 12, 13, 14], today) == (
        "Odors 3-5 in this panel are expired!",
        "warning",
    )
    assert expired_note(vials, [11, 14], today) == (
        "Odor 3 in this panel is expired!",
        "warning",
    )
    assert expired_note(vials, [11, 12], today) == (
        "All odors in this panel are expired!",
        "error",
    )
    # Fresh, undated or not installed: nothing to say
    assert expired_note(vials, [14, 15], today) == ("", None)
    assert expired_note(vials, [], today) == ("", None)
    # Exactly two weeks old is still fine
    assert expired_note(vials, [14], date(2026, 10, 24)) == ("", None)


def test_the_panel_picker_warns_about_its_installed_vials(app):
    from datetime import date

    import pandas as pd

    from odyn.app.forms import PanelPicker

    vials = pd.DataFrame(
        {
            "panel_id": 1,
            "vial_position": [1, 2],
            "odor_id": [3, 4],
            "odor_name": ["a", "b"],
            "made_on": ["2026-09-01", "2026-10-10"],
        },
        index=pd.Index([11, 12], name="vial_id"),
    )
    picker = PanelPicker()
    picker.today = date(2026, 10, 14)
    picker.set_panels([(1, "2 Odors")], vials)
    picker.show_vials([11, 12], 1)

    assert picker.expired.text() == "Odor 3 in this panel is expired!"

    # Taking the old vial out of the olfactometer clears it
    picker.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert picker.expired.text() == "" and picker.expired.isHidden()


def test_a_table_sorts_by_any_column_and_keeps_its_rows(app):
    import pandas as pd

    frame = pd.DataFrame(
        {"name": ["b", "A", None, "c"], "n": [2.0, None, 1.0, 10.0]},
        index=pd.Index([10, 11, 12, 13], name="mouse_id"),
    )
    model = FrameModel(frame)
    column = model._columns.index

    # Its own order until a header is clicked
    assert [model.index_value(r) for r in range(4)] == [10, 11, 12, 13]

    model.sort(column("n"), Qt.SortOrder.AscendingOrder)
    assert [model.index_value(r) for r in range(4)] == [12, 10, 13, 11]  # blank last
    assert model.row_values(0)["n"] == 1.0
    assert [model.source_row(r) for r in range(4)] == [2, 0, 3, 1]

    model.sort(column("name"), Qt.SortOrder.DescendingOrder)
    assert [model.row_values(r)["name"] for r in range(3)] == ["c", "b", "A"]
    assert model.index_value(3) == 12  # the blank, last either way

    # A reload keeps the column picked; a negative one is the frame's order
    model.set_frame(frame.iloc[::-1])
    assert [model.row_values(r)["name"] for r in range(3)] == ["c", "b", "A"]
    assert model.index_value(3) == 12  # the blank, last
    model.sort(-1)
    assert [model.index_value(r) for r in range(4)] == [13, 12, 11, 10]


def test_a_sorted_qc_list_still_picks_the_row_selected(window):
    import pandas as pd

    from odyn.app.pane_qc import EXPERIMENT_COLUMNS, EXPERIMENT_HIDDEN

    qc = window.qc
    rows = [
        dict.fromkeys([*EXPERIMENT_COLUMNS, *EXPERIMENT_HIDDEN], "")
        | {"path": path, "local copy": "no", "folders": [], "bytes": 0}
        for path in ("b", "c", "a")
    ]
    qc.experiments = pd.DataFrame(rows)
    qc._model(qc.tests_view).set_frame(qc.experiments[EXPERIMENT_COLUMNS])

    view = qc.tests_view
    view.sortByColumn(EXPERIMENT_COLUMNS.index("path"), Qt.SortOrder.AscendingOrder)
    assert [view.model().row_values(r)["path"] for r in range(3)] == ["a", "b", "c"]
    view.selectRow(1)
    assert [label for label, _ in qc._picked()] == ["b"]


def test_a_date_shows_without_its_midnight_time():
    import pandas as pd

    from odyn.app.models import show

    assert show(pd.Timestamp("2026-10-14")) == "2026-10-14"
    assert show(pd.Timestamp("2026-10-14 09:30:05")) == "2026-10-14 09:30:05"
    assert show(pd.NaT) == ""


def test_an_experiments_status_follows_its_pick():
    import pandas as pd

    from odyn.app.pane_qc import _picked_status

    pick = {"call_id": 5, "run": "s128", "correct": True}

    def whole(state, call_id=5):
        return pd.DataFrame({"pick_call_id": [call_id], "state": [state]})

    assert (
        _picked_status(pick, whole("finished", call_id=None)) == "correcting with s128"
    )
    assert _picked_status(pick, whole("waiting")) == "correcting with s128"
    assert _picked_status(pick, whole("not finished")) == "correcting with s128"
    assert _picked_status(pick, whole("stopped")) == "correction with s128 stopped"
    assert _picked_status(pick, whole("finished")) == "ready for final check"

    # The last whole correction had its settings already
    same = {**pick, "correct": False}
    assert (
        _picked_status(same, whole("finished", call_id=None)) == "ready for final check"
    )


def test_a_download_says_how_far_it_is():
    from odyn.app.pane_qc import _download_line, _duration

    assert _download_line("e1", 0, 560e6, 100) == "Downloading e1: 0 B of 560.0 MB"
    assert _download_line("e1 (1 of 2)", 120e6, 560e6, 10_000) == (
        "Downloading e1 (1 of 2): 120.0 MB of 560.0 MB, 12.0 MB/s, about 37 s left"
    )
    assert [_duration(s) for s in (0.2, 40, 200, 4900)] == [
        "1 s",
        "40 s",
        "3 min",
        "1 h 22 min",
    ]


def test_a_download_can_be_cancelled(tmp_path):
    from odyn.app.pane_qc import CANCELLED, Downloads

    main = tmp_path / "share"
    folder = main / "cache" / "mcor-tests" / "e1_g1"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text("{}")
    downloads = Downloads(main, [("e1", [folder])], tmp_path / "base")
    ended = []
    downloads.done.connect(lambda count, problem: ended.append((count, problem)))
    downloads.isInterruptionRequested = lambda: True  # Cancel, at the first report
    downloads.run()  # here, not in a thread
    assert ended == [(0, CANCELLED)]


def test_a_database_this_computer_cannot_write_is_said_up_front(app, tmp_path):
    from odyn.app.worker import write_problem

    Database(tmp_path / "data", can_create=True, user="tester")
    worker = DatabaseWorker(tmp_path / "data", None, user="tester")
    opened = []
    worker.opened.connect(lambda: opened.append(True))
    worker.start()
    wait_until(lambda: bool(opened))
    assert worker.read_only == ""  # this one can be written
    assert write_problem(tmp_path / "missing.db") == "No such file or directory"

    # As SMB refuses it when the lab's group has no write access
    worker.read_only = "Permission denied"
    window = MainWindow(worker, drafts=tmp_path / "drafts")
    assert window.read_only_label.isVisibleTo(window)
    # At the right, before the cache and the user
    window.show()
    app.processEvents()
    labels = [window.read_only_label, window.cache_label, window.user_label]
    xs = [label.mapTo(window, label.rect().topLeft()).x() for label in labels]
    assert xs == sorted(set(xs))
    title, message = window.context.last_report
    assert title == "Read-only database" and "(Permission denied)" in message

    worker.stop()
    window.close()


def test_stopped_jobs_are_run_again_from_the_qc_page(window, tmp_path, monkeypatch):
    import gc
    import json

    from odyn.cluster import cluster_folder

    monkeypatch.setenv("PATH", "")  # off the cluster: the recorded states
    db, group_id = a_tested_experiment(tmp_path / "data")
    del db
    gc.collect()
    folder = cluster_folder(tmp_path / "data")

    # The tick recorded the cluster test and the sweep's piecewise run stopped
    sent = folder / "mcor" / f"{group_id}.json"
    sent.write_text(json.dumps(json.loads(sent.read_text()) | {"state": "stopped"}))
    log = folder / "jobs" / f"odyn-mcor-main-g{group_id}-11.out"
    log.write_text("started\nKilled\n")  # it did not finish after all
    sweep = folder / "sweeps" / f"g{group_id}.json"
    content = json.loads(sweep.read_text())
    sweep.write_text(json.dumps(content | {"states": {"21_1": "stopped"}}))
    asked = []
    monkeypatch.setattr(
        window.context,
        "confirm",
        lambda title, message, action, parent=None: asked.append(message) or True,
    )

    window.context.set_user("tester")
    qc = window.qc
    window.show_page(qc)
    qc.reload()
    settle(window.context)
    row = qc.experiments.iloc[0]
    assert row["runs"] == "1 of 3 finished, 2 stopped"
    qc.tests_view.selectRow(0)
    assert qc.run_again.isEnabled()

    qc.run_again.click()
    settle(window.context)
    assert asked == ["Run again the stopped jobs of 20260304/m1/e1: s128_o64_d6, test?"]
    calls = Database(tmp_path / "data").latest_calls("run_again")
    assert calls["user"].iloc[0] == "tester"
    assert calls["output_runs"].iloc[0] == ["correction", "s128_o64_d6"]
