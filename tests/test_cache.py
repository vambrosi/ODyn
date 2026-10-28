"""`odyn.app.cache`: copies of test motion corrections on this computer."""

import os

import pytest

from pathlib import Path

from odyn.app.cache import (
    HERE,
    Listing,
    NEWER,
    NOT_HERE,
    cache_root,
    cache_size,
    delete,
    download,
    human_size,
    local_folder,
    remote_size,
    small_files,
    state,
)


def a_test(main: Path) -> Path:
    """A test folder with its small files, a large one, and a run inside."""
    folder = main / "projects" / "p" / "cache" / "mcor-tests" / "e1_g1"
    run = folder / "s128_o64_d6"
    run.mkdir(parents=True)

    for name, size in (
        ("Group_1_e1_raw_test.avi", 300),
        ("shifts.png", 20),
        ("reference.tif", 40),
        ("flow_average.tif", 40),
        ("summary.csv", 5),
        ("metrics.json", 5),
        ("shifts.npz", 900),
        ("side_by_side.tif", 900),
    ):
        (folder / name).write_bytes(b"x" * size)

    (run / "Group_1_e1_test.avi").write_bytes(b"y" * 100)

    return folder


def test_only_a_tests_own_small_files_are_copied(tmp_path):
    main, base = tmp_path / "share", tmp_path / ".odyn"
    folder = a_test(main)

    names = [path.name for path in small_files(folder)]
    assert "shifts.npz" not in names and "side_by_side.tif" not in names
    assert len(names) == 6
    assert remote_size(folder) == 410

    assert state(main, folder, base) == NOT_HERE
    heard = []
    local = download(
        main, folder, base, progress=lambda name, n: n or heard.append(name)
    )

    assert local == local_folder(main, folder, base)
    assert sorted(path.name for path in local.iterdir()) == sorted(names)
    assert len(heard) == 6
    assert state(main, folder, base) == HERE
    assert not (local / "s128_o64_d6").exists()  # the run is its own test

    # Again: nothing to copy
    heard.clear()
    download(main, folder, base, progress=lambda name, n: n or heard.append(name))
    assert heard == []


def test_a_reload_lists_each_folder_on_the_share_once(tmp_path, monkeypatch):
    import odyn.app.cache as module

    main, base = tmp_path / "share", tmp_path / ".odyn"
    folder = a_test(main)
    download(main, folder, base)
    listed = []
    scan = module._scan
    monkeypatch.setattr(
        module, "_scan", lambda path: listed.append(Path(path)) or scan(path)
    )

    # Every question about the folder, from one listing of it on the share
    listing = Listing()
    assert len(small_files(folder, listing)) == 6
    assert remote_size(folder, listing) == 410
    assert state(main, folder, base, listing) == HERE
    assert state(main, folder, base, listing) == HERE
    assert listed.count(folder) == 1  # the copy here is listed for each state

    # The same answers as asked one by one
    assert small_files(folder) == small_files(folder, Listing())
    assert state(main, folder, base) == HERE


def test_a_file_changed_on_the_share_is_copied_again(tmp_path):
    main, base = tmp_path / "share", tmp_path / ".odyn"
    folder = a_test(main)
    download(main, folder, base)

    changed = folder / "summary.csv"
    changed.write_bytes(b"z" * 7)
    os.utime(changed, (changed.stat().st_atime, changed.stat().st_mtime + 60))
    assert state(main, folder, base) == NEWER

    heard = []
    download(main, folder, base, progress=lambda name, n: n or heard.append(name))
    assert heard == ["summary.csv"]
    assert state(main, folder, base) == HERE


def test_two_main_folders_with_the_same_paths_stay_apart(tmp_path):
    base = tmp_path / ".odyn"
    real = cache_root(tmp_path / "ImagingData", base)
    tests = cache_root(tmp_path / "ImagingData" / "tests", base)

    assert real != tests
    assert real.name.startswith("ImagingData-") and tests.name.startswith("tests-")
    assert real.parent == base / "cache"


def test_deleting_a_test_keeps_the_runs_inside_it(tmp_path):
    main, base = tmp_path / "share", tmp_path / ".odyn"
    folder = a_test(main)
    run = folder / "s128_o64_d6"
    download(main, folder, base)
    download(main, run, base)
    assert cache_size(base) == 510

    assert delete(main, folder, base) == 410
    assert state(main, folder, base) == NOT_HERE
    assert state(main, run, base) == HERE
    assert cache_size(base) == 100

    assert delete(main, run, base) == 100
    assert cache_size(base) == 0
    assert not cache_root(main, base).exists()  # empty folders go too
    assert delete(main, run, base) == 0


def test_sizes_read_well():
    assert human_size(0) == "0 B"
    assert human_size(999) == "999 B"
    assert human_size(1_234_567) == "1.2 MB"
    assert human_size(2_500_000_000) == "2.5 GB"


def test_a_download_copies_no_file_flags(tmp_path, monkeypatch):
    import errno
    import shutil

    from odyn.app import cache

    # macOS refuses a user the flags an SMB share sets on its files
    def refuse(*args, **kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(shutil, "copystat", refuse)
    main = tmp_path / "main"
    test = main / "cache" / "mcor-tests" / "e1_g1"
    test.mkdir(parents=True)
    (test / "metrics.json").write_text("{}")

    local = cache.download(main, test, tmp_path / "base")
    assert (local / "metrics.json").read_text() == "{}"
    assert cache.state(main, test, tmp_path / "base") == cache.HERE


def test_a_download_tells_its_bytes_and_can_be_stopped(tmp_path, monkeypatch):
    from odyn.app import cache

    monkeypatch.setattr(cache, "CHUNK_BYTES", 100)
    main, base = tmp_path / "share", tmp_path / ".odyn"
    folder = a_test(main)
    assert sum(p.stat().st_size for p in cache.pending(main, folder, base)) == 410

    # Every chunk heard, all the bytes
    copied = []
    cache.download(main, folder, base, progress=lambda name, n: copied.append(n))
    assert sum(copied) == 410
    assert cache.pending(main, folder, base) == []

    # Stopped mid-file: no partial file left, nothing taken for downloaded
    (folder / "summary.csv").write_bytes(b"z" * 350)

    def stop(name, n):
        if n:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        cache.download(main, folder, base, progress=stop)

    local = cache.local_folder(main, folder, base)
    assert not list(local.glob(".*.part"))
    assert cache.pending(main, folder, base) == [folder / "summary.csv"]
