"""
Marks `tests/` for pytest.

There are no fixtures here on purpose. The shared setup is plain functions in
`helpers.py`, because that is what they are -- nothing needs per-test setup or
teardown, and a fixture would add a parameter to every test signature that the
test does not conceptually need.

This file still earns its place: with pytest's default import mode the folder
holding `conftest.py` is put on `sys.path`, which is what makes
`from helpers import ...` work from the test modules.
"""
