"""
The desktop app for entering session and experiment details.

It replaces the log spreadsheet: someone opens it at the start of a session,
fills things in as the experiment runs, and submits once at the end. Everything
before that submit lives in a `Draft` on the local disk, never in the database.

`draft` imports nothing else, so drafts can be read and written without a
database, and without Qt. The window needs PySide6, which is an optional
dependency -- install it with `pip install -e .[gui]`.
"""

from .draft import Draft, drafts_folder

__all__ = ["Draft", "drafts_folder"]
