"""
The desktop app for entering session and experiment details.

It replaces the log spreadsheet: someone opens it at the start of a session,
fills things in as the experiment runs, and submits once at the end. Everything
before that submit lives in a `Draft` on the local disk, never in the database.

Four pieces, in order of how much they depend on:

- `draft` — the day's work on disk. Imports nothing at all.
- `fields` — what the form asks for, built from the annotation registry, and
  how a typed-in string is read as its type. No Qt.
- `submit` — draft to database: ingest the recordings, then annotate.
- `window` — the widgets. The only part that needs Qt, which is an optional
  dependency: `pip install -e .[gui]`, then `python -m odyn.app MAIN_FOLDER`.

Only the last one needs Qt, so everything that decides what the app *does* can
be tested without an event loop.
"""

from .draft import Draft, drafts_folder
from .fields import Field, form_fields

__all__ = ["Draft", "Field", "drafts_folder", "form_fields"]
