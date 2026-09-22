"""
The entry app: the one place users register, record, annotate, submit and check.

`python -m odyn.app <main_folder> [--project main-sync]` opens it. Qt is
optional for the package (`pip install -e .[gui]`); everything here that is
not a widget -- `fields`, `draft`, `submit`, `recorder`'s engine -- imports
no Qt and is tested without it.
"""
