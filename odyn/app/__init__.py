"""
The entry app: the one place users register, record, annotate, submit and check.

`python -m odyn.app <main_folder> [--project main-sync]` opens it. Qt is
optional for the package (`pip install -e .[gui]`). What is not a widget
lives in `odyn.drafts` (drafts, checks, submissions, forms) or imports no Qt
here (`recorder`'s engine, `cache`), and is tested without it.
"""
