-- MIGRATION v1 -> v2
--
-- CHANGES:
-- - Store every path with '/' separators, normalising the ones already stored.
-- - Add mcor_files.source, backfilling 'caiman'.
--
-- NOTES:
-- - Paths are stored relative to main_folder so the DB works from any machine
--   on the network, but they were written with str(Path), which emits the
--   separator of whichever OS ran the call. Rows written on Windows therefore
--   did not resolve on macOS/Linux. '/' works on Windows too, so normalising is
--   safe in both directions. The writers now use Path.as_posix().
-- - Add and rename mcor_files_new is needed because you cannot add NOT NULL
--   without a default value, and a default would silently mislabel a source
--   the caller forgot to pass -- exactly what the column exists to prevent.
-- - Everything already in the DB was motion corrected by run_motion_correction,
--   so 'caiman' is the correct backfill.

UPDATE acquisitions SET raw_path     = REPLACE(raw_path,     '\', '/');
UPDATE programs     SET program_path = REPLACE(program_path, '\', '/');
UPDATE outputs      SET file_path    = REPLACE(file_path,    '\', '/');

CREATE TABLE mcor_files_new
    ( acq_id            INTEGER PRIMARY KEY
    , mcor_path         TEXT NOT NULL
    , source            TEXT NOT NULL CHECK(source IN ('caiman', 'patchwarp'))
    , approved          INTEGER NOT NULL DEFAULT FALSE
    , last_updated_by   INTEGER NOT NULL

    , FOREIGN KEY (acq_id)          REFERENCES acquisitions(acq_id)
    , FOREIGN KEY (last_updated_by) REFERENCES method_calls(method_call_id)
    ) STRICT;

INSERT INTO mcor_files_new
    ( acq_id
    , mcor_path
    , source
    , approved
    , last_updated_by
    ) SELECT  acq_id
            , REPLACE(mcor_path, '\', '/')
            , 'caiman'
            , approved
            , last_updated_by
        FROM mcor_files;

DROP TABLE mcor_files;
ALTER TABLE mcor_files_new RENAME TO mcor_files;
