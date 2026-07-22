-- MIGRATION v0 -> v1
--
-- CHANGES:
-- - Add parameters_used, backfilling '{}'.
--
-- NOTES:
-- - Add and rename method_calls_new is needed because
--   you cannot add NOT NULL without a default value.

CREATE TABLE method_calls_new
    ( method_call_id    INTEGER PRIMARY KEY
    , group_id          INTEGER NOT NULL
    , called_at         TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    , method_name       TEXT NOT NULL
    , parameter_inputs  TEXT NOT NULL CHECK(json_valid(parameter_inputs))
    , git_commit        TEXT NOT NULL
    , call_log          TEXT NOT NULL DEFAULT ''
    , call_flag         INTEGER NOT NULL DEFAULT 0
    , call_output       TEXT CHECK(call_output IS NULL OR json_valid(call_output))
    , parameters_used   TEXT NOT NULL CHECK(json_valid(parameters_used))

    , FOREIGN KEY (group_id) REFERENCES groups(group_id)
    ) STRICT;

INSERT INTO method_calls_new
    ( method_call_id
    , group_id
    , called_at
    , method_name
    , parameter_inputs
    , git_commit
    , call_log
    , call_flag
    , call_output
    , parameters_used
    ) SELECT  method_call_id
            , group_id
            , called_at
            , method_name
            , parameters
            , git_commit
            , call_log
            , call_flag
            , call_output
            , '{}'
        FROM method_calls;

DROP TABLE method_calls;
ALTER TABLE method_calls_new RENAME TO method_calls;
