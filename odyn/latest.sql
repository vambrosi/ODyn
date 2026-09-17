-- MIGRATION v3 -> v4
--
-- CHANGES:
-- - Add method_calls.user, module, code, environment and consumed_calls.
-- - Drop method_calls.git_commit; its commit moves into `code`.
--
-- NOTES:
-- - Usual create, copy, and drop to add NOT NULL columns without a default.
-- - Calls recorded before this migration get:
--      (1) user 'unknown';
--      (2) module from the class in method_name ('unknown' for custom functions)
--      (3) {"odyn": {"commit": <git_commit>, "dirty": null}}
--      (4) environment and consumed_calls stay NULL.

CREATE TABLE method_calls_v4
    ( method_call_id    INTEGER PRIMARY KEY
    , group_id          INTEGER NOT NULL
    , user              TEXT NOT NULL
    , called_at         TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    , method_name       TEXT NOT NULL
    , module            TEXT NOT NULL
    , code              TEXT NOT NULL CHECK(json_valid(code))
    , environment       TEXT CHECK(environment IS NULL OR json_valid(environment))
    , parameter_inputs  TEXT NOT NULL CHECK(json_valid(parameter_inputs))
    , parameters_used   TEXT NOT NULL CHECK(json_valid(parameters_used))
    , consumed_calls    TEXT CHECK(consumed_calls IS NULL OR json_valid(consumed_calls))
    , call_log          TEXT NOT NULL DEFAULT ''
    , call_flag         INTEGER NOT NULL DEFAULT 0
    , call_output       TEXT CHECK(call_output IS NULL OR json_valid(call_output))
    , ended_at          TEXT CHECK(ended_at IS NULL OR datetime(ended_at) IS NOT NULL)

    , FOREIGN KEY (group_id) REFERENCES groups(group_id)
    ) STRICT;

INSERT INTO method_calls_v4
    ( method_call_id
    , group_id
    , user
    , called_at
    , method_name
    , module
    , code
    , environment
    , parameter_inputs
    , parameters_used
    , consumed_calls
    , call_log
    , call_flag
    , call_output
    , ended_at
    )
    SELECT method_call_id
         , group_id
         , 'unknown'
         , called_at
         , method_name
         , CASE
               WHEN method_name LIKE 'Database.%' THEN 'odyn.database'
               WHEN method_name LIKE 'Group.%' THEN 'odyn.groups'
               WHEN method_name LIKE 'Admin.%' THEN 'odyn.admin'
               ELSE 'unknown'
           END
         , json_object(
               'odyn',
               json_object(
                   'commit', NULLIF(git_commit, 'unknown-hash'),
                   'dirty', NULL
               )
           )
         , NULL
         , parameter_inputs
         , parameters_used
         , NULL
         , call_log
         , call_flag
         , call_output
         , ended_at
        FROM method_calls;

DROP TABLE method_calls;

ALTER TABLE method_calls_v4 RENAME TO method_calls;
