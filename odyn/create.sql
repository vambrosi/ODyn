CREATE TABLE IF NOT EXISTS mice
    ( mouse_id      TEXT PRIMARY KEY
    , sex           TEXT NOT NULL CHECK(sex IN ('M', 'F'))
    , DOB           TEXT NOT NULL CHECK(date(DOB) IS NOT NULL)
    , "line"        TEXT NOT NULL
    , genotype      TEXT NOT NULL
    , injection     TEXT NOT NULL
    ) STRICT;

CREATE TABLE IF NOT EXISTS groups
    ( group_id      INTEGER PRIMARY KEY ) STRICT;

CREATE TABLE IF NOT EXISTS experiments
    ( exp_id                INTEGER PRIMARY KEY
    , exp_name              TEXT NOT NULL
    , exp_type              TEXT NOT NULL CHECK(exp_type IN ('loop', 'grab'))
    , exp_start             TEXT UNIQUE CHECK(datetime(exp_start) IS NOT NULL)
    , mouse_id              TEXT NOT NULL
    , height_px             INTEGER NOT NULL
    , width_px              INTEGER NOT NULL
    , height_um             REAL NOT NULL
    , width_um              REAL NOT NULL
    , frame_count           INTEGER NOT NULL
    , frame_rate            REAL NOT NULL
    , laser_power_920       INTEGER NOT NULL
    , laser_power_1040      INTEGER NOT NULL
    , loop_acq_interval_s   REAL NOT NULL
    , added_to_db_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    -- , FOREIGN KEY (mouse_id) REFERENCES mice(mouse_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS group_experiments
    ( group_id          INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , PRIMARY KEY (group_id, exp_id)
    , FOREIGN KEY (group_id) REFERENCES groups(group_id)
    , FOREIGN KEY (exp_id)   REFERENCES experiments(exp_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS raw_files
    ( acq_id                INTEGER PRIMARY KEY
    , exp_id                INTEGER NOT NULL
    , raw_path              TEXT NOT NULL
    , first_frame_start_s   REAL NOT NULL
    , added_to_db_at        TEXT DEFAULT (datetime('now', 'localtime'))

    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS mcor_files
    ( acq_id            INTEGER PRIMARY KEY
    , mcor_path         TEXT NOT NULL
    , approved          INTEGER NOT NULL DEFAULT FALSE
    , last_updated_by   INTEGER
    , updated_at        TEXT DEFAULT (datetime('now', 'localtime'))

    , FOREIGN KEY (acq_id)          REFERENCES raw_files(acq_id)
    -- , FOREIGN KEY (last_updated_by) REFERENCES method_calls(call_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS method_calls
    ( method_call_id    INTEGER PRIMARY KEY
    , group_id          INTEGER NOT NULL
    , called_at         TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    , method_name       TEXT NOT NULL
    , parameters        TEXT NOT NULL CHECK(json_valid(parameters))
    , git_commit        TEXT NOT NULL

    , FOREIGN KEY (group_id) REFERENCES groups(group_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS outputs
    ( output_id         INTEGER PRIMARY KEY
    , method_call_id    INTEGER NOT NULL
    , file_path         TEXT
    , log_text          TEXT NOT NULL
    , removed           INTEGER CHECK( removed IN (FALSE, TRUE) )

    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;