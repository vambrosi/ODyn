CREATE TABLE IF NOT EXISTS experiments
    ( exp_id            INTEGER PRIMARY KEY
    , exp_name          TEXT NOT NULL
    , exp_type          TEXT NOT NULL CHECK(exp_type IN ('loop', 'grab', 'test'))
    , loop_start        TEXT UNIQUE CHECK(exp_type = 'test' OR datetime(loop_start) IS NOT NULL)
    , added_at          TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    ) STRICT;

CREATE TABLE IF NOT EXISTS raw_files
    ( acq_id                INTEGER PRIMARY KEY
    , raw_path              TEXT NOT NULL
    , height_px             INTEGER NOT NULL
    , width_px              INTEGER NOT NULL
    , height_um             REAL NOT NULL
    , width_um              REAL NOT NULL
    , frame_count           INTEGER NOT NULL
    , frame_rate            REAL NOT NULL
    , laser_power_920       INTEGER NOT NULL
    , laser_power_1040      INTEGER NOT NULL
    , loop_acq_interval_s   REAL NOT NULL
    , first_frame_start_s   REAL NOT NULL
    , added_at              TEXT DEFAULT (datetime('now', 'localtime'))
    ) STRICT;

CREATE TABLE IF NOT EXISTS mcor_files
    ( acq_id            INTEGER PRIMARY KEY
    , mcor_path         TEXT NOT NULL
    , approved          INTEGER NOT NULL DEFAULT FALSE
    , last_updated_by   INTEGER
    , updated_at        TEXT DEFAULT (datetime('now', 'localtime'))

    , FOREIGN KEY (acq_id)          REFERENCES raw_files(acq_id)
    , FOREIGN KEY (last_updated_by) REFERENCES method_calls(call_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS exp_files
    ( exp_id INTEGER REFERENCES experiments(exp_id)
    , acq_id INTEGER REFERENCES raw_files(acq_id)

    , PRIMARY KEY (exp_id, acq_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS method_calls
    ( call_id       INTEGER PRIMARY KEY
    , exp_id        INTEGER REFERENCES experiments(exp_id)
    , call_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    , method_name   TEXT NOT NULL
    , parameters    TEXT NOT NULL CHECK(json_valid(parameters))
    , git_commit    TEXT NOT NULL
    ) STRICT;
