-- CREATE DATABASE WITH SCHEMA v1
CREATE TABLE IF NOT EXISTS mice
    ( mouse_id          TEXT PRIMARY KEY
    , mouse_sex         TEXT NOT NULL CHECK(mouse_sex IN ('M', 'F'))
    , mouse_dob         TEXT NOT NULL CHECK(date(mouse_dob) IS NOT NULL)
    , mouse_line        TEXT NOT NULL
    , mouse_genotype    TEXT NOT NULL
    , injection         TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS acquisitions
    ( acq_id            INTEGER PRIMARY KEY
    , exp_id            INTEGER NOT NULL
    , acq_start         TEXT CHECK(datetime(acq_start) IS NOT NULL)
    , odor_start        TEXT CHECK(odor_start IS NULL OR datetime(odor_start) IS NOT NULL)
    , odor_end          TEXT CHECK(odor_end IS NULL OR datetime(odor_end) IS NOT NULL)
    , h5_to_acq_ms      REAL CHECK(odor_start IS NULL OR h5_to_acq_ms NOT NULL)
    , raw_path          TEXT NOT NULL

    , UNIQUE (exp_id, acq_id)
    , UNIQUE (exp_id, acq_start)

    , FOREIGN KEY (exp_id)  REFERENCES experiments(exp_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS mcor_files
    ( acq_id            INTEGER PRIMARY KEY
    , mcor_path         TEXT NOT NULL
    , approved          INTEGER NOT NULL DEFAULT FALSE
    , last_updated_by   INTEGER NOT NULL

    , FOREIGN KEY (acq_id)          REFERENCES acquisitions(acq_id)
    , FOREIGN KEY (last_updated_by) REFERENCES method_calls(method_call_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS programs
    ( program_id        INTEGER PRIMARY KEY
    , exp_id            INTEGER NOT NULL
    , program_name      TEXT NOT NULL
    , program_type      TEXT NOT NULL
    , program_start     TEXT CHECK(datetime(program_start) IS NOT NULL)
    , program_path      TEXT NOT NULL

    , UNIQUE (exp_id, program_id)

    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS trials
    ( trial_id          INTEGER PRIMARY KEY
    , trial_start       TEXT CHECK(datetime(trial_start) IS NOT NULL)
    , odor_start        TEXT CHECK(datetime(odor_start) IS NOT NULL)
    , odor_end          TEXT CHECK(datetime(odor_end) IS NOT NULL)
    , odor_id           INTEGER NOT NULL
    , outcome           TEXT NOT NULL
    , acq_id            INTEGER
    , h5_to_trial_ms    REAL CHECK(acq_id IS NULL OR h5_to_trial_ms NOT NULL)
    , program_id        INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL

    , UNIQUE (trial_id, program_id)
    , UNIQUE (trial_start, program_id)

    , FOREIGN KEY (program_id, exp_id)  REFERENCES programs(program_id, exp_id)
    , FOREIGN KEY (acq_id, exp_id)      REFERENCES acquisitions(acq_id, exp_id)
    , FOREIGN KEY (odor_id)             REFERENCES odors(odor_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS events
    ( event_id      INTEGER PRIMARY KEY
    , event_time    TEXT CHECK(datetime(event_time) IS NOT NULL)
    , event_type    TEXT NOT NULL
    , event_tag     TEXT NOT NULL
    , program_id    INTEGER NOT NULL
    , trial_id      INTEGER

    , FOREIGN KEY (trial_id, program_id)    REFERENCES trials(trial_id, program_id)
    , FOREIGN KEY (program_id)              REFERENCES programs(program_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS odors
    ( odor_id   INTEGER PRIMARY KEY
    , odor_name TEXT NOT NULL
    ) STRICT;

CREATE TABLE IF NOT EXISTS method_calls
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

CREATE TABLE IF NOT EXISTS outputs
    ( output_id         INTEGER PRIMARY KEY
    , method_call_id    INTEGER NOT NULL
    , file_path         TEXT
    , removed           INTEGER CHECK(removed IN (FALSE, TRUE))

    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;

INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (0 , 'mineral oil');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (1 , 'eugenol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (2 , 'methyl salicylate');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (3 , 'acetophenone');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (4 , '1-butanol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (5 , '1-pentanol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (6 , '1-hexanol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (7 , '1-heptanol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (8 , '1-octanol');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (9 , '(+) alpha pinene');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (10, '(-) alpha pinene');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (11, '(-) beta pinene');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (12, '(-) limonene');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (13, '(+) limonene');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (14, 'citral');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (15, 'allyl sulfide');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (17, 'alpha');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (18, 'alpha''');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (19, 'beta');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (20, 'beta''');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (21, 'ethyl butyrate');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (22, 'cyclopentanecarboxylic acid');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (23, 'cinnamaldehyde');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (24, 'isovaleric acid');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (25, 'gamma');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (26, 'gamma''');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (27, 'delta');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (28, 'delta''');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (29, 'pyruvic acid');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (30, 'trans-2-methyl-2-pentenoic acid');