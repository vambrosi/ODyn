-- CREATE DATABASE WITH SCHEMA v5
--
-- The tier organization described below explains the overall DB structure.
-- What tier the metadata belongs to should be considered before adding fields.
--
--  METADATA TIERS:
--  1) Needed to determine data identity/origin/ownership;
--  2) Code needs it to process data or it is a well-established data point;
--  3) Used only in analysis/filtering/labeling or it is project-specific.
--
--  The first two (*identity* and *contracts*) should be columns in the schema,
--  but the last ones are *annotations*. This keeps the DB general enough to be
--  used for all lab needs, while making DB SQLite checks useful/meaningful.
--
--  SOURCE PREFIXES:
--  - `acq_*` comes from ScanImage TIFFs;
--  - `trial_*` from the olfactometer logs;
--  - `sync_*` from the DAQ sync file.
--
--  IDS:
--  Tables that annotations can point at use AUTOINCREMENT, so a deleted row's
--  id is never reused and an annotation can never silently attach to a newer
--  row. `mice.mouse_id` is the SID, so it is the only exception.
--
--  PATHS:
--  Every stored path is relative to the main folder, with `/` separators.

-------------------------------------------------------------------------------
-- Users
-------------------------------------------------------------------------------

-- List of users that can make method calls. In shared-user computers this is
-- the only way of knowing who did what. `method_calls.user` stores the name,
-- not the id, so the provenance section stays independent of this table.
CREATE TABLE IF NOT EXISTS users
    ( user_id       INTEGER PRIMARY KEY
    , user_name     TEXT NOT NULL UNIQUE
    , display_name  TEXT NOT NULL UNIQUE
    , retired       INTEGER NOT NULL DEFAULT FALSE CHECK(retired IN (FALSE, TRUE))
    ) STRICT;

-------------------------------------------------------------------------------
-- Subjects
-------------------------------------------------------------------------------

-- Unchanging data about the mice.
-- `mouse_id` is the SID, the number after `m` in the folder names.
CREATE TABLE IF NOT EXISTS mice
    ( mouse_id          INTEGER PRIMARY KEY
    , mouse_sex         TEXT CHECK(mouse_sex IS NULL OR mouse_sex IN ('M', 'F'))
    , mouse_dob         TEXT CHECK(mouse_dob IS NULL OR date(mouse_dob) IS NOT NULL)
    , sensor            TEXT
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    ) STRICT;

-- One entry for each line/genotype of a mouse
CREATE TABLE IF NOT EXISTS mouse_lines
    ( mouse_id      INTEGER NOT NULL
    , line          TEXT NOT NULL
    , genotype      TEXT CHECK(genotype IS NULL OR genotype IN ('wt', 'het', 'hom'))
    , PRIMARY KEY (mouse_id, line)
    , FOREIGN KEY (mouse_id) REFERENCES mice(mouse_id) ON DELETE CASCADE
    ) STRICT;

-- Session exist only to add annotations about a mice and experiment setup
-- in a particular day. Panels, streams, and others belong to an experiment.
CREATE TABLE IF NOT EXISTS sessions
    ( session_id        INTEGER PRIMARY KEY AUTOINCREMENT
    , mouse_id          INTEGER NOT NULL
    , session_date      TEXT NOT NULL CHECK(date(session_date) IS NOT NULL)
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    , UNIQUE (mouse_id, session_date)
    , FOREIGN KEY (mouse_id) REFERENCES mice(mouse_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Recordings
-------------------------------------------------------------------------------

-- Groups are *processing units*, e.g. experiments that share a field of view
-- so they can be motion corrected together and have a single ROI mask. This is
-- kept simple so that the provenance and data sections are easily excisable.
CREATE TABLE IF NOT EXISTS groups
    ( group_id      INTEGER PRIMARY KEY AUTOINCREMENT ) STRICT;

-- Experiment is a ScanImage recording unit (has a unique loop start).
--
-- Identity: exp_id, session_id, exp_name, exp_type, exp_path, exp_start.
-- Contract: height_px, width_px, height_um, width_um, frame_count, frame_rate,
--           objective. Minimal set so odyn can run.
--
-- NOTES:
-- - `exp_path` is the folder, relative to the main folder. A loop has a folder
-- of its own, but grabs and z-stacks share one per session.
-- - `objective` is needed by anything scale-dependent (µm/px is stored by
-- ScanImage as if every acquisition were 20x), so it is required at insertion.
-- - `frame_count` is equal to ScanImage framesPerSlice (not actually counted)
CREATE TABLE IF NOT EXISTS experiments
    ( exp_id                INTEGER PRIMARY KEY AUTOINCREMENT
    , session_id            INTEGER NOT NULL
    , exp_name              TEXT NOT NULL
    , exp_type              TEXT NOT NULL CHECK(exp_type IN ('loop', 'grab', 'zstack'))
    , exp_path              TEXT NOT NULL
    , exp_start             TEXT NOT NULL UNIQUE CHECK(datetime(exp_start) IS NOT NULL)
    , height_px             INTEGER NOT NULL
    , width_px              INTEGER NOT NULL
    , height_um             REAL NOT NULL
    , width_um              REAL NOT NULL
    , frame_count           INTEGER NOT NULL
    , frame_rate            REAL NOT NULL
    , objective             INTEGER NOT NULL CHECK(objective > 0)
    , added_to_db_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , UNIQUE (session_id, exp_name)
    , FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS group_experiments
    ( group_id          INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , PRIMARY KEY (group_id, exp_id)
    , FOREIGN KEY (group_id) REFERENCES groups(group_id)     ON DELETE CASCADE
    , FOREIGN KEY (exp_id)   REFERENCES experiments(exp_id)  ON DELETE CASCADE
    ) STRICT;

-- Acquisition is a ScanImage TIFF output.
-- Alignment to other streams is in `sync_blocks`.
CREATE TABLE IF NOT EXISTS acquisitions
    ( acq_id            INTEGER PRIMARY KEY AUTOINCREMENT
    , exp_id            INTEGER NOT NULL
    , acq_start         TEXT NOT NULL CHECK(datetime(acq_start) IS NOT NULL)
    , raw_path          TEXT NOT NULL UNIQUE

    , UNIQUE (exp_id, acq_start)
    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE
    ) STRICT;

-- Other recordings streams, i.e. the DAQ sync H5, pupil videos.
-- The database only stores the path and a header.
--
-- NOTES:
-- - `stop_time` NULL means the recording was interrupted. The rig writer
-- streams samples and only writes stop_time, n_samples and filter_changes when
-- the it is stopped, so a crash leaves a readable file missing those.
-- - `attributes` keeps the file's own header as written.
CREATE TABLE IF NOT EXISTS streams
    ( stream_id         INTEGER PRIMARY KEY
    , exp_id            INTEGER NOT NULL
    , stream_type       TEXT NOT NULL CHECK(stream_type IN ('sync', 'pupil_video'))
    , file_path         TEXT NOT NULL UNIQUE
    , rate_hz           REAL
    , start_time        TEXT CHECK(start_time IS NULL OR datetime(start_time) IS NOT NULL)
    , stop_time         TEXT CHECK(stop_time  IS NULL OR datetime(stop_time)  IS NOT NULL)
    , n_samples         INTEGER
    , attributes        TEXT CHECK(attributes IS NULL OR json_valid(attributes))
    , method_call_id    INTEGER NOT NULL
    , FOREIGN KEY (exp_id)         REFERENCES experiments(exp_id) ON DELETE CASCADE
    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;

-- Sync blocks are busts of imaging frames from the sync stream.
-- We skip burst not associated with acquisitions.
--
-- The rows of one stream come from one decode call and are replaced as a
-- whole when it is rerun, so the table can be filled after the experiment was
-- inserted (the TIFFs and the sync file are copied at different times).
--
-- NOTES:
-- - `sync_frame_count` is to be compared against `experiments.frame_count`
-- to check for dropped frames.
-- - `sync_odor_on_frame` is the frame during which the valve opened, and
-- `sync_odor_on_residual_s` how long after that frame began (always >= 0).
-- `sync_odor_off_frame` is the first frame with the valve closed.
-- - `sync_odor_pulses` counts valve pulses inside the burst; anything but 1
-- on an imaging burst is a defect.
-- - `sync_camera_frames` counts camera strobes inside the burst. Zero means
-- no pupil data: the camera block is a fixed frame count, so a long
-- experiment can outrun it (which has happened).
-- - `clock_offset_ms` = `acq_start` (ScanImage) - burst start (olfactometer),
-- a measure of the skew between the two computers.
CREATE TABLE IF NOT EXISTS sync_blocks
    ( stream_id                 INTEGER NOT NULL
    , sync_block                INTEGER NOT NULL
    , acq_id                    INTEGER UNIQUE
    , sync_start_sample         INTEGER NOT NULL
    , sync_frame_count          INTEGER NOT NULL
    , sync_frame_period_s       REAL NOT NULL
    , sync_odor_pulses          INTEGER NOT NULL
    , sync_odor_on_frame        INTEGER
    , sync_odor_off_frame       INTEGER
    , sync_odor_on_residual_s   REAL
    , sync_odor_start           TEXT CHECK(sync_odor_start IS NULL OR datetime(sync_odor_start) IS NOT NULL)
    , sync_odor_end             TEXT CHECK(sync_odor_end   IS NULL OR datetime(sync_odor_end)   IS NOT NULL)
    , sync_camera_frames        INTEGER
    , clock_offset_ms           REAL
    , method_call_id            INTEGER NOT NULL

    , PRIMARY KEY (stream_id, sync_block)

    , FOREIGN KEY (stream_id)      REFERENCES streams(stream_id)      ON DELETE CASCADE
    , FOREIGN KEY (acq_id)         REFERENCES acquisitions(acq_id)    ON DELETE CASCADE
    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS mcor_files
    ( acq_id            INTEGER PRIMARY KEY
    , mcor_path         TEXT NOT NULL
    , source            TEXT NOT NULL CHECK(source IN ('caiman', 'patchwarp'))
    , approved          INTEGER NOT NULL DEFAULT FALSE
    , last_updated_by   INTEGER NOT NULL

    -- No cascade for the second because deleting calls does not delete data
    , FOREIGN KEY (acq_id)          REFERENCES acquisitions(acq_id) ON DELETE CASCADE
    , FOREIGN KEY (last_updated_by) REFERENCES method_calls(method_call_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Stimulus
-------------------------------------------------------------------------------

-- One olfactometer run inside an experiment.
--
-- `program_config` is the olfactometer program's definition as it was stored.
CREATE TABLE IF NOT EXISTS programs
    ( program_id        INTEGER PRIMARY KEY AUTOINCREMENT
    , exp_id            INTEGER NOT NULL
    , program_name      TEXT NOT NULL
    , program_type      TEXT NOT NULL
    , program_start     TEXT NOT NULL CHECK(datetime(program_start) IS NOT NULL)
    , program_path      TEXT NOT NULL
    , program_config    TEXT CHECK(program_config IS NULL OR json_valid(program_config))

    , UNIQUE (exp_id, program_id)
    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE
    ) STRICT;

-- Olfactometer trial. `acq_id` can be null because it is filled by the
-- matching step, which is run after insertion.
CREATE TABLE IF NOT EXISTS trials
    ( trial_id          INTEGER PRIMARY KEY
    , trial_start       TEXT NOT NULL CHECK(datetime(trial_start) IS NOT NULL)
    , trial_odor_start  TEXT NOT NULL CHECK(datetime(trial_odor_start) IS NOT NULL)
    , trial_odor_end    TEXT NOT NULL CHECK(datetime(trial_odor_end) IS NOT NULL)
    , odor_id           INTEGER NOT NULL
    , outcome           TEXT NOT NULL
    , acq_id            INTEGER UNIQUE
    , program_id        INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL

    , UNIQUE (trial_id, program_id)
    , UNIQUE (trial_start, program_id)

    , FOREIGN KEY (program_id, exp_id) REFERENCES programs(program_id, exp_id) ON DELETE CASCADE
    , FOREIGN KEY (acq_id)             REFERENCES acquisitions(acq_id)         ON DELETE SET NULL
    , FOREIGN KEY (odor_id)            REFERENCES odors(odor_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS events
    ( event_id      INTEGER PRIMARY KEY
    , event_time    TEXT NOT NULL CHECK(datetime(event_time) IS NOT NULL)
    , event_type    TEXT NOT NULL
    , event_tag     TEXT NOT NULL
    , program_id    INTEGER NOT NULL
    , trial_id      INTEGER
    , FOREIGN KEY (trial_id, program_id) REFERENCES trials(trial_id, program_id) ON DELETE CASCADE
    , FOREIGN KEY (program_id)           REFERENCES programs(program_id)         ON DELETE CASCADE
    ) STRICT;

-------------------------------------------------------------------------------
-- Odors
-------------------------------------------------------------------------------

-- The odors as in the olfactometer configs. Includes single chemicals and
-- mixes (`mix` = 1) since the olfactometer doesn't distinguish them. What a
-- mix is made of is in `odor_components` and never changes: a vial of it
-- only chooses the concentration of each component.
--
-- `vapor_pressure_mmhg` (at 25 C) is what the panel calculator needs to turn
-- a target concentration into the volume needed.
CREATE TABLE IF NOT EXISTS odors
    ( odor_id               INTEGER PRIMARY KEY
    , odor_name             TEXT NOT NULL UNIQUE
    , mix                   INTEGER NOT NULL DEFAULT FALSE CHECK(mix IN (FALSE, TRUE))
    , vapor_pressure_mmhg   REAL CHECK(vapor_pressure_mmhg IS NULL OR vapor_pressure_mmhg > 0)
    , vapor_pressure_source TEXT
    , retired               INTEGER NOT NULL DEFAULT FALSE CHECK(retired IN (FALSE, TRUE))
    ) STRICT;

-- The single chemicals a mix is made of (the mix's number, then each
-- component's). A single chemical has no rows here.
CREATE TABLE IF NOT EXISTS odor_components
    ( odor_id       INTEGER NOT NULL
    , component_id  INTEGER NOT NULL

    , PRIMARY KEY (odor_id, component_id)
    , FOREIGN KEY (odor_id)      REFERENCES odors(odor_id) ON DELETE CASCADE
    , FOREIGN KEY (component_id) REFERENCES odors(odor_id)
    ) STRICT;

-- Panels are named set of vials (as displayed in the olfactometer)
CREATE TABLE IF NOT EXISTS panels
    ( panel_id          INTEGER PRIMARY KEY
    , panel_name        TEXT NOT NULL UNIQUE
    , description       TEXT
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    ) STRICT;

-- Vials are literal physical vials together with their odor delivery
-- configurations. Remaking them with the same properties inserts a new row.
--
-- NOTES:
-- - `vial_sccm` is the flow drawn through the vial, `total_sccm` the flow
-- reaching the animal, so the dilution at the nose is their ratio. They are
-- the flows the recipe was aiming for; what the olfactometer actually used is
-- in the program (`OdorFlow`, `DilutionRate`), and a difference is a check.
-- - `total_volume_ml` is what the vial holds once made up, and
-- `solvent_volume_ml` the mineral oil in it: the total minus the odorant
-- volumes in `vial_odors`, computed when the recipe was made.
-- - `made_on` NULL means recorded as unknown.
CREATE TABLE IF NOT EXISTS vials
    ( vial_id           INTEGER PRIMARY KEY
    , panel_id          INTEGER NOT NULL
    , vial_position     INTEGER NOT NULL CHECK(vial_position > 0)
    , odor_id           INTEGER NOT NULL
    , vial_sccm         REAL NOT NULL CHECK(vial_sccm > 0)
    , total_sccm        REAL NOT NULL CHECK(total_sccm > 0)
    , total_volume_ml   REAL NOT NULL CHECK(total_volume_ml > 0)
    , solvent_volume_ml REAL CHECK(solvent_volume_ml IS NULL OR solvent_volume_ml >= 0)
    , made_on           TEXT CHECK(made_on IS NULL OR date(made_on) IS NOT NULL)
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , FOREIGN KEY (panel_id) REFERENCES panels(panel_id)
    , FOREIGN KEY (odor_id)  REFERENCES odors(odor_id)
    ) STRICT;

-- What was pipetted into a vial: one row per component, the mix's from
-- `odor_components`. A pure odor has one row (itself), mineral oil has none.
--
-- `target_ppm` is what concentration the recipe was aiming for; `liquid_ul`
-- and `percent_vv` are the volume pipetted and the resulting dilution, as
-- computed from it, the odor's vapor pressure and the vial's flows and volume
-- when the recipe was made. They are stored rather than recomputed because
-- the conversion follows published estimates that may be revised. The row
-- keeps what was actually put in the vial under the formula of the day.
CREATE TABLE IF NOT EXISTS vial_odors
    ( vial_id       INTEGER NOT NULL
    , odor_id       INTEGER NOT NULL
    , target_ppm    REAL NOT NULL CHECK(target_ppm > 0)
    , liquid_ul     REAL CHECK(liquid_ul IS NULL OR liquid_ul > 0)
    , percent_vv    REAL CHECK(percent_vv IS NULL OR percent_vv > 0)

    , PRIMARY KEY (vial_id, odor_id)
    , FOREIGN KEY (vial_id) REFERENCES vials(vial_id) ON DELETE CASCADE
    , FOREIGN KEY (odor_id) REFERENCES odors(odor_id)
    ) STRICT;

-- Which vials were in the olfactometer for an experiment. One experiment is
-- one olfactometer run, so it always uses one rack, but the link is per vial
-- so a single refilled vial is recorded as exactly that. The positions come
-- from `vials` and are checked against the olfactometer config at submit.
CREATE TABLE IF NOT EXISTS experiment_vials
    ( exp_id        INTEGER NOT NULL
    , vial_id       INTEGER NOT NULL

    , PRIMARY KEY (exp_id, vial_id)
    , FOREIGN KEY (exp_id)  REFERENCES experiments(exp_id) ON DELETE CASCADE
    , FOREIGN KEY (vial_id) REFERENCES vials(vial_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Annotations
-------------------------------------------------------------------------------

-- Record of possible annotations for the project. Referenced by name so forms
-- and queries can be easily readable.
--
-- DEFINITIONS:
-- - `required` means it must be recorded for the data to be "finished". Users
-- are prompted to fill it. It is project-wide, so it assumes a consistent use
-- of `Database(project=...)`.
-- - `retired` is True if it shouldn't be used in forms anymore.
-- - `source` has three possible values:
--      - `user` if it needs to be typed in;
--      - `metadata` is read from a file (TIFF header, olfactometer program);
--      - `computed` is derived from other fields.
--
-- Every key holds one value, the latest written. Annotations are append-only,
-- so correcting one is writing it again and the older rows stay as history.
CREATE TABLE IF NOT EXISTS annotation_keys
    ( applies_to        TEXT NOT NULL CHECK(applies_to IN
                            ( 'mouse'
                            , 'session'
                            , 'experiment'
                            , 'program'
                            , 'acquisition'
                            , 'group'
                            ))
    , key               TEXT NOT NULL
    , label             TEXT NOT NULL
    , value_type        TEXT NOT NULL CHECK(value_type IN
                            ( 'text'
                            , 'long_text' --stored as text but uses a larger display box
                            , 'integer'
                            , 'real'
                            , 'boolean'
                            , 'date'
                            , 'enum'
                            ))
    , allowed_values    TEXT CHECK(allowed_values IS NULL OR json_valid(allowed_values))
    , unit              TEXT
    , description       TEXT NOT NULL
    , source            TEXT NOT NULL DEFAULT 'user' CHECK(source IN ('user', 'metadata', 'computed'))
    , required          INTEGER NOT NULL DEFAULT FALSE CHECK(required IN (FALSE, TRUE))
    , retired           INTEGER NOT NULL DEFAULT FALSE CHECK(retired IN (FALSE, TRUE))

    , CHECK(value_type <> 'enum' OR allowed_values IS NOT NULL)
    , PRIMARY KEY (applies_to, key)
    ) STRICT;

-- Append-only annotations list.
--
-- Target is polymorphic so target existence cannot be enforced by SQLite. The
-- composite key does enforce that session keys refer sessions, for example.
-- ANY can be used in STRICT tables and comparisons still work. Author, commit,
-- and values as originally passed are stored in the originating method call.
--
-- A NULL `value` clears the key: the latest row wins, so writing NULL makes
-- the key read as unset while keeping what it used to say.
--
-- Type and value consistency needs to be enforced on write.
CREATE TABLE IF NOT EXISTS annotations
    ( annotation_id     INTEGER PRIMARY KEY
    , target_type       TEXT NOT NULL
    , target_id         INTEGER NOT NULL
    , key               TEXT NOT NULL
    , value             ANY
    , method_call_id    INTEGER NOT NULL
    , FOREIGN KEY (target_type, key)  REFERENCES annotation_keys(applies_to, key)
    , FOREIGN KEY (method_call_id)    REFERENCES method_calls(method_call_id)
    ) STRICT;

CREATE INDEX IF NOT EXISTS annotations_target
    ON annotations (target_type, target_id, key, annotation_id);

-------------------------------------------------------------------------------
-- Provenance
-------------------------------------------------------------------------------

-- Coupled to the data section by `group_id` alone (see `groups` table).
--
-- NOTES:
-- - `group_id` is which object recorded the call, not what the call is about.
-- Only `Database` and `Group` record calls, and group 0 is the `Database`'s own
-- row, seeded when the file is created (no third option currently, so NOT NULL).
-- - `user` = the name picked in the GUI, or the OS user outside it;
-- - `module` says which repo code came from (func.__module__);
-- - `code` holds commit hashes and if there are uncommitted edits for repos.
-- Example: {"odyn": {"commit": "a1b2c3d", "dirty": true}, "<project>": {...}}
-- - `environment` keeps named packages only as an approximation;
-- - `consumed_calls` is a JSON list of the method_call_ids `latest_output`
-- actually read (makes message passing reconstructable).
-- - `call_output` serve as message passing between functions.
-- - `ended_at` is NULL while the call runs, or if its process died.
CREATE TABLE IF NOT EXISTS method_calls
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

-- Registry of the on-disk artifacts a method call produced. Can be used to
-- pass binary data as messages (using paths).
CREATE TABLE IF NOT EXISTS outputs
    ( output_id         INTEGER PRIMARY KEY
    , method_call_id    INTEGER NOT NULL
    , file_path         TEXT
    , removed           INTEGER NOT NULL DEFAULT FALSE CHECK(removed IN (FALSE, TRUE))
    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Initial inserts
-------------------------------------------------------------------------------

-- Group 0 records the `Database`'s own calls.
INSERT OR IGNORE INTO groups (group_id) VALUES (0);

-- The odor registry, ids as in the olfactometer. Vapor pressures in mmHg at
-- 25 C, from the lab's odor workbook.
INSERT OR IGNORE INTO odors
    (odor_id,   odor_name,                          mix,    vapor_pressure_mmhg,    vapor_pressure_source                   ) VALUES
    ( 0,        'mineral oil',                      FALSE,  NULL,                   NULL                                    ),
    ( 1,        'eugenol',                          FALSE,  0.0226,                 'Wachowiak Lab & PubChem'               ),
    ( 2,        'methyl salicylate',                FALSE,  0.0343,                 'Wachowiak Lab & PubChem'               ),
    ( 3,        'acetophenone',                     FALSE,  0.397,                  'Wachowiak Lab & PubChem'               ),
    ( 4,        '1-butanol',                        FALSE,  7.78,                   'PubChem (7) & Dewan paper (7.78)'      ),
    ( 5,        '1-pentanol',                       FALSE,  2.65,                   'PubChem (2.2) & Dewan paper (2.65)'    ),
    ( 6,        '1-hexanol',                        FALSE,  0.928,                  'PubChem'                               ),
    ( 7,        '1-heptanol',                       FALSE,  0.2163,                 'PubChem'                               ),
    ( 8,        '1-octanol',                        FALSE,  0.0794,                 'thegoodscentscompany.com'              ),
    ( 9,        '(+) alpha pinene',                 FALSE,  4.75,                   'Wachowiak Lab & PubChem'               ),
    (10,        '(-) alpha pinene',                 FALSE,  3.489,                  'thegoodscentscompany.com'              ),
    (11,        '(-) beta pinene',                  FALSE,  2.93,                   'PubChem'                               ),
    (12,        '(-) limonene',                     FALSE,  1.44,                   'Wachowiak Lab'                         ),
    (13,        '(+) limonene',                     FALSE,  1.44,                   'Wachowiak Lab'                         ),
    (14,        'citral',                           FALSE,  NULL,                   NULL                                    ),
    (15,        'allyl sulfide',                    FALSE,  NULL,                   NULL                                    ),
    (17,        'alpha',                            TRUE,   NULL,                   NULL                                    ),
    (18,        'alpha''',                          TRUE,   NULL,                   NULL                                    ),
    (19,        'beta',                             TRUE,   NULL,                   NULL                                    ),
    (20,        'beta''',                           TRUE,   NULL,                   NULL                                    ),
    (21,        'ethyl butyrate',                   FALSE,  12.8,                   'Wachowiak Lab & PubChem'               ),
    (22,        'cyclopentanecarboxylic acid',      FALSE,  0.1,                    'thegoodscentscompany.com'              ),
    (23,        'cinnamaldehyde',                   FALSE,  0.0289,                 'Wachowiak Lab & PubChem'               ),
    (24,        'isovaleric acid',                  FALSE,  0.44,                   'Wachowiak Lab & PubChem'               ),
    (25,        'gamma',                            TRUE,   NULL,                   NULL                                    ),
    (26,        'gamma''',                          TRUE,   NULL,                   NULL                                    ),
    (27,        'delta',                            TRUE,   NULL,                   NULL                                    ),
    (28,        'delta''',                          TRUE,   NULL,                   NULL                                    ),
    (29,        'pyruvic acid',                     FALSE,  0.968,                  'thegoodscentscompany.com'              ),
    (30,        'trans-2-methyl-2-pentenoic acid',  FALSE,  0.24,                   'EPI Suite'                             ),
    (31,        'epsilon',                          TRUE,   NULL,                   NULL                                    ),
    (32,        'epsilon''',                        TRUE,   NULL,                   NULL                                    ),
    (33,        'zeta',                             TRUE,   NULL,                   NULL                                    ),
    (34,        'zeta''',                           TRUE,   NULL,                   NULL                                    ),
    (35,        '2-heptanone',                      FALSE,  3.85,                   'PubChem'                               ),
    (36,        'methyl tiglate',                   FALSE,  6.65,                   'Wachowiak Lab & EPI Suite'             ),
    (37,        'kappa',                            TRUE,   NULL,                   NULL                                    ),
    (38,        'kappa''',                          TRUE,   NULL,                   NULL                                    ),
    (39,        'lambda',                           TRUE,   NULL,                   NULL                                    ),
    (40,        'lambda''',                         TRUE,   NULL,                   NULL                                    ),
    (41,        'omicron',                          TRUE,   NULL,                   NULL                                    ),
    (42,        'omicron''',                        TRUE,   NULL,                   NULL                                    ),
    (43,        'rho',                              TRUE,   NULL,                   NULL                                    ),
    (44,        'rho''',                            TRUE,   NULL,                   NULL                                    );

-- What each mix is made of (`Mixes` sheet of the odor workbook)
INSERT OR IGNORE INTO odor_components (odor_id, component_id) VALUES
    (17,  4), (17, 10),
    (18,  4), (18, 10),
    (19,  8), (19, 12),
    (20,  8), (20, 12),
    (25, 21), (25, 22),
    (26, 21), (26, 22),
    (27, 23), (27, 24),
    (28, 23), (28, 24),
    (31, 22), (31, 30),
    (32, 22), (32, 30),
    (33, 11), (33,  5),
    (34, 11), (34,  5),
    (37, 10), (37, 35),
    (38, 10), (38, 35),
    (39,  3), (39, 12),
    (40,  3), (40, 12),
    (41, 21), (41,  4),
    (42, 21), (42,  4),
    (43, 36), (43,  5),
    (44, 36), (44,  5);

-- Annotations taken from the log files (other keys can be added later)..
INSERT OR IGNORE INTO annotation_keys
    ( applies_to
    , key, label
    , value_type, allowed_values, unit
    , description
    , source, required
    ) VALUES

-- mouse
    ( 'mouse'
    , 'stax_injection', 'Stax injection'
    , 'text', NULL, NULL
    , 'Stereotaxic injection'
    , 'user', FALSE
    ),

    ( 'mouse'
    , 'note', 'Note'
    , 'long_text', NULL, NULL
    , 'Notes about this mouse.'
    , 'user', FALSE
    ),

-- session
    ( 'session'
    , 'goal', 'Goal'
    , 'text', NULL, NULL
    , 'Goal of this whole session.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'mouse_weight_g', 'Mouse weight'
    , 'real', NULL, 'g'
    , 'Weight on the day of the experiment.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'injection_volume', 'S.q. injection vol'
    , 'real', NULL, 'ml'
    , 'Subcutaneous injection volume given during the session.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'injection_drug', 'S.q. injection drug'
    , 'text', NULL, NULL
    , 'What was injected.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'injection_time', 'S.q. injection time'
    , 'text', NULL, NULL
    , 'When the injection was given.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'headplate', 'Headplate'
    , 'text', NULL, NULL
    , 'Which headplate was used.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'pitch_angle', 'Pitch angle'
    , 'real', NULL, 'degrees'
    , 'Head pitch relative to horizontal.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'left_right_correction', 'Left-right-correction'
    , 'long_text', NULL, NULL
    , 'What was done to correct the head left-right tilt, if anything.'
    , 'user', FALSE
    ),

    ( 'session'
    , 'note', 'Note'
    , 'long_text', NULL, NULL
    , 'Notes about the session (including what went wrong).'
    , 'user', FALSE
    ),

    ( 'session'
    , 'flag', 'Flag'
    , 'boolean', NULL, NULL
    , 'Check this if something went wrong in the session.'
    , 'user', FALSE
    ),

-- experiment
    ( 'experiment'
    , 'goal', 'Goal'
    , 'text', NULL, NULL
    , 'Experiment goal.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'fov_description', 'FOV description'
    , 'text', NULL, NULL
    , 'Description of the field of view, e.g. "medial-rostral left bulb".'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'fov_depth_um', 'FOV depth'
    , 'real', NULL, 'um'
    , 'Depth below the surface.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'fov_depth_raw', 'FOV depth (as written)'
    , 'text', NULL, NULL
    , 'The depth as written down, signs and approximation marks kept. (For records only.)'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'depth_goal', 'Depth goal'
    , 'enum', '["superficial", "deep"]', NULL
    , 'Experiment depth goal. Not recomputed from fov_depth_um.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'pmt_gain', 'PMT gain'
    , 'real', NULL, NULL
    , 'PMT gain setting.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'laser_power_920', 'Laser power 920 nm'
    , 'integer', NULL, '%'
    , 'Rig setting, from the TIFF header.'
    , 'metadata', FALSE
    ),

    ( 'experiment'
    , 'laser_power_1040', 'Laser power 1040 nm'
    , 'integer', NULL, '%'
    , 'Rig setting, from the TIFF header.'
    , 'metadata', FALSE
    ),

    ( 'experiment'
    , 'loop_acq_interval_s', 'Acq interval'
    , 'real', NULL, 's'
    , 'Programmed interval between acquisitions in a loop, from the TIFF header.'
    , 'metadata', FALSE
    ),

    ( 'experiment'
    , 'treatment', 'Treatment'
    , 'text', NULL, NULL
    , 'What was administered, if anything.'
    , 'user', FALSE
    ),

    -- The trial block, as typed in before the recording. Can be checked
    -- against `programs.program_config` and the sync values.
    ( 'experiment'
    , 'block_baseline_s', 'Pre-odor baseline'
    , 'real', NULL, 's'
    , 'Planned time from trial start to odor onset.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_odor_s', 'Odor presentation'
    , 'real', NULL, 's'
    , 'Planned odor pulse length.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_post_odor_delay_s', 'Post-odor delay'
    , 'real', NULL, 's'
    , 'Planned time after the odor before the trial ends.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_iti_s', 'Inter-trial interval'
    , 'real', NULL, 's'
    , 'Planned time between trials.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_post_odor_acq_s', 'Post-odor acq time'
    , 'real', NULL, 's'
    , 'Imaging time after odor onset, as set in ScanImage.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_n_odors', '# odors'
    , 'integer', NULL, NULL
    , 'Planned number of odors in the program.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'block_n_trials', '# trials'
    , 'integer', NULL, NULL
    , 'Planned number of trials per program.'
    , 'user', FALSE
    ),

    ( 'experiment'
    , 'flag', 'Flag'
    , 'boolean', NULL, NULL
    , 'Check this if something went wrong in this experiment.'
    , 'user', FALSE
    ),

-- program
    ( 'program'
    , 'description', 'Description'
    , 'text', NULL, NULL
    , 'Block description, e.g. "baseline", "post-ketamine".'
    , 'user', FALSE
    ),

-- acquisition
    ( 'acquisition'
    , 'exclusion_reason', 'Exclusion reason'
    , 'text', NULL, NULL
    , 'Reason why this acquisition should be excluded.'
    , 'user', FALSE
    ),

    ( 'acquisition'
    , 'note', 'Note'
    , 'long_text', NULL, NULL
    , 'Anything worth knowing about this acquisition.'
    , 'user', FALSE
    ),

-- group
    ( 'group'
    , 'name', 'Name'
    , 'text', NULL, NULL
    , 'Display name for this group.'
    , 'user', FALSE
    ),

    ( 'group'
    , 'note', 'Note'
    , 'long_text', NULL, NULL
    , 'Anything worth knowing about this group.'
    , 'user', FALSE
    );
