-- CREATE DATABASE WITH SCHEMA v3
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

-------------------------------------------------------------------------------
-- Subjects
-------------------------------------------------------------------------------

-- Unchanging data about the mice
-- `mouse_id` is the number after `m` or `sid` in the folders.
CREATE TABLE IF NOT EXISTS mice
    ( mouse_id          INTEGER PRIMARY KEY
    , mouse_sex         TEXT CHECK(mouse_sex IS NULL OR mouse_sex IN ('M', 'F'))
    , mouse_dob         TEXT CHECK(mouse_dob IS NULL OR date(mouse_dob) IS NOT NULL)
    , stax_injection    TEXT
    , sensor            TEXT
    ) STRICT;

-- One entry for each line/genotype of a mouse
CREATE TABLE IF NOT EXISTS mouse_lines
    ( mouse_id      INTEGER NOT NULL
    , line          TEXT NOT NULL
    , genotype      TEXT CHECK(genotype IS NULL OR genotype IN ('wt', 'het', 'hom'))

    , PRIMARY KEY (mouse_id, line)
    , FOREIGN KEY (mouse_id) REFERENCES mice(mouse_id) ON DELETE CASCADE
    ) STRICT;

-- The mouse/day level of hierarchy, e.g. `20260708/m442`. Annotations pointing
-- to it hold settings that might vary day-to-day but apply to many experiments
-- or records at once (mouse weight, headplate, the notes block).

CREATE TABLE IF NOT EXISTS sessions
    ( session_id        INTEGER PRIMARY KEY
    , mouse_id          INTEGER NOT NULL
    , session_date      TEXT NOT NULL CHECK(date(session_date) IS NOT NULL)
    , session_path      TEXT NOT NULL
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , UNIQUE (mouse_id, session_date)

    -- Not a foreign key because experiments can be added before the mice are
    -- registered in the DB (but mouse_id is known from TIFF and folder naming).
    -- , FOREIGN KEY (mouse_id) REFERENCES mice(mouse_id)

    ) STRICT;

-------------------------------------------------------------------------------
-- Recordings
-------------------------------------------------------------------------------

-- Groups are *processing units*, e.g. experiments that share a field of view
-- so they can be motion corrected together and have a single ROI mask. This is
-- kept simple so that the provenance and data sections are easily excisable.

CREATE TABLE IF NOT EXISTS groups
    ( group_id      INTEGER PRIMARY KEY ) STRICT;

-- Identity: exp_id, session_id, exp_name, exp_type, exp_start.
-- Contract:  height_px, width_px, height_um, width_um, frame_count, frame_rate,
--            objective. Minimal set so odyn can run.
--
-- NOTES:
-- - `objective` is nullable because it might be manually entered after
-- experiment ingestion. Used by TEN_X_CORRECTION, thus not an annotation.
-- - `frame_count` is equal to ScanImage framesPerSlice (not actually counted)

CREATE TABLE IF NOT EXISTS experiments
    ( exp_id                INTEGER PRIMARY KEY
    , session_id            INTEGER NOT NULL
    , exp_name              TEXT NOT NULL
    , exp_type              TEXT NOT NULL CHECK(exp_type IN ('loop', 'grab'))
    , exp_start             TEXT UNIQUE CHECK(datetime(exp_start) IS NOT NULL)
    , height_px             INTEGER NOT NULL
    , width_px              INTEGER NOT NULL
    , height_um             REAL NOT NULL
    , width_um              REAL NOT NULL
    , frame_count           INTEGER NOT NULL
    , frame_rate            REAL NOT NULL
    , objective             INTEGER CHECK(objective IS NULL OR objective > 0)
    , added_to_db_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    ) STRICT;

CREATE TABLE IF NOT EXISTS group_experiments
    ( group_id          INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))

    , PRIMARY KEY (group_id, exp_id)
    , FOREIGN KEY (group_id) REFERENCES groups(group_id)     ON DELETE CASCADE
    , FOREIGN KEY (exp_id)   REFERENCES experiments(exp_id)  ON DELETE CASCADE
    ) STRICT;

-- Pure ScanImage metadata. Syncing data is in `acquisition_sync`.
CREATE TABLE IF NOT EXISTS acquisitions
    ( acq_id            INTEGER PRIMARY KEY
    , exp_id            INTEGER NOT NULL
    , acq_start         TEXT CHECK(datetime(acq_start) IS NOT NULL)
    , raw_path          TEXT NOT NULL

    , UNIQUE (exp_id, acq_id)
    , UNIQUE (exp_id, acq_start)

    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE
    ) STRICT;

-- How to align acquisitions with other recorded streams. A separate table
-- because it can more easily be updated after ingestion, if the experiment
-- was added before the sync file was copied. Acquisitions belonging to an
-- experiment form a provenance unit, it must be replaced as a whole if
-- re-decoded.
--
-- GRAIN RULE:
-- - Acquisition scalars in this table;
-- - Frame arrays stay in the stream files (`streams` holds only their paths);
-- - Pupil alignment separately (~100k rows per session).
--
-- NOTES:
-- - `sync_frame_count` is to be compared against `experiments.frame_count`,
-- to check if there were any dropped frames.
-- - `sync_odor_residual_s` measures seconds from the start of the assigned
-- frame to the odor onset, so it is positive when the valve opened after that
-- frame began.
-- - `sync_camera_frames` counts how many camera frames per acquisition. Zero
-- means no pupil data: the camera block is a fixed frame count, so a long
-- session can outrun it (which has happened).
-- - `clock_offset_ms` = `acq_start` (ScanImage) - burst start (olfactometer)

CREATE TABLE IF NOT EXISTS acquisition_sync
    ( acq_id                INTEGER PRIMARY KEY
    , sync_block            INTEGER NOT NULL
    , sync_frame_count      INTEGER NOT NULL
    , sync_odor_on_frame    INTEGER
    , sync_odor_off_frame   INTEGER
    , sync_odor_residual_s  REAL
    , sync_odor_start       TEXT CHECK(sync_odor_start IS NULL OR datetime(sync_odor_start) IS NOT NULL)
    , sync_odor_end         TEXT CHECK(sync_odor_end   IS NULL OR datetime(sync_odor_end)   IS NOT NULL)
    , sync_camera_frames    INTEGER
    , clock_offset_ms       REAL
    , method_call_id        INTEGER NOT NULL

    , FOREIGN KEY (acq_id)         REFERENCES acquisitions(acq_id) ON DELETE CASCADE
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

-- Whole-file recordings that are not per-trial: the DAQ sync H5, pupil videos.
-- The database stores the path and an inventory; the arrays are parsed on
-- demand. Session-grain, unlike `acquisition_sync`.
CREATE TABLE IF NOT EXISTS streams
    ( stream_id         INTEGER PRIMARY KEY
    , session_id        INTEGER NOT NULL
    , stream_type       TEXT NOT NULL CHECK(stream_type IN ('sync', 'pupil_video'))
    , file_path         TEXT NOT NULL
    , rate_hz           REAL
    , start_time        TEXT CHECK(start_time IS NULL OR datetime(start_time) IS NOT NULL)

    -- NULL means the recording was interrupted. The rig writer streams samples
    -- and only writes stop_time, n_samples and filter_changes when the operator
    -- presses Stop, so a crash leaves a readable file missing exactly those.
    , stop_time         TEXT CHECK(stop_time IS NULL OR datetime(stop_time) IS NOT NULL)
    , channels          TEXT CHECK(channels IS NULL OR json_valid(channels))
    , method_call_id    INTEGER NOT NULL

    , UNIQUE (session_id, file_path)
    , FOREIGN KEY (session_id)          REFERENCES sessions(session_id) ON DELETE CASCADE
    , FOREIGN KEY (method_call_id)      REFERENCES method_calls(method_call_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Stimulus
-------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS programs
    ( program_id        INTEGER PRIMARY KEY
    , exp_id            INTEGER NOT NULL
    , program_name      TEXT NOT NULL
    , program_type      TEXT NOT NULL
    , program_start     TEXT CHECK(datetime(program_start) IS NOT NULL)
    , program_path      TEXT NOT NULL

    , UNIQUE (exp_id, program_id)

    , FOREIGN KEY (exp_id) REFERENCES experiments(exp_id) ON DELETE CASCADE
    ) STRICT;

CREATE TABLE IF NOT EXISTS trials
    ( trial_id          INTEGER PRIMARY KEY
    , trial_start       TEXT CHECK(datetime(trial_start) IS NOT NULL)
    , trial_odor_start  TEXT CHECK(datetime(trial_odor_start) IS NOT NULL)
    , trial_odor_end    TEXT CHECK(datetime(trial_odor_end) IS NOT NULL)
    , odor_id           INTEGER NOT NULL
    , outcome           TEXT NOT NULL
    , acq_id            INTEGER
    , sync_to_trial_ms  REAL CHECK(acq_id IS NULL OR sync_to_trial_ms NOT NULL)
    , program_id        INTEGER NOT NULL
    , exp_id            INTEGER NOT NULL

    , UNIQUE (trial_id, program_id)
    , UNIQUE (trial_start, program_id)

    , FOREIGN KEY (program_id, exp_id) REFERENCES programs(program_id, exp_id) ON DELETE CASCADE
    , FOREIGN KEY (acq_id, exp_id)     REFERENCES acquisitions(acq_id, exp_id) ON DELETE CASCADE
    , FOREIGN KEY (odor_id)            REFERENCES odors(odor_id)
    ) STRICT;

CREATE TABLE IF NOT EXISTS events
    ( event_id      INTEGER PRIMARY KEY
    , event_time    TEXT CHECK(datetime(event_time) IS NOT NULL)
    , event_type    TEXT NOT NULL
    , event_tag     TEXT NOT NULL
    , program_id    INTEGER NOT NULL
    , trial_id      INTEGER

    , FOREIGN KEY (trial_id, program_id) REFERENCES trials(trial_id, program_id) ON DELETE CASCADE
    , FOREIGN KEY (program_id)           REFERENCES programs(program_id)         ON DELETE CASCADE
    ) STRICT;

-- Records odor identity
CREATE TABLE IF NOT EXISTS odors
    ( odor_id   INTEGER PRIMARY KEY
    , odor_name TEXT NOT NULL
    ) STRICT;

-- Panels of odors to be used in a session
CREATE TABLE IF NOT EXISTS panels
    ( panel_id          INTEGER PRIMARY KEY
    , panel_name        TEXT NOT NULL UNIQUE
    , description       TEXT
    , added_to_db_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    ) STRICT;

-- One row per vial position in a panel: what the olfactometer delivers when it
-- opens that vial, and how the liquid in it was made up.
--
-- `odor_id` is the odor as delivered, so a two-component mix points at the
-- mix's own row in `odors` ('alpha', 'lambda') rather than at either component.
-- What went into the vial is in `vial_components`; a pure odor has one
-- component row, mineral oil has none.
--
-- NOTES:
-- - `odor_sccm` is the flow drawn through the vial, `total_sccm` the flow
-- reaching the animal, so the dilution at the nose is their ratio.
-- - `solvent_volume_ml` is the mineral oil the odorant was dissolved in, and
-- `total_volume_ml` what the vial holds once it is.
CREATE TABLE IF NOT EXISTS panel_vials
    ( panel_id          INTEGER NOT NULL
    , vial_position     INTEGER NOT NULL CHECK(vial_position > 0)
    , odor_id           INTEGER NOT NULL
    , odor_sccm         REAL
    , total_sccm        REAL
    , total_volume_ml   REAL
    , solvent_volume_ml REAL

    , PRIMARY KEY (panel_id, vial_position)
    , FOREIGN KEY (panel_id) REFERENCES panels(panel_id) ON DELETE CASCADE
    , FOREIGN KEY (odor_id)  REFERENCES odors(odor_id)
    ) STRICT;

-- What was pipetted into a vial. Separate from `panel_vials` so a mix is not
-- capped at the two components the current panels happen to use.
--
-- `target_ppm` is the headspace concentration the recipe was solved for and
-- `liquid_ul` the volume that produced it, so the pair records both what was
-- wanted and what was actually measured out.
CREATE TABLE IF NOT EXISTS vial_components
    ( panel_id          INTEGER NOT NULL
    , vial_position     INTEGER NOT NULL
    , odor_id           INTEGER NOT NULL
    , target_ppm        REAL
    , liquid_ul         REAL
    , percent_vv        REAL

    , PRIMARY KEY (panel_id, vial_position, odor_id)
    , FOREIGN KEY (panel_id, vial_position)
        REFERENCES panel_vials(panel_id, vial_position) ON DELETE CASCADE
    , FOREIGN KEY (odor_id) REFERENCES odors(odor_id)
    ) STRICT;

-- Which panel a session ran. The recipe is in `panels` because many sessions
-- share it; what varies per session is the mixing, in `session_vials`.
CREATE TABLE IF NOT EXISTS session_panels
    ( session_id    INTEGER PRIMARY KEY
    , panel_id      INTEGER NOT NULL

    -- So `session_vials` can key on the pair and inherit the panel.
    , UNIQUE (session_id, panel_id)

    , FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    , FOREIGN KEY (panel_id)   REFERENCES panels(panel_id)
    ) STRICT;

-- When each vial a session ran was last mixed. Usually a whole rack is remade
-- at once and every vial carries the same date, but they have also been refilled
-- one at a time, so the date belongs to the vial rather than to the session.
--
-- A vial with no row was not recorded; a row with a NULL `made_on` was recorded
-- as unknown.
CREATE TABLE IF NOT EXISTS session_vials
    ( session_id    INTEGER NOT NULL
    , panel_id      INTEGER NOT NULL
    , vial_position INTEGER NOT NULL
    , made_on       TEXT CHECK(made_on IS NULL OR date(made_on) IS NOT NULL)

    , PRIMARY KEY (session_id, vial_position)

    , FOREIGN KEY (session_id, panel_id)
        REFERENCES session_panels(session_id, panel_id) ON DELETE CASCADE

    -- A session can only date a vial its own panel defines.
    , FOREIGN KEY (panel_id, vial_position)
        REFERENCES panel_vials(panel_id, vial_position)
    ) STRICT;

-------------------------------------------------------------------------------
-- Annotations
-------------------------------------------------------------------------------

-- Record of possible annotations for the project. Referenced by name so forms
-- and queries can be easily readable.
--
-- DEFINITIONS:
-- - `required` means it must be recorded for the data to be "finished". Users
-- are prompted to fill it regularly. It is project-wide, so it assumes a
-- consistent use of `Database(project=...)`
--
-- - `multi_valued` is False if only the latest value should be used, and True
-- if it is to be considered as a list of values. Former case permit later
-- corrections, and the latter allows for list of related notes.
--
-- - `retired` is True if it shouldn't be used in forms anymore.

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
                            , 'integer'
                            , 'real'
                            , 'boolean'
                            , 'date'
                            , 'enum'
                            ))

    , allowed_values    TEXT CHECK(allowed_values IS NULL OR json_valid(allowed_values))
    , unit              TEXT
    , description       TEXT NOT NULL
    , required          INTEGER NOT NULL DEFAULT FALSE
    , multi_valued      INTEGER NOT NULL DEFAULT FALSE
    , retired           INTEGER NOT NULL DEFAULT FALSE

    -- An enum with no options renders as an empty dropdown, which looks like a
    -- missing value rather than a broken key.
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
-- Type and value consistency needs to be enforced on write.

CREATE TABLE IF NOT EXISTS annotations
    ( annotation_id     INTEGER PRIMARY KEY
    , target_type       TEXT NOT NULL
    , target_id         INTEGER NOT NULL
    , key               TEXT NOT NULL
    , value             ANY NOT NULL
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
-- - `user` = OS user, for now (wrong for always-logged-in computers);
-- - `module` says which repo code came from (func.__module__);
-- - `code` holds commit hashes and if there are uncommitted edits for repos.
-- Example: {"odyn": {"commit": "a1b2c3d", "dirty": true}, "<project>": {...}}
-- - `environment` keeps named packages only as an approximation;
-- - `consumed_calls` is a JSON list of the method_call_ids `latest_output`
-- actually read (makes message passing reconstructable).
-- - `call_output` serve as message passing between functions.

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

    , FOREIGN KEY (group_id) REFERENCES groups(group_id)
    ) STRICT;

-- Registry of the on-disk artifacts a method call produced. Can be used to
-- pass binary data as messages (using paths).

CREATE TABLE IF NOT EXISTS outputs
    ( output_id         INTEGER PRIMARY KEY
    , method_call_id    INTEGER NOT NULL
    , file_path         TEXT
    , removed           INTEGER CHECK(removed IN (FALSE, TRUE))

    , FOREIGN KEY (method_call_id) REFERENCES method_calls(method_call_id)
    ) STRICT;

-------------------------------------------------------------------------------
-- Initial inserts
-------------------------------------------------------------------------------

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
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (31, 'epsilon');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (32, 'epsilon''');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (39, 'lambda');
INSERT OR IGNORE INTO odors (odor_id, odor_name) VALUES (40, 'lambda''');

-- Annotations taken from log files (but other keys can be added later)

INSERT OR IGNORE INTO annotation_keys
    ( applies_to
    , key, label
    , value_type, allowed_values, unit
    , description
    , required, multi_valued
    ) VALUES

-- session
    ( 'session'
    , 'goal', 'Goal'
    , 'text', NULL, NULL
    , 'Goal of this whole session.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'mouse_weight_g', 'Mouse weight'
    , 'real', NULL, 'g'
    , 'Weight on the day of the experiment.'
    , TRUE, FALSE
    ),

    ( 'session'
    , 'injection_volume', 'S.q. injection volume'
    , 'real', NULL, 'ml'
    , 'Subcutaneous injection volume given during the session.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'injection_drug', 'S.q. injection drug'
    , 'text', NULL, NULL
    , 'What was injected, as written down.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'injection_time', 'S.q. injection time'
    , 'text', NULL, NULL
    , 'When the injection was given, as written down.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'headplate', 'Headplate'
    , 'text', NULL, NULL
    , 'Which headplate the mouse was mounted on.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'pitch_angle', 'Pitch angle'
    , 'real', NULL, 'degrees'
    , 'Head pitch relative to horizontal.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'left_right_correction', 'Left-right correction'
    , 'boolean', NULL, NULL
    , 'Whether a left-right tilt correction was applied.'
    , FALSE, FALSE
    ),

    ( 'session'
    , 'note', 'Note'
    , 'text', NULL, NULL
    , 'Comments about this session. One entry per note.'
    , FALSE, TRUE
    ),

    ( 'session'
    , 'flag', 'Flag'
    , 'text', NULL, NULL
    , 'Something that went wrong. One entry per problem.'
    , FALSE, TRUE
    ),

-- experiment
    ( 'experiment'
    , 'goal', 'Goal'
    , 'text', NULL, NULL
    , 'Experiment goal.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'fov_description', 'FOV description'
    , 'text', NULL, NULL
    , 'Where the field of view sits (in words), e.g. "medial-rostral left bulb".'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'fov_depth_um', 'FOV depth'
    , 'real', NULL, 'um'
    , 'Depth below the surface (used for analysis).'
    , TRUE, FALSE
    ),

    ( 'experiment'
    , 'fov_depth_raw', 'FOV depth (as written)'
    , 'text', NULL, NULL
    , 'The depth as written down, signs and approximation marks kept. (For records only.)'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'depth_class', 'Depth class'
    , 'enum', '["superficial", "deep"]', NULL
    , 'Experiment depth goal. Not recomputed from fov_depth_um.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'pmt_gain', 'PMT gain'
    , 'real', NULL, NULL
    , 'PMT gain setting.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'laser_power_920', 'Laser power 920 nm'
    , 'integer', NULL, '%'
    , 'Rig setting.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'laser_power_1040', 'Laser power 1040 nm'
    , 'integer', NULL, '%'
    , 'Rig setting.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'loop_acq_interval_s', 'Loop acquisition interval'
    , 'real', NULL, 's'
    , 'Programmed interval between acquisitions in a loop.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'treatment', 'Treatment'
    , 'text', NULL, NULL
    , 'What was administered, if anything.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'reporter', 'Reporter'
    , 'text', NULL, NULL
    , 'Reporter population imaged.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'inclusion_status', 'Inclusion'
    , 'enum', '["included", "excluded", "undecided"]', NULL
    , 'Cohort selection. Left undecided until selection is done.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'exclusion_reason', 'Exclusion reason'
    , 'text', NULL, NULL
    , 'Reason this experiment should be excluded.'
    , FALSE, FALSE
    ),

    ( 'experiment'
    , 'note', 'Note'
    , 'text', NULL, NULL
    , 'Notes about this experiment. One entry per note.'
    , FALSE, TRUE
    ),

    ( 'experiment'
    , 'flag', 'Flag'
    , 'text', NULL, NULL
    , 'Something that went wrong. One entry per problem.'
    , FALSE, TRUE
    ),

-- program
    ( 'program'
    , 'description', 'Description'
    , 'text', NULL, NULL
    , 'Block description, e.g. "baseline", "post-ketamine".'
    , FALSE, FALSE
    ),

-- acquisition
    ( 'acquisition'
    , 'exclusion_reason', 'Exclusion reason'
    , 'text', NULL, NULL
    , 'Reason why this acquisition should be excluded.'
    , FALSE, FALSE),

    ( 'acquisition'
    , 'note', 'Note'
    , 'text', NULL, NULL
    , 'Notes about this acquisition.'
    , FALSE, TRUE
    ),

-- group
    ( 'group'
    , 'name', 'Name'
    , 'text', NULL, NULL
    , 'Display name for this group.'
    , FALSE, FALSE),

    ( 'group'
    , 'note', 'Note'
    , 'text', NULL, NULL
    , 'Notes about this group. One entry per note.'
    , FALSE, TRUE);
