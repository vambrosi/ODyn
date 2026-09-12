-- Diamond integrity check (expected to FAIL with a FOREIGN KEY error).
--
-- A trial's program and acquisition must belong to the same experiment. Here
-- program 1 is on experiment 2, but the trial claims experiment 1, so the
-- composite FOREIGN KEY (program_id, exp_id) -> programs must reject it.
--
-- Literal SQL on purpose: the schema is what this file tests, so it has to be
-- updated by hand when the schema changes.
PRAGMA foreign_keys = ON;

INSERT INTO sessions (session_id, mouse_id, session_date, session_path)
    VALUES (1, 'm1', '2025-01-01', '20250101/m1');

INSERT INTO experiments
    ( session_id, exp_name, exp_type, exp_start
    , height_px, width_px, height_um, width_um
    , frame_count, frame_rate
    ) VALUES
        ( 1, 'e1', 'loop', '2025-01-01 00:00:00'
        , 100, 200, 300, 400
        , 30, 60
        ),
        ( 1, 'e2', 'loop', '2025-01-01 01:00:00'
        , 100, 200, 300, 400
        , 30, 60
        );

-- acquisition on experiment 1 -> acq_id 1
INSERT INTO acquisitions (exp_id, acq_start, raw_path)
    VALUES (1, '2025-01-01 00:00:00', 'raw1');

-- program on experiment 2 -> program_id 1
INSERT INTO programs (exp_id, program_name, program_type, program_start, program_path)
    VALUES (2, 'p', 'passive', '2025-01-01 01:00:00', 'path');

-- trial uses acquisition 1 (experiment 1) and program 1 (experiment 2): the
-- composite foreign keys can't both be satisfied, so this insert must fail.
INSERT INTO trials
    ( trial_start
    , trial_odor_start
    , trial_odor_end
    , odor_id, outcome
    , acq_id, sync_to_trial_ms
    , program_id, exp_id
    ) VALUES
        ( '2025-01-01 00:00:00'
        , '2025-01-01 00:00:01'
        , '2025-01-01 00:00:02'
        , 1, 'na'
        , 1, 0.0
        , 1, 1
        );
