-- Diamond integrity check (expected to FAIL with a FOREIGN KEY error).
--
-- A trial's program must belong to the trial's experiment. Here program 1 is
-- on experiment 2, but the trial claims experiment 1, so the composite
-- FOREIGN KEY (program_id, exp_id) -> programs must reject it.
PRAGMA foreign_keys = ON;

INSERT INTO mice (mouse_id) VALUES (1);
INSERT INTO sessions (session_id, mouse_id, session_date) VALUES (1, 1, '2025-01-01');

INSERT INTO experiments
    ( session_id, exp_number, exp_path, exp_start
    , height_px, width_px, height_um, width_um
    , frames_per_slice, frame_rate, objective, exp_type
    ) VALUES
        ( 1, 1, '20250101/m1/e1', '2025-01-01 00:00:00'
        , 100, 200, 300, 400
        , 30, 60, 20, 'loop'
        ),
        ( 1, 2, '20250101/m1/e2', '2025-01-01 01:00:00'
        , 100, 200, 300, 400
        , 30, 60, 20, 'loop'
        );

-- acquisition on experiment 1 -> acq_id 1
INSERT INTO acquisitions (exp_id, acq_start, raw_path, frame_count)
    VALUES (1, '2025-01-01 00:00:00', 'raw1', 30);

-- program on experiment 2 -> program_id 1
INSERT INTO programs (exp_id, program_name, program_type, program_start, program_path)
    VALUES (2, 'p', 'passive', '2025-01-01 01:00:00', 'path');

-- trial on experiment 1 with program 1 (experiment 2): the composite foreign
-- key cannot be satisfied, so this insert must fail.
INSERT INTO trials
    ( trial_start
    , trial_odor_start
    , trial_odor_end
    , odor_id, outcome
    , acq_id
    , program_id, exp_id
    ) VALUES
        ( '2025-01-01 00:00:00'
        , '2025-01-01 00:00:01'
        , '2025-01-01 00:00:02'
        , 1, 'na'
        , 1
        , 1, 1
        );
