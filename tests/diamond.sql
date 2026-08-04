-- Diamond integrity check (expected to FAIL with a FOREIGN KEY error).
--
-- A trial's program and acquisition must belong to the same experiment. Here
-- program 1 is on experiment 2, but the trial claims experiment 1, so the
-- composite FOREIGN KEY (program_id, exp_id) -> programs must reject it.
PRAGMA foreign_keys = ON;

INSERT INTO experiments
    ( exp_name, exp_type, exp_start
    , mouse_id, height_px, width_px, height_um, width_um
    , frame_count, frame_rate, laser_power_920, laser_power_1040
    , loop_acq_interval_s
    ) VALUES
        ( 'e1', 'loop', '2025-01-01 00:00:00'
        , 'm1', 100, 200, 300, 400
        , 30, 60, 0, 8
        , 2.0
        ),
        ( 'e2', 'loop', '2025-01-01 01:00:00'
        , 'm1', 100, 200, 300, 400
        , 30, 60, 0, 8
        , 2.0
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
    , odor_start
    , odor_end
    , odor_id, outcome
    , acq_id, h5_to_trial_ms
    , program_id, exp_id
    ) VALUES
        ( '2025-01-01 00:00:00'
        , '2025-01-01 00:00:01'
        , '2025-01-01 00:00:02'
        , 1, 'na'
        , 1, 0.0
        , 1, 1
        );
