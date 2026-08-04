-- Valid-chain insertion check (expected to SUCCEED).
--
-- A consistent experiment -> program / acquisition -> trial -> event chain, all
-- on the same experiment, must insert without error.
PRAGMA foreign_keys = ON;

INSERT INTO experiments
    ( exp_name, exp_type, exp_start
    , mouse_id, height_px, width_px, height_um, width_um
    , frame_count, frame_rate, laser_power_920, laser_power_1040
    , loop_acq_interval_s
    ) VALUES
        ( 'e1', 'loop', '2025-02-01 00:00:00'
        , 'm1', 100, 200, 300, 400
        , 30, 60, 0, 8
        , 2.0
        );

INSERT INTO programs (exp_id, program_name, program_type, program_start, program_path)
    VALUES (1, 'p', 'passive', '2025-02-01 00:00:00', 'path');

INSERT INTO acquisitions (exp_id, acq_start, raw_path)
    VALUES (1, '2025-02-01 00:00:00', 'raw');

INSERT INTO trials
    ( trial_start
    , odor_start
    , odor_end
    , odor_id, outcome
    , acq_id, h5_to_trial_ms
    , program_id, exp_id
    ) VALUES
        ( '2025-02-01 00:00:00'
        , '2025-02-01 00:00:01'
        , '2025-02-01 00:00:02'
        , 1, 'na'
        , 1, 0.0
        , 1, 1
        );

INSERT INTO events (event_time, event_type, event_tag, program_id, trial_id)
    VALUES ('2025-02-01 00:00:00', 'Trial', '1', 1, 1);
