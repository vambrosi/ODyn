-- Valid-chain insertion check (expected to SUCCEED).
--
-- A consistent mouse -> session -> experiment -> program / acquisition -> trial
-- -> event chain, all on the same experiment, must insert without error, and so
-- must a sync block, a panel and an annotation hanging off it.
PRAGMA foreign_keys = ON;

INSERT INTO users (user_name, display_name) VALUES ('someone', 'Someone');
INSERT INTO mice (mouse_id, mouse_sex) VALUES (1, 'F');
INSERT INTO sessions (session_id, mouse_id, session_date) VALUES (1, 1, '2025-02-01');

INSERT INTO experiments
    ( session_id, exp_name, exp_type, exp_path, exp_start
    , height_px, width_px, height_um, width_um
    , frame_count, frame_rate, objective
    ) VALUES
        ( 1, 'e1', 'loop', '20250201/m1/e1', '2025-02-01 00:00:00'
        , 100, 200, 300, 400
        , 30, 60, 20
        );

INSERT INTO programs (exp_id, program_name, program_type, program_start, program_path)
    VALUES (1, 'p', 'passive', '2025-02-01 00:00:00', 'path');

INSERT INTO acquisitions (exp_id, acq_start, raw_path)
    VALUES (1, '2025-02-01 00:00:00', 'raw');

INSERT INTO trials
    ( trial_start
    , trial_odor_start
    , trial_odor_end
    , odor_id, outcome
    , acq_id
    , program_id, exp_id
    ) VALUES
        ( '2025-02-01 00:00:00'
        , '2025-02-01 00:00:01'
        , '2025-02-01 00:00:02'
        , 1, 'na'
        , 1
        , 1, 1
        );

INSERT INTO events (event_time, event_type, event_tag, program_id, trial_id)
    VALUES ('2025-02-01 00:00:00', 'Trial', '1', 1, 1);

INSERT INTO method_calls
    ( group_id, user, method_name, module, code, parameter_inputs, parameters_used )
    VALUES (0, 'someone', 'Database.decode_sync', 'odyn.database', '{}', '{}', '{}');

INSERT INTO streams (exp_id, stream_type, file_path, rate_hz, method_call_id)
    VALUES (1, 'sync', '20250201/m1/e1/sync/m1_e1.h5', 5000, 1);

INSERT INTO sync_blocks
    ( stream_id, sync_block, acq_id, sync_start_sample, sync_frame_count
    , sync_frame_period_s, sync_odor_pulses, sync_odor_on_frame, method_call_id
    ) VALUES (1, 0, 1, 100, 30, 0.0166, 1, 5, 1);

INSERT INTO panels (panel_name) VALUES ('print_v3');
INSERT INTO panel_positions (panel_id, vial_position, odor_id) VALUES (1, 1, 1);
INSERT INTO vials (panel_id, vial_position, vial_sccm, total_sccm, total_volume_ml, made_on)
    VALUES (1, 1, 200, 1500, 5, '2025-01-30');
INSERT INTO vial_odors (vial_id, odor_id, target_ppm, liquid_ul, percent_vv)
    VALUES (1, 1, 0.3, 12.5, 0.25);
INSERT INTO experiment_vials (exp_id, vial_id) VALUES (1, 1);

INSERT INTO annotations (target_type, target_id, key, value, method_call_id)
    VALUES ('experiment', 1, 'fov_depth_um', 70, 1);
