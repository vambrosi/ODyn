CREATE TABLE IF NOT EXISTS metadata
    ( id INTEGER PRIMARY KEY
    , exp_date TEXT NOT NULL
    , mouse_id TEXT NOT NULL
    , exp_name TEXT NOT NULL
    , first_acq INTEGER NOT NULL
    , last_acq INTEGER NOT NULL
    , n_acq INTEGER NOT NULL
    , height_px INTEGER NOT NULL
    , width_px INTEGER NOT NULL
    , height_um REAL NOT NULL
    , width_um REAL NOT NULL
    , frame_count INTEGER NOT NULL
    , frame_rate REAL NOT NULL
    , tiff_stem TEXT NOT NULL
    );

CREATE TABLE IF NOT EXISTS acquisitions
    ( id INTEGER PRIMARY KEY
    , raw_filename TEXT NOT NULL
    , should_include BOOLEAN NOT NULL
    );

CREATE TABLE IF NOT EXISTS function_calls
    ( id INTEGER PRIMARY KEY
    , call_time TEXT NOT NULL
    , object_class TEXT NOT NULL
    , method_name TEXT NOT NULL
    , git_commit TEXT NOT NULL
    );

CREATE TABLE IF NOT EXISTS parameters
    ( id INTEGER PRIMARY KEY
    , parameter_name TEXT NOT NULL
    , parameter_type TEXT NOT NULL
    , parameter_value TEXT NOT NULL
    , function_call_id INTEGER REFERENCES function_calls(id)
    );
