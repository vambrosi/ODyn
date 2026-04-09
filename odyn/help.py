help_strings = {
    "Experiment": """
\033[1;31mEXPERIMENT\033[0m
Class that runs data processing/analysis.

\033[1;34mUSAGE\033[0m
    exp = Experiment(experimentFolder)

\033[1;34mRELEVANT METHODS\033[0m
    exp.run_motion_correction()
    exp.play_movie()
    exp.delete_temp_files()

Run Experiment.help('method_name') to know more about one of the methods above.

\033[1;34mEXAMPLE\033[0m
    Experiment.help('play_movie')
""",
    "run_motion_correction": """
\033[1;31mRUN_MOTION_CORRECTION\033[0m
Method that does test/final motion correction

\033[1;34mUSAGE\033[0m
    exp = Experiment(experimentFolder)
    exp.run_motion_correction(...)

\033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)
    use_last_parameters = False             Use last run's parameters as the defaults
    is_test             = True              Whether to use a limit range of acquisitions in this run
    first_acq           = 1                 Number of the first acquisition to motion correct
    step_acq            = 1                 Get one acquisition for every 'step_acq' acquisitions
    last_acq            = 3                 Number of the last acquisition to motion correct
    border_nan          = "copy"            copy along the boundary (if True, fill in with NaN)
    nonneg_movie        = False             make SAVED movie mostly non-negative
    pw_rigid            = True              Piecewise-rigid (True) or rigid motion correction
    shifts_opencv       = False             True = bicubic, False = FFT (True is faster)
    max_deviation_um    = 12.0              max deviation for patch with respect to rigid shifts
    max_shift_um        = [128.0, 128.0]    max allowed rigid shift
    overlap_um          = [96.0, 96.0]      overlap between patches (patch = strides + overlaps)
    strides_um          = [128.0, 128.0]    start a new patch every x or y um (only for pw-rigid)

\033[1;34mEXAMPLES\033[0m
    exp.run_motion_correction(is_test=False, last_acq=10)
""",
    "delete_temp_files": """
\033[1;31mDELETE_TEMP_FILES\033[0m
Deletes all temp files associated with this experiment

\033[1;34mUSAGE\033[0m
    exp = Experiment(experimentFolder)
    exp.delete_temp_files()

\033[1;34mEXAMPLES\033[0m
    exp.delete_temp_files()
""",
}
