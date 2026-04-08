def normalize_config(config):
    """
    Edit config so that parameters are in valid ranges.
    """
    size_ums = config["metadata"]["size_ums"]
    max_shift_um = config["test"]["motion_correction"]["max_shift_um"]

    # The coordinates of max_shift_um must be between 0 and image_size / 4
    # where image_size is the size of the image in the corresponding axis.
    # The upper bound is more or less arbitrary, but it needs to be
    # below image_size / 2

    config["test"]["motion_correction"]["max_shift_um"] = list(
        map(
            lambda s: clamp(s[0], 0, s[1] / 4),
            (s for s in zip(max_shift_um, size_ums)),
        )
    )


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
