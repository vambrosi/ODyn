from enum import Enum

INFO = "\033[1;34mINFO\033[0m"

class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"