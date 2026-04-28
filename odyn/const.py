from enum import Enum

INFO = "\033[1;34mINFO\033[0m"
PASS = "\033[1;32mTEST\033[0m"
FAIL = "\033[1;31mTEST\033[0m"
CHECK = "\033[1;32m\u2714\033[0m"
CROSS = "\033[1;31m\u2718\033[0m"


class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"
