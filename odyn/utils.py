from dataclasses import dataclass

from .const import INFO


# TODO: - Fix the logging interaction with this class
@dataclass
class ProgressBar:
    total: int
    current: int = 0

    def show(self) -> None:
        filled_squares = int(40 * self.current / self.total)

        progress_str = f"[{INFO}] [ \033[1;34m"
        progress_str += "\u2588" * filled_squares
        progress_str += "-" * (40 - filled_squares)
        progress_str += f"\033[0m ] {self.current:03d}/{self.total:03d} Files processed"

        print(progress_str, end="\r")

    def step(self) -> None:
        self.current = min(self.current + 1, self.total)
        self.show()

    def end(self) -> None:
        msg = f"[{INFO}] {self.total} files processed sucessfully."
        print(f"{msg:<90}")


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
