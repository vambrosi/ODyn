from .database import Database
from .groups import Group
from .utils import INFO

__all__ = ["Database", "Group"]

# Print a helpful string when user imports this library
print(f"{INFO} Import Database and Group classes for db UI and help functions!")
print(f"{INFO} You can use 'from odyn import Database, Group' to import them.")
print(f"{INFO} Run Database.help() to get examples of how to use ODyn.")
