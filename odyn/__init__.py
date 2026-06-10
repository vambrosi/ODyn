from .database import Database
from .groups import Group
from .utils import logger

__all__ = ["Database", "Group"]

# Print a helpful string when user imports this library
logger.info("Import Database and Group classes for db UI and help functions!")
logger.info("You can use 'from odyn import Database, Group' to import them.")
logger.info("Run Database.help() to get examples of how to use ODyn.")
