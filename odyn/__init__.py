from .database import Database
from .groups import Group
from .utils import logger, record_call

__all__ = ["Database", "Group", "record_call"]

# Print a helpful string when user imports this library
logger.info("Import Database and Group classes for db UI and help functions!")
logger.info("You can use 'from odyn import Database, Group' to import them.")
logger.info("Hovering over 'Database' and 'Group' will give you some tips.")
