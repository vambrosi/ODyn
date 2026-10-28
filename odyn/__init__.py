from .database import Database
from .groups import Group
from .utils import interactive, logger, record_call

__all__ = ["Database", "Group", "record_call"]

# A hint for a person typing (a notebook, the prompt); not in jobs and logs
if interactive():
    logger.info("Import Database and Group classes for db UI and help functions!")
    logger.info("You can use 'from odyn import Database, Group' to import them.")
    logger.info("Hovering over 'Database' and 'Group' will give you some tips.")
