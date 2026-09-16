import pytest

from odyn.utils import logger


@pytest.fixture
def odyn_log(caplog):
    """`caplog` for odyn's logger, which does not pass records up to the root."""
    logger.addHandler(caplog.handler)

    yield caplog
    logger.removeHandler(caplog.handler)
