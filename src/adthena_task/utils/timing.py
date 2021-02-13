"""Implements time utility functions."""

import inspect
import logging
import time
from typing import Any, Callable


def timed(f: Callable, logger: logging) -> Any:
    """Function to calculate and log time needed to execute some function.

    Args:
        f: Callable to be measured time with. Returns output of callable.
        logger: Logger in current file.

    Returns:
        Output of Callable f of any type.
    """
    start = time.time()
    ret = f()
    elapsed = time.time() - start
    source_code = inspect.getsource(f).strip("\n")
    logger.info(source_code + ":  " + str(elapsed) + " seconds")
    return ret
