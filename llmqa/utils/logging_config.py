"""Logging configuration for the LLMQA package.

This module provides a centralized configuration for logging across the package.
"""

import logging
import sys
import time
import functools
from typing import Optional

def measure_logging_overhead(func):
    """Decorator to measure logging overhead in a function.
    
    Usage:
        @measure_logging_overhead
        def your_function():
            logger.debug("Your debug message")
            # rest of function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        duration = (end_time - start_time) / 1_000_000  # Convert to milliseconds
        print(f"Logging overhead for {func.__name__}: {duration:.3f}ms")
        return result
    return wrapper

def setup_logging(level: Optional[int] = None, verbose: bool = False):
    """Set up logging configuration.
    
    Args:
        level (int, optional): The logging level to use. If None, uses INFO or DEBUG based on verbose flag.
        verbose (bool): Whether to enable verbose (DEBUG) logging.
    """
    # Set default level based on verbose flag
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('llmqa')
    logger.setLevel(level)
    
    # Create formatters - one detailed for DEBUG, one simple for other levels
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a filter for DEBUG level messages
    class DebugLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.DEBUG

    # Create separate handlers for DEBUG and other levels
    debug_handler = logging.StreamHandler(sys.stdout)
    debug_handler.setFormatter(debug_formatter)
    debug_handler.addFilter(DebugLevelFilter())
    debug_handler.setLevel(logging.DEBUG)

    other_handler = logging.StreamHandler(sys.stdout)
    other_handler.setFormatter(simple_formatter)
    other_handler.setLevel(logging.INFO)
    
    # Add handlers to logger if it doesn't already have them
    if not logger.handlers:
        logger.addHandler(debug_handler)
        logger.addHandler(other_handler)
    
    return logger