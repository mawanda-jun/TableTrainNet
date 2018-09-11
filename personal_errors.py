"""
This is a personalized class for managing exceptions in project.
This is also responsible for writing error messages inside log file.

I decided to create a general class and then use subclasses for the sake of decoupling
"""
import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(TimeHandler().handler)


class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message):
        self.message = message
        logger.error(message)

    def __str__(self):
        return 'Some error occurred: \n{msg}\nAborting...'.format(msg=self.message)


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, message):
        super().__init__(message)


class OutputError(Error):
    """Exception raised for errors in the output.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, message):
        super().__init__(message)


class APIError(Error):
    """
    Exception raised for general API call exceptions
    """
    def __init__(self, message):
        super().__init__(message)
