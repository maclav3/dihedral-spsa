#!/usr/bin/python3

import logging
from sys import argv

import coloredlogs

_logger = None


def get_default_logger():
    if not _logger:
        logger = logging.getLogger(__name__)
        coloredlogs.install('DEBUG', logger=logger)
    return _logger


if __name__ == '__main__':
    print(argv[0])
