#!/usr/bin/python3

import logging
from sys import argv

import coloredlogs


class Loggable(object):
    @classmethod
    def get_default_logger(cls):
        logger = getattr(cls, '_default_logger', None)
        if not logger:
            logger = logging.getLogger(__name__)
            coloredlogs.install('DEBUG', logger=logger)
            cls._default_logger = logger

        return logger


if __name__ == '__main__':
    print(argv[0])
