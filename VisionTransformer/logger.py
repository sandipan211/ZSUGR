import os
import sys
import logging
import functools
from termcolor import colored

@functools.lru_cache()
def create_logger(output_dir, eval=False, setting_name=None):
    # create logger
    name = 'eval_' if eval else 'train_'
    name += setting_name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    # if dist_rank == 0:
    #     console_handler = logging.StreamHandler(sys.stdout)
    #     console_handler.setLevel(logging.DEBUG)
    #     console_handler.setFormatter(
    #         logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    #     logger.addHandler(console_handler)

    # create file handlers
    file_name = 'log_'+name+'.txt' 
    file_handler = logging.FileHandler(os.path.join(output_dir, file_name), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger