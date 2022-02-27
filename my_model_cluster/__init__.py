import os
from .estimator import Estimator

name = "textaugment"

__version__ = '1.0.0'
__licence__ = 'vadim_abronin'
__author__ = 'vadim_abronin'
__url__ = 'https://github.com/amazon-research/sccl'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ['Estimator']
