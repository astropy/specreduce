# Licensed under a 3-clause BSD style license - see LICENSE.rst

from specreduce.core import *  # noqa
from specreduce.wavelength_calibration import * # noqa


try:
    from .version import version as __version__
except ImportError:
    __version__ = ''
