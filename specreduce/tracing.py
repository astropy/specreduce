# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData

__all__ = ['BasicTrace']


@dataclass
class BasicTrace:
    """
    Basic tracing class that traces a constant horizontal position, trace_pos, in the image.

    Parameters
    ----------
    image : `~astropy.nddata.CCDData`
        Image to be traced
    trace_pos : float
        Position of trace along vertical axis
    """
    image: CCDData
    trace_pos: float

    def __post_init__(self):
        self.trace = np.ones_like(self.image[0]) * self.trace_pos

    def __getitem__(self, i):
        return self.trace[i]

    def __call__(self, i):
        return self.trace[i]

    @property
    def shape(self):
        return self.trace.shape
