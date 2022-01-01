# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData

__all__ = ['Trace']


@dataclass
class Trace:
    """
    Basic tracing class that by default traces a constant horizontal
    position, trace_pos, in the image.

    Parameters
    ----------
    image : `~astropy.nddata.CCDData`
        Image to be traced
    trace_pos : float
        Position of trace along vertical axis. If not specified, set to middle
        of vertical axis
    """
    image: CCDData
    trace_pos: float = None

    def __post_init__(self):
        if self.trace_pos is None:
            self.trace_pos = self.image.shape[0] / 2
        self.__call__(self.trace_pos)

    def __getitem__(self, i):
        return self.trace[i]

    def __call__(self, trace_pos):
        """
        Set vertical position of the trace and calculate the trace
        """
        self.trace_pos = trace_pos
        self.trace = np.ones_like(self.image[0]) * self.trace_pos
        return self.trace

    @property
    def shape(self):
        return self.trace.shape
