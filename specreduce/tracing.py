# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData

__all__ = ['Trace', 'FlatTrace', 'ArrayTrace']


@dataclass
class Trace:
    """
    Basic tracing class that by default traces the middle of the image.

    Parameters
    ----------
    image : `~astropy.nddata.CCDData`
        Image to be traced

    Properties
    ----------
    shape : tuple
        Shape of the array describing the trace
    """
    image: CCDData

    def __post_init__(self):
        self.trace_pos = self.image.shape[0] / 2
        self.trace = np.ones_like(self.image[0]) * self.trace_pos

    def __getitem__(self, i):
        return self.trace[i]

    @property
    def shape(self):
        return self.trace.shape

    def shift(self, delta):
        """
        Shift the trace by delta pixels perpendicular to the axis being traced

        Parameters
        ----------
        delta : float
            Shift to be applied to the trace
        """
        self.trace += delta
        self._bound_trace()

    def _bound_trace(self):
        """
        Mask trace positions that are outside the upper/lower bounds of the image.
        """
        ny = self.image.shape[0]
        self.trace = np.ma.masked_outside(self.trace, 0, ny-1)


@dataclass
class FlatTrace(Trace):
    """
    Trace that is constant along the axis being traced

    Parameters
    ----------
    trace_pos : float
        Position of the trace
    """
    trace_pos: float

    def __post_init__(self):
        self.set_position(self.trace_pos)

    def set_position(self, trace_pos):
        """
        Set the trace position within the image

        Parameters
        ----------
        trace_pos : float
            Position of the trace
        """
        self.trace_pos = trace_pos
        self.trace = np.ones_like(self.image[0]) * self.trace_pos
        self._bound_trace()


@dataclass
class ArrayTrace(Trace):
    """
    Define a trace given an array of trace positions

    Parameters
    ----------
    trace : `numpy.ndarray`
        Array containing trace positions
    """
    trace: np.ndarray

    def __post_init__(self):
        nx = self.image.shape[1]
        nt = len(self.trace)
        if nt != nx:
            if nt > nx:
                # truncate trace to fit image
                self.trace = self.trace[0:nx]
            else:
                # assume trace starts at beginning of image and pad out trace to fit.
                # padding will be the last value of the trace, but will be masked out.
                padding = np.ma.MaskedArray(np.ones(nx - nt) * self.trace[-1], mask=True)
                self.trace = np.ma.hstack([self.trace, padding])
        self._bound_trace()
