# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np
from astropy.nddata import CCDData

from specreduce.core import SpecreduceOperation
from specreduce.extract import _ap_weight_image
from specreduce.tracing import Trace, FlatTrace

__all__ = ['Background']


@dataclass
class Background(SpecreduceOperation):
    """
    Determine the background from an image for subtraction

    Parameters
    ----------
    image : nddata-compatible image
        image with 2-D spectral image data
    trace_object : Trace
        trace object
    width : float
        width of extraction aperture in pixels
    disp_axis : int
        dispersion axis
    crossdisp_axis : int
        cross-dispersion axis

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted 1d spectrum expressed in DN and pixel units
    """
    # required so numpy won't call __rsub__ on individual elements
    # https://stackoverflow.com/a/58409215
    __array_ufunc__ = None

    image: CCDData
    trace_object: Trace
    separation: float = 5
    width: float = 5
    disp_axis: int = 1
    crossdisp_axis: int = 0

    def __post_init__(self):
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image : nddata-compatible image
            image with 2-D spectral image data
        trace_object : Trace or int
            trace object or an integer to use a FloatTrace
        separation: float
            separation between trace and extraction apertures on each
            side of the trace
        width : float
            width of each background aperture in pixels
        disp_axis : int
            dispersion axis
        crossdisp_axis : int
            cross-dispersion axis
        """
        if isinstance(self.trace_object, (int, float)):
            self.trace_object = FlatTrace(self.image, self.trace_object)

        # TODO: this check can be removed if/when implemented as a check in FlatTrace
        if isinstance(self.trace_object, FlatTrace):
            if self.trace_object.trace_pos < 1:
                raise ValueError('trace_object.trace_pos must be >= 1')

        bkg_wimage = _ap_weight_image(
            self.trace_object-self.separation,
            self.width,
            self.disp_axis,
            self.crossdisp_axis,
            self.image.shape)

        bkg_wimage += _ap_weight_image(
            self.trace_object+self.separation,
            self.width,
            self.disp_axis,
            self.crossdisp_axis,
            self.image.shape)

        self.bkg_wimage = bkg_wimage
        self.bkg_array = np.average(self.image, weights=self.bkg_wimage, axis=0)

    def bkg_image(self, image=None):
        """
        Expose the background tiled to the dimension of ``image``.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will use ``image`` passed
            to extract the background.

        Returns
        -------
        array with same shape as ``image``.
        """
        if image is None:
            image = self.image

        return np.tile(self.bkg_array, (image.shape[0], 1))

    def sub_image(self, image=None):
        """
        Subtract the computed background from ``image``.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will use ``image`` passed
            to extract the background.

        Returns
        -------
        array with same shape as ``image``
        """
        if image is None:
            image = self.image

        return image - self.bkg_image(image)

    def __rsub__(self, image):
        """
        Subtract the background from an image.
        """
        return self.sub_image(image)
