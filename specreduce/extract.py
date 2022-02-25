# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np

from astropy import units as u

from specreduce.core import SpecreduceOperation
from specreduce.tracing import FlatTrace
from specutils import Spectrum1D

__all__ = ['BoxcarExtract']


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

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

    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    def __call__(self, image, trace_object, width=5,
                 disp_axis=1, crossdisp_axis=0):
        """
        Extract the 1D spectrum using the boxcar method.

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
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        def _get_boxcar_weights(center, hwidth, npix):
            """
            Compute weights given an aperture center, half width,
            and number of pixels
            """
            weights = np.zeros((npix))

            # pixels with full weight
            fullpixels = [max(0, int(center - hwidth + 1)),
                          min(int(center + hwidth), npix)]
            weights[fullpixels[0]:fullpixels[1]] = 1.0

            # pixels at the edges of the boxcar with partial weight
            if fullpixels[0] > 0:
                w = hwidth - (center - fullpixels[0] + 0.5)
                if w >= 0:
                    weights[fullpixels[0] - 1] = w
                else:
                    weights[fullpixels[0]] = 1. + w
            if fullpixels[1] < npix:
                weights[fullpixels[1]] = hwidth - (fullpixels[1] - center - 0.5)

            return weights

        def _ap_weight_image(trace, width, disp_axis, crossdisp_axis, image_shape):

            """
            Create a weight image that defines the desired extraction aperture.

            Parameters
            ----------
            trace : Trace
                trace object
            width : float
                width of extraction aperture in pixels
            disp_axis : int
                dispersion axis
            crossdisp_axis : int
                cross-dispersion axis
            image_shape : tuple with 2 elements
                size (shape) of image

            Returns
            -------
            wimage : 2D image
                weight image defining the aperture
            """
            wimage = np.zeros(image_shape)
            hwidth = 0.5 * width
            image_sizes = image_shape[crossdisp_axis]

            # loop in dispersion direction and compute weights.
            for i in range(image_shape[disp_axis]):
                # TODO trace must handle transposed data (disp_axis == 0)
                wimage[:, i] = _get_boxcar_weights(trace[i], hwidth, image_sizes)

            return wimage

        # TODO: this check can be removed if/when implemented as a check in FlatTrace
        if isinstance(trace_object, FlatTrace):
            if trace_object.trace_pos < 1:
                raise ValueError('trace_object.trace_pos must be >= 1')

        # weight image to use for extraction
        wimage = _ap_weight_image(
            trace_object,
            width,
            disp_axis,
            crossdisp_axis,
            image.shape)

        # extract
        ext1d = np.sum(image * wimage, axis=crossdisp_axis)

        # TODO: add wavelenght units, uncertainty and mask to spectrum1D object
        spec = Spectrum1D(spectral_axis=np.arange(len(ext1d)) * u.pixel,
                          flux=ext1d * getattr(image, 'unit', u.DN))

        return spec
