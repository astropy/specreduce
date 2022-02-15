# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np

from astropy import units as u

from specreduce.core import SpecreduceOperation
from specutils import Spectrum1D

__all__ = ['BoxcarExtract']


def _get_boxcar_weights(center, hwidth, npix):
    """
    Compute the weights given an aperture center, half widths, and number of pixels
    """
    # TODO: this code may fail when regions fall partially or entirely outside the image.

    weights = np.zeros((npix))

    # 2d
    if type(npix) is not tuple:
        # pixels with full weight
        fullpixels = [max(0, int(center - hwidth + 1)), min(int(center + hwidth), npix)]
        weights[fullpixels[0]:fullpixels[1]] = 1.0

        # pixels at the edges of the boxcar with partial weight
        if fullpixels[0] > 0:
            weights[fullpixels[0] - 1] = hwidth - (center - fullpixels[0])
        if fullpixels[1] < npix:
            weights[fullpixels[1]] = hwidth - (fullpixels[1] - center)
    # 3d
    else:
        # pixels with full weight
        fullpixels_x = [max(0, int(center[1] - hwidth + 1)), min(int(center[1] + hwidth), npix[1])]
        fullpixels_y = [max(0, int(center[0] - hwidth + 1)), min(int(center[0] + hwidth), npix[0])]
        weights[fullpixels_x[0]:fullpixels_x[1], fullpixels_y[0]:fullpixels_y[1]] = 1.0

        # not yet handling pixels at the edges of the boxcar

    return weights


def _ap_weight_images(center, width, disp_axis, crossdisp_axis, image_shape):

    """
    Create a weight image that defines the desired extraction aperture.

    Parameters
    ----------
    center : float
        center of aperture in pixels
    width : float
        width of apeture in pixels
    disp_axis : int
        dispersion axis
    crossdisp_axis : int or tuple
        cross-dispersion axis
    image_shape : tuple with 2 or 3 elements
        size (shape) of image
    wavescale : float
        scale the width with wavelength (default=None)
        wavescale gives the reference wavelenth for the width value  NOT USED

    Returns
    -------
    wimage : 2D image, 2D image
        weight image defining the aperature
    """
    wimage = np.zeros(image_shape)
    hwidth = 0.5 * width

    if len(crossdisp_axis) == 1:
        # 2d
        image_sizes = image_shape[crossdisp_axis[0]]
    else:
        # 3d
        image_shape_array = np.array(image_shape)
        crossdisp_axis_array = np.array(crossdisp_axis)
        image_sizes = image_shape_array[crossdisp_axis_array]
        image_sizes = tuple(image_sizes.tolist())

    # loop in dispersion direction and compute weights
    #
    # This loop may be removed or highly cleaned up, replaced by
    # vectorized operations, when the extraction parameters are the
    # same in every image column. We leave it in place for now, so
    # the code may be upgraded later to support height-variable and
    # PSF-weighted extraction modes.

    for i in range(image_shape[disp_axis]):
        if len(crossdisp_axis) == 1:
            wimage[:, i] = _get_boxcar_weights(center, hwidth, image_sizes)
        else:
            wimage[i, ::] = _get_boxcar_weights(center, hwidth, image_sizes)

    return wimage


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

    Parameters
    ----------
    image : nddata-compatible image
        The input image
    trace_object :
        The trace of the spectrum to be extracted TODO: define
    center : float
        center of aperture in pixels
    width : float
        width of aperture in pixels

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted spectrum
    skyspec : `~specutils.Spectrum1D`
        The sky spectrum used in the extraction process
    """
    # TODO: what are reasonable defaults?
    # TODO: ints or floats?
    center: int = 10
    width: int = 8

    #def __call__(self, image, trace_object):
    def __call__(self, image, disp_axis, crossdisp_axis):
        """
        Extract the 1D spectrum using the boxcar method.
        Does a background subtraction as part of the extraction.

        Parameters
        ----------
        image : ndarray
            array with 2-D spectral image data
        disp_axis : int
            dispersion axis
        crossdisp_axis : int or tuple
            cross-dispersion axis


        Returns
        -------
        waves, ext1d : (ndarray, ndarray)
            2D `float` array with wavelengths
            1D `float` array with extracted 1d spectrum in Jy
        """
#        self.last_trace = trace_object
#        self.last_image = image

        for attr in ['center', 'width']:
            if getattr(self, attr) < 1:
                raise ValueError(f'{attr} must be >= 1')

        # images to use for extraction
        wimage = _ap_weight_images(
            self.center,
            self.width,
            disp_axis,
            crossdisp_axis,
            image.shape)

        # extract. Note that, for a cube, this is arbitrarily picking one of the
        # spatial axis to collapse. This should be handled by the API somehow.
        ext1d = np.sum(image * wimage, axis=crossdisp_axis)

        # TODO: add uncertainty and mask to spectrum1D object
        spec = Spectrum1D(spectral_axis=np.arange(len(ext1d)) * u.pixel,
                          flux=ext1d * getattr(image, 'unit', u.DN))

        return spec
