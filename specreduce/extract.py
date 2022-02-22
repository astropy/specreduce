# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np

from astropy import units as u

from specreduce.core import SpecreduceOperation
from specutils import Spectrum1D

__all__ = ['BoxcarExtract']


def _get_boxcar_weights(center, hwidth, npix):
    """
    Compute weights given an aperture center, half width, and number of pixels
    """
    weights = np.zeros((npix))

    # 2d
    if type(npix) is not tuple:
        # pixels with full weight
        fullpixels = [max(0, int(center - hwidth + 1)), min(int(center + hwidth), npix)]
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
    # 3d
    else:
        # pixels with full weight
        fullpixels_x = [max(0, int(center[1] - hwidth + 1)), min(int(center[1] + hwidth), npix[1])]
        fullpixels_y = [max(0, int(center[0] - hwidth + 1)), min(int(center[0] + hwidth), npix[0])]
        weights[fullpixels_x[0]:fullpixels_x[1], fullpixels_y[0]:fullpixels_y[1]] = 1.0

        # not yet handling pixels at the edges of the boxcar

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
    crossdisp_axis : int (2D image) or tuple (3D image)
        cross-dispersion axis
    image_shape : tuple with 2 or 3 elements
        size (shape) of image

    Returns
    -------
    wimage : 2D image
        weight image defining the aperture
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

    # loop in dispersion direction and compute weights.
    for i in range(image_shape[disp_axis]):
        if len(crossdisp_axis) == 1:
            # 2d
            # TODO trace must handle transposed data (disp_axis == 0)
            wimage[:, i] = _get_boxcar_weights(trace[i], hwidth, image_sizes)
        else:
            # 3d
            wimage[i, ::] = _get_boxcar_weights(trace[i], hwidth, image_sizes)

    return wimage


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

    Parameters
    ----------
    image : nddata-compatible image
        image with 2-D spectral image data
    width : float
        width of extraction aperture in pixels

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted 1d spectrum expressed in DN and pixel units
    """
    # TODO: what is a reasonable default?
    # TODO: int or float?
    width: int = 5

    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    def __call__(self, image, trace_object, disp_axis=1, crossdisp_axis=(0,)):
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image : nddata-compatible image
            image with 2-D spectral image data
        trace_object : Trace
            object with the trace
        disp_axis : int
            dispersion axis
        crossdisp_axis : tuple (to support both 2D and 3D data)
            cross-dispersion axis


        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The extracted 1d spectrum expressed in DN and pixel units
        """
        # this check only applies to FlatTrace instances
        if hasattr(trace_object, 'trace_pos'):
            self.center = trace_object.trace_pos
            for attr in ['center', 'width']:
                if getattr(self, attr) < 1:
                    raise ValueError(f'{attr} must be >= 1')

        # images to use for extraction
        wimage = _ap_weight_image(
            trace_object,
            self.width,
            disp_axis,
            crossdisp_axis,
            image.shape)

        # extract. Note that, for a cube, this is arbitrarily picking one of the
        # spatial axis to collapse. This should be handled by the API somehow.
        ext1d = np.sum(image * wimage, axis=crossdisp_axis)

        # TODO: add wavelenght units, uncertainty and mask to spectrum1D object
        spec = Spectrum1D(spectral_axis=np.arange(len(ext1d)) * u.pixel,
                          flux=ext1d * getattr(image, 'unit', u.DN))

        return spec
