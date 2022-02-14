# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.nddata import StdDevUncertainty

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


def _ap_weight_images(center, width, disp_axis, crossdisp_axis, bkg_offset, bkg_width, image_shape):

    """
    Create a weight image that defines the desired extraction aperture
    and the weight image for the requested background regions.

    The disp_axis and crossdisp_axis parameters could perhaps be derived from the
    jdatamodel wcs and/or meta instances. Since we have test data that lacks the 
    these, for now we pass then explictly via calling sequence.


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
    bkg_offset : float
        offset from the extaction edge for the background
        never scaled for wavelength
    bkg_width : float
        width of background region
        never scaled with wavelength
    image_shape : tuple with 2 or 3 elements
        size (shape) of image
    wavescale : float
        scale the width with wavelength (default=None)
        wavescale gives the reference wavelenth for the width value  NOT USED

    Returns
    -------
    wimage, bkg_wimage : (2D image, 2D image)
        wimage is the weight image defining the aperature
        bkg_image is the weight image defining the background regions
    """
    wimage = np.zeros(image_shape)
    bkg_wimage = np.zeros(image_shape)
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

        # bkg regions (only for s2d for now)
        if (len(crossdisp_axis) == 1) & (bkg_width is not None) & (bkg_offset is not None):
            bkg_wimage[:, i] = _get_boxcar_weights(
                center - hwidth - bkg_offset, bkg_width, image_shape[0]
            )
            bkg_wimage[:, i] += _get_boxcar_weights(
                center + hwidth + bkg_offset, bkg_width, image_shape[0]
            )
        else:
            bkg_wimage = None

    return (wimage, bkg_wimage)


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

    Parameters
    ----------
    img : nddata-compatible image
        The input image
    trace_object :
        The trace of the spectrum to be extracted TODO: define
    apwidth : int
        The width of the extraction aperture in pixels
    skysep : int
        The spacing between the aperture and the sky regions
    skywidth : int
        The width of the sky regions in pixels
    skydeg : int
        The degree of the polynomial that's fit to the sky

    center : float
        center of aperture in pixels
    width : float
        width of aperture in pixels
    bkg_offset : float
        offset from the extaction edge for the background
        never scaled for wavelength
    bkg_width : float
        width of background region
        never scaled with wavelength

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
    bkg_offset: int = 1
    bkg_width: int = 8

    #def __call__(self, image, trace_object, pixelarea):
    def __call__(self, image, disp_axis, crossdisp_axis, pixelarea=1):
        """
        Extract the 1D spectrum using the boxcar method.
        Does a background subtraction as part of the extraction.

        Parameters
        ----------
        image : ndarray
            array with 2-D spectral image data
        jdatamodel : jwst.DataModel
            jwst datamodel with the 2d spectral image

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
        self.last_image = image

        for attr in ['center', 'width', 'bkg_offset', 'bkg_width']:
            if getattr(self, attr) < 1:
                raise ValueError(f'{attr} must be >= 1')

        # images to use for extraction
        wimage, bkg_wimage = _ap_weight_images(
            self.center,
            self.width,
            disp_axis,
            crossdisp_axis,
            self.bkg_width,
            self.bkg_offset,
            image.shape
        )

        # select weight images
        if bkg_wimage is not None:
            ext1d_boxcar_bkg = np.average(image, weights=bkg_wimage, axis=0)
            data_bkgsub = image - np.tile(ext1d_boxcar_bkg, (image.shape[0], 1))
        else:
            data_bkgsub = image

        # extract. Note that, for a cube, this is arbitrarily picking one of the
        # spatial axis to collapse. This should be handled by the API somehow.
        ext1d = np.sum(data_bkgsub * wimage, axis=crossdisp_axis)
        ext1d *= pixelarea

        # TODO: used to return a Spectrum object but now just returns a 1D and 2D array
        return (ext1d, data_bkgsub)

    def get_checkplot(self):
        trace_line = self.last_trace.line

        fig = plt.figure()
        plt.imshow(self.last_image, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(self.last_image, (5, 98)))

        plt.plot(np.arange(len(trace_line)), trace_line, c='C0')
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line + self.width,
            trace_line - self.width,
            color='C0',
            alpha=0.5
        )
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line + self.width + self.bkg_offset,
            trace_line + self.width + self.bkg_offset + self.bkg_width,
            color='C1',
            alpha=0.5
        )
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line - self.width - self.bkg_offset,
            trace_line - self.width - self.bkg_offset - self.bkg_width,
            color='C1',
            alpha=0.5
        )
        plt.ylim(
            np.min(
                trace_line - (self.width + self.bkg_offset + self.bkg_width) * 2
            ),
            np.max(
                trace_line + (self.width + self.bkg_offset + self.bkg_width) * 2
            )
        )

        return fig
