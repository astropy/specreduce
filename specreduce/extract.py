# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass

import numpy as np

from astropy import units as u
from astropy.modeling import models, fitting
from astropy.nddata import StdDevUncertainty

from specreduce.core import SpecreduceOperation
from specreduce.tracing import FlatTrace
from specutils import Spectrum1D

__all__ = ['BoxcarExtract', 'HorneExtract', 'OptimalExtract']


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


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne extraction (a.k.a. optimal) on a region in a
    two-dimensional spectrum.
    """

    def __call__(self, image, variance, trace_object,
                 disp_axis=1, crossdisp_axis=0,
                 bkgrd_prof=models.Polynomial1D(2)):
        """
        Run the Horne calculation on a region of an image and extract a
        1D spectrum.

        Parameters
        ----------

        image : `~astropy.nddata.CCDData` or array-like, required
            The input 2D spectrum from which to extract a source.

        variance : `~astropy.nddata.CCDData` or array-like, required
            The associated variances for each pixel in the image.
            Should have the same dimensions as `image`.

        trace_object : `~specreduce.tracing.Trace`, required
            The associated 1D trace object created for the 2D image.

        disp_axis : int, optional
            The index of the image's dispersion axis. [default: 1]

        crossdisp_axis : int, optional
            The index of the image's cross-dispersion axis. [default: 0]

        bkgrd_prof : `~astropy.modeling.Model`
            A model for the image's background flux.
            [default: models.Polynomial1D(2)]

        Returns
        -------
        spec_1d : `~specutils.Spectrum1D`
            The final, Horne extracted 1D spectrum.
        """
        # co-add signal in each column
        ncols = image.shape[crossdisp_axis]
        xd_pixels = np.arange(ncols) # y plot dir / x spec dir
        coadd = image.sum(axis=disp_axis) / ncols

        # fit source profile, using Gaussian model as a template
        # NOTE: could add argument for users to provide their own model
        gauss_prof = models.Gaussian1D(amplitude=coadd.max(),
                                       mean=coadd.argmax(), stddev=2)

        # Fit extraction kernel to column with combined gaussian/bkgrd model
        ext_prof = gauss_prof + bkgrd_prof
        fitter = fitting.LevMarLSQFitter()
        fit_ext_kernel = fitter(ext_prof, xd_pixels, coadd)

        # generate 1D spectrum, column by column
        extracted = []
        for col_pix in range(image.shape[disp_axis]):
            # set gaussian model's mean as column's corresponding trace value
            fit_ext_kernel.mean_0 = trace_object.trace[col_pix]
            # kernel_col.stddev_0 = self.fwhm_fit(x)
            # NOTE: support for variable FWHMs forthcoming

            # fit compound model to column
            kernel_vals = fit_ext_kernel(xd_pixels)
            # NOTE: support user-provided kernels? would require different handling
            # NOTE: allow smoothing in x direction via boxcar/x axis collapse/etc?

            # fetch matching columns from original and variance images
            image_col = image[:, col_pix]
            variance_col = variance[:, col_pix]

            # calculate kernel normalization
            g_x = np.ma.sum(kernel_vals**2 / variance_col)
            if np.ma.is_masked(g_x):
                continue

            # sum by column weights
            weighted_col = np.ma.divide(image_col * kernel_vals, variance_col)
            result = np.ma.sum(weighted_col) / g_x

            # multiply kernel normalization into the extracted signal
            result *= (fit_ext_kernel.amplitude_0
                       * fit_ext_kernel.stddev_0 * np.sqrt(2*np.pi))
            extracted.append(result)

        # convert the extraction to a Spectrum1D object
        pixels = np.arange(image.shape[disp_axis]) * u.pix
        spec_1d = Spectrum1D(spectral_axis=pixels,
                             flux=extracted * getattr(image, 'unit', u.DN))

        return spec_1d


@dataclass
class OptimalExtract(HorneExtract):
    """
    Perform a Horne extraction (a.k.a. optimal) on a region in a
    two-dimensional spectrum.
    """
    pass
