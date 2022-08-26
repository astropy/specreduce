# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np

from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import NDData

from specreduce.core import SpecreduceOperation
from specreduce.tracing import Trace, FlatTrace
from specutils import Spectrum1D

__all__ = ['BoxcarExtract', 'HorneExtract', 'OptimalExtract']


def _get_boxcar_weights(center, hwidth, npix):
    """
    Compute weights given an aperture center, half width,
    and number of pixels.

    Based on `get_boxcar_weights()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    center : float, required
        The index of the aperture's center pixel on the larger image's
        cross-dispersion axis.

    hwidth : float, required
        Half of the aperture's width in the cross-dispersion direction.

    npix : float, required
        The number of pixels in the larger image's cross-dispersion
        axis.

    Returns
    -------
    weights : `~numpy.ndarray`
        A 2D image with weights assigned to pixels that fall within the
        defined aperture.
    """
    weights = np.zeros(npix)

    # shift center from integer to pixel space, where pixel N is [N-0.5, N+0.5),
    # not [N, N+1). a pixel's integer index corresponds to its middle, not edge
    center += 0.5

    # pixels given full weight because they sit entirely within the aperture
    fullpixels = [max(0, int(np.ceil(center - hwidth))),
                  min(int(np.floor(center + hwidth)), npix)]
    weights[fullpixels[0]:fullpixels[1]] = 1

    # pixels at the edges of the boxcar with partial weight, if any
    if fullpixels[0] > 0:
        w0 = hwidth - (center - fullpixels[0])
        if w0 >= 0:
            weights[fullpixels[0] - 1] = w0
    if fullpixels[1] < npix:
        w1 = hwidth - (fullpixels[1] - center)
        if w1 >= 0:
            weights[fullpixels[1]] = w1

    return weights


def _ap_weight_image(trace, width, disp_axis, crossdisp_axis, image_shape):

    """
    Create a weight image that defines the desired extraction aperture.

    Based on `ap_weight_images()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    trace : `~specreduce.tracing.Trace`, required
        trace object
    width : float, required
        width of extraction aperture in pixels
    disp_axis : int, required
        dispersion axis
    crossdisp_axis : int, required
        cross-dispersion axis
    image_shape : tuple with 2 elements, required
        size (shape) of image

    Returns
    -------
    wimage : `~numpy.ndarray`
        a 2D weight image defining the aperture
    """
    wimage = np.zeros(image_shape)
    hwidth = 0.5 * width
    image_sizes = image_shape[crossdisp_axis]

    # loop in dispersion direction and compute weights.
    for i in range(image_shape[disp_axis]):
        # TODO trace must handle transposed data (disp_axis == 0)
        # pass trace.trace.data[i] to avoid any mask if part of the regions is out-of-bounds
        wimage[:, i] = _get_boxcar_weights(trace.trace.data[i], hwidth, image_sizes)

    return wimage


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        extract = BoxcarExtract(image, trace)
        spectrum = extract(width=width)


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
    image: NDData
    trace_object: Trace
    width: float = 5
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def __call__(self, image=None, trace_object=None, width=None,
                 disp_axis=None, crossdisp_axis=None):
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image : nddata-compatible image
            image with 2-D spectral image data
        trace_object : Trace
            trace object
        width : float
            width of extraction aperture in pixels [default: 5]
        disp_axis : int
            dispersion axis [default: 1]
        crossdisp_axis : int
            cross-dispersion axis [default: 0]


        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        image = image if image is not None else self.image
        trace_object = trace_object if trace_object is not None else self.trace_object
        width = width if width is not None else self.width
        disp_axis = disp_axis if disp_axis is not None else self.disp_axis
        crossdisp_axis = crossdisp_axis if crossdisp_axis is not None else self.crossdisp_axis

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
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    Parameters
    ----------

    image : `~astropy.nddata.NDData` or array-like, required
        The input 2D spectrum from which to extract a source. An
        NDData object must specify uncertainty and a mask. An array
        requires use of the `variance`, `mask`, & `unit` arguments.

    trace_object : `~specreduce.tracing.Trace`, required
        The associated 1D trace object created for the 2D image.

    disp_axis : int, optional
        The index of the image's dispersion axis. [default: 1]

    crossdisp_axis : int, optional
        The index of the image's cross-dispersion axis. [default: 0]

    bkgrd_prof : `~astropy.modeling.Model`, optional
        A model for the image's background flux.
        [default: models.Polynomial1D(2)]

    variance : `~numpy.ndarray`, optional
        (Only used if `image` is not an NDData object.)
        The associated variances for each pixel in the image. Must
        have the same dimensions as `image`. If all zeros, the variance
        will be ignored and treated as all ones.  If any zeros, those
        elements will be excluded via masking.  If any negative values,
        an error will be raised. [default: None]

    mask : `~numpy.ndarray`, optional
        (Only used if `image` is not an NDData object.)
        Whether to mask each pixel in the image. Must have the same
        dimensions as `image`. If blank, all non-NaN pixels are
        unmasked. [default: None]

    unit : `~astropy.units.core.Unit` or str, optional
        (Only used if `image` is not an NDData object.)
        The associated unit for the data in `image`. If blank,
        fluxes are interpreted as unitless. [default: None]

    """
    image: NDData
    trace_object: Trace
    bkgrd_prof: Model = field(default=models.Polynomial1D(2))
    variance: np.ndarray = field(default=None)
    mask: np.ndarray = field(default=None)
    unit: np.ndarray = field(default=None)
    disp_axis: int = 1
    crossdisp_axis: int = 0
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def __call__(self, image=None, trace_object=None,
                 disp_axis=None, crossdisp_axis=None,
                 bkgrd_prof=None,
                 variance=None, mask=None, unit=None):
        """
        Run the Horne calculation on a region of an image and extract a
        1D spectrum.

        Parameters
        ----------

        image : `~astropy.nddata.NDData` or array-like, required
            The input 2D spectrum from which to extract a source. An
            NDData object must specify uncertainty and a mask. An array
            requires use of the `variance`, `mask`, & `unit` arguments.

        trace_object : `~specreduce.tracing.Trace`, required
            The associated 1D trace object created for the 2D image.

        disp_axis : int, optional
            The index of the image's dispersion axis.

        crossdisp_axis : int, optional
            The index of the image's cross-dispersion axis.

        bkgrd_prof : `~astropy.modeling.Model`, optional
            A model for the image's background flux.

        variance : `~numpy.ndarray`, optional
            (Only used if `image` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as `image`. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.

        mask : `~numpy.ndarray`, optional
            (Only used if `image` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as `image`. If blank, all non-NaN pixels are
            unmasked.

        unit : `~astropy.units.core.Unit` or str, optional
            (Only used if `image` is not an NDData object.)
            The associated unit for the data in `image`. If blank,
            fluxes are interpreted as unitless.


        Returns
        -------
        spec_1d : `~specutils.Spectrum1D`
            The final, Horne extracted 1D spectrum.
        """
        image = image if image is not None else self.image
        trace_object = trace_object if trace_object is not None else self.trace_object
        disp_axis = disp_axis if disp_axis is not None else self.disp_axis
        crossdisp_axis = crossdisp_axis if crossdisp_axis is not None else self.crossdisp_axis
        bkgrd_prof = bkgrd_prof if bkgrd_prof is not None else self.bkgrd_prof
        variance = variance if variance is not None else self.variance
        mask = mask if mask is not None else self.mask
        unit = unit if unit is not None else self.unit

        # handle image and associated data based on image's type
        if isinstance(image, NDData):
            img = np.ma.array(image.data, mask=image.mask)
            unit = image.unit if image.unit is not None else u.Unit()

            if image.uncertainty is not None:
                # prioritize NDData's uncertainty over variance argument
                if image.uncertainty.uncertainty_type == 'var':
                    variance = image.uncertainty.array
                elif image.uncertainty.uncertainty_type == 'std':
                    # NOTE: CCDData defaults uncertainties given as pure arrays
                    # to std and logs a warning saying so upon object creation.
                    # should we remind users again here?
                    warnings.warn("image NDData object's uncertainty "
                                  "interpreted as standard deviation. if "
                                  "incorrect, use VarianceUncertainty when "
                                  "assigning image object's uncertainty.")
                    variance = image.uncertainty.array**2
                elif image.uncertainty.uncertainty_type == 'ivar':
                    variance = 1 / image.uncertainty.array
                else:
                    # other options are InverseVariance and UnknownVariance
                    raise ValueError("image NDData object has unexpected "
                                     "uncertainty type. instead, try "
                                     "VarianceUncertainty or StdDevUncertainty.")
            else:
                # ignore variance arg to focus on updating NDData object
                raise ValueError('image NDData object lacks uncertainty')

        else:
            if variance is None:
                raise ValueError('if image is a numpy array, a variance must '
                                 'be specified. consider wrapping it into one '
                                 'object by instead passing an NDData image.')
            elif image.shape != variance.shape:
                raise ValueError('image and variance shapes must match')

            # check optional arguments, filling them in if absent
            if mask is None:
                mask = np.ma.masked_invalid(image).mask
            elif image.shape != mask.shape:
                raise ValueError('image and mask shapes must match.')

            if isinstance(unit, str):
                unit = u.Unit(unit)
            else:
                unit = unit if unit is not None else u.Unit()

            # create image
            img = np.ma.array(image, mask=mask)

        if np.all(variance == 0):
            # technically would result in infinities, but since they're all zeros
            # we can just do the unweighted case by overriding with all ones
            variance = np.ones_like(variance)

        if np.any(variance < 0):
            raise ValueError("variance must be fully positive")

        if np.any(variance == 0):
            # exclude these elements by editing the input mask
            img.mask[variance == 0] = True
            # replace the variances to avoid a divide by zero warning
            variance[variance == 0] = np.nan

        # co-add signal in each image column
        ncols = img.shape[crossdisp_axis]
        xd_pixels = np.arange(ncols)  # y plot dir / x spec dir
        coadd = img.sum(axis=disp_axis) / ncols

        # fit source profile, using Gaussian model as a template
        # NOTE: could add argument for users to provide their own model
        gauss_prof = models.Gaussian1D(amplitude=coadd.max(),
                                       mean=coadd.argmax(), stddev=2)

        # Fit extraction kernel to column with combined gaussian/bkgrd model
        ext_prof = gauss_prof + bkgrd_prof
        fitter = fitting.LevMarLSQFitter()
        fit_ext_kernel = fitter(ext_prof, xd_pixels, coadd)

        # use compound model to fit a kernel to each image column
        # NOTE: infers Gaussian1D source profile; needs generalization for others
        kernel_vals = []
        norms = []
        for col_pix in range(img.shape[disp_axis]):
            # set gaussian model's mean as column's corresponding trace value
            fit_ext_kernel.mean_0 = trace_object.trace[col_pix]
            # NOTE: support for variable FWHMs forthcoming and would be here

            # fit compound model to column
            fitted_col = fit_ext_kernel(xd_pixels)

            # save result and normalization
            kernel_vals.append(fitted_col)
            norms.append(fit_ext_kernel.amplitude_0
                         * fit_ext_kernel.stddev_0 * np.sqrt(2*np.pi))

        # transform fit-specific information
        kernel_vals = np.array(kernel_vals).T
        norms = np.array(norms)

        # calculate kernel normalization, masking NaNs
        g_x = np.ma.sum(kernel_vals**2 / variance, axis=crossdisp_axis)

        # sum by column weights
        weighted_img = np.ma.divide(img * kernel_vals, variance)
        result = np.ma.sum(weighted_img, axis=crossdisp_axis) / g_x

        # multiply kernel normalization into the extracted signal
        extraction = result * norms

        # convert the extraction to a Spectrum1D object
        pixels = np.arange(img.shape[disp_axis]) * u.pix
        spec_1d = Spectrum1D(spectral_axis=pixels, flux=extraction * unit)

        return spec_1d


@dataclass
class OptimalExtract(HorneExtract):
    """
    An alias for `HorneExtract`.
    """
    __doc__ += HorneExtract.__doc__
    pass
