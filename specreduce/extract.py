# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np

from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import NDData, VarianceUncertainty

from specreduce.core import _ImageParser, SpecreduceOperation
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
    weights = np.zeros((npix))
    if hwidth == 0:
        # the logic below would return all zeros anyways, so might as well save the time
        # (negative widths should be avoided by earlier logic!)
        return weights

    if center-hwidth > npix-0.5 or center+hwidth < -0.5:
        # entire window is out-of-bounds
        return weights

    lower_edge = max(-0.5, center-hwidth)  # where -0.5 is lower bound of the image
    upper_edge = min(center+hwidth, npix-0.5)  # where npix-0.5 is upper bound of the image

    # let's avoid recomputing the round repeatedly
    int_round_lower_edge = int(round(lower_edge))
    int_round_upper_edge = int(round(upper_edge))

    # inner pixels that get full weight
    # the round in conjunction with the +1 handles the half-pixel "offset",
    # the upper bound doesn't have the +1 because array slicing is inclusive on the lower index and
    # exclusive on the upper-index
    # NOTE: round(-0.5) == 0, which is helpful here for the case where lower_edge == -0.5
    weights[int_round_lower_edge+1:int_round_upper_edge] = 1

    # handle edge pixels (for cases where an edge pixel is fully-weighted, this will set it again,
    # but should still compute a weight of 1.  By using N:N+1, we avoid index errors if the edge
    # is outside the image bounds.  But we do need to avoid negative indices which would count
    # from the end of the array.
    if int_round_lower_edge >= 0:
        weights[int_round_lower_edge:int_round_lower_edge+1] = round(lower_edge) + 0.5 - lower_edge
    weights[int_round_upper_edge:int_round_upper_edge+1] = upper_edge - (round(upper_edge) - 0.5)

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


def _to_spectrum1d_pixels(fluxes):
    # TODO: add wavelength units, uncertainty and mask to spectrum1D object
    return Spectrum1D(spectral_axis=np.arange(len(fluxes)) * u.pixel,
                      flux=fluxes)


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
    image : `~astropy.nddata.NDData`-like or array-like, required
        image with 2-D spectral image data
    trace_object : Trace, required
        trace object
    width : float, optional
        width of extraction aperture in pixels
    disp_axis : int, optional
        dispersion axis
    crossdisp_axis : int, optional
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
        image : `~astropy.nddata.NDData`-like or array-like, required
            image with 2-D spectral image data
        trace_object : Trace, required
            trace object
        width : float, optional
            width of extraction aperture in pixels [default: 5]
        disp_axis : int, optional
            dispersion axis [default: 1]
        crossdisp_axis : int, optional
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

        # handle image processing based on its type
        self.image = self._parse_image(image)

        # TODO: this check can be removed if/when implemented as a check in FlatTrace
        if isinstance(trace_object, FlatTrace):
            if trace_object.trace_pos < 1:
                raise ValueError('trace_object.trace_pos must be >= 1')

        if width < 0:
            raise ValueError("width must be positive")

        # weight image to use for extraction
        wimg = _ap_weight_image(
            trace_object,
            width,
            disp_axis,
            crossdisp_axis,
            self.image.shape)

        # extract
        ext1d = np.sum(self.image.data * wimg, axis=crossdisp_axis)
        return _to_spectrum1d_pixels(ext1d * self.image.unit)


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    Parameters
    ----------

    image : `~astropy.nddata.NDData`-like or array-like, required
        The input 2D spectrum from which to extract a source. An
        NDData object must specify uncertainty and a mask. An array
        requires use of the ``variance``, ``mask``, & ``unit`` arguments.

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
        (Only used if ``image`` is not an NDData object.)
        The associated variances for each pixel in the image. Must
        have the same dimensions as ``image``. If all zeros, the variance
        will be ignored and treated as all ones.  If any zeros, those
        elements will be excluded via masking.  If any negative values,
        an error will be raised. [default: None]

    mask : `~numpy.ndarray`, optional
        (Only used if ``image`` is not an NDData object.)
        Whether to mask each pixel in the image. Must have the same
        dimensions as ``image``. If blank, all non-NaN pixels are
        unmasked. [default: None]

    unit : `~astropy.units.Unit` or str, optional
        (Only used if ``image`` is not an NDData object.)
        The associated unit for the data in ``image``. If blank,
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

    def _parse_image(self, variance=None, mask=None, unit=None):
        """
        Convert all accepted image types to a consistently formatted Spectrum1D.
        Takes some extra arguments exactly as they come from self.__call__() to
        handle cases where users specify them as arguments instead of as
        attributes of their image object.
        """

        if isinstance(self.image, np.ndarray):
            img = self.image
        elif isinstance(self.image, u.quantity.Quantity):
            img = self.image.value
        else:  # NDData, including CCDData and Spectrum1D
            img = self.image.data

        # mask is set as None when not specified upon creating a Spectrum1D
        # object, so we must check whether it is absent *and* whether it's
        # present but set as None
        if getattr(self.image, 'mask', None) is not None:
            mask = self.image.mask
        else:
            mask = np.ma.masked_invalid(img).mask

        # Process uncertainties, converting to variances when able and throwing
        # an error when uncertainties are missing or less easily converted
        if (hasattr(self.image, 'uncertainty')
                and self.image.uncertainty is not None):
            if self.image.uncertainty.uncertainty_type == 'var':
                variance = self.image.uncertainty.array
            elif self.image.uncertainty.uncertainty_type == 'std':
                warnings.warn("image NDData object's uncertainty "
                              "interpreted as standard deviation. if "
                              "incorrect, use VarianceUncertainty when "
                              "assigning image object's uncertainty.")
                variance = self.image.uncertainty.array**2
            elif self.image.uncertainty.uncertainty_type == 'ivar':
                variance = 1 / self.image.uncertainty.array
            else:
                # other options are InverseVariance and UnknownVariance
                raise ValueError("image NDData object has unexpected "
                                 "uncertainty type. instead, try "
                                 "VarianceUncertainty or StdDevUncertainty.")
        elif (hasattr(self.image, 'uncertainty')
              and self.image.uncertainty is None):
            # ignore variance arg to focus on updating NDData object
            raise ValueError('image NDData object lacks uncertainty')
        else:
            if variance is None:
                raise ValueError("if image is a numpy or Quantity array, a "
                                 "variance must be specified. consider "
                                 "wrapping it into one object by instead "
                                 "passing an NDData image.")
            elif self.image.shape != variance.shape:
                raise ValueError("image and variance shapes must match")

        if np.any(variance < 0):
            raise ValueError("variance must be fully positive")
        if np.all(variance == 0):
            # technically would result in infinities, but since they're all
            # zeros, we can override ones to simulate an unweighted case
            variance = np.ones_like(variance)
        if np.any(variance == 0):
            # exclude such elements by editing the input mask
            mask[variance == 0] = True
            # replace the variances to avoid a divide by zero warning
            variance[variance == 0] = np.nan

        variance = VarianceUncertainty(variance)

        unit = getattr(self.image, 'unit',
                       u.Unit(self.unit) if self.unit is not None else u.Unit())

        spectral_axis = getattr(self.image, 'spectral_axis',
                                (np.arange(img.shape[self.disp_axis])
                                 if hasattr(self, 'disp_axis')
                                 else np.arange(img.shape[1])) * u.pix)

        self.image = Spectrum1D(img * unit, spectral_axis=spectral_axis,
                                uncertainty=variance, mask=mask)

    def __call__(self, image=None, trace_object=None,
                 disp_axis=None, crossdisp_axis=None,
                 bkgrd_prof=None,
                 variance=None, mask=None, unit=None):
        """
        Run the Horne calculation on a region of an image and extract a
        1D spectrum.

        Parameters
        ----------

        image : `~astropy.nddata.NDData`-like or array-like, required
            The input 2D spectrum from which to extract a source. An
            NDData object must specify uncertainty and a mask. An array
            requires use of the ``variance``, ``mask``, & ``unit`` arguments.

        trace_object : `~specreduce.tracing.Trace`, required
            The associated 1D trace object created for the 2D image.

        disp_axis : int, optional
            The index of the image's dispersion axis.

        crossdisp_axis : int, optional
            The index of the image's cross-dispersion axis.

        bkgrd_prof : `~astropy.modeling.Model`, optional
            A model for the image's background flux.

        variance : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as ``image``. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.

        mask : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as ``image``. If blank, all non-NaN pixels are
            unmasked.

        unit : `~astropy.units.Unit` or str, optional
            (Only used if ``image`` is not an NDData object.)
            The associated unit for the data in ``image``. If blank,
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

        # parse image and replace optional arguments with updated values
        self._parse_image(variance, mask, unit)
        variance = self.image.uncertainty.array
        unit = self.image.unit

        # mask any previously uncaught invalid values
        or_mask = np.logical_or(mask,
                                np.ma.masked_invalid(self.image.data).mask)
        img = np.ma.masked_array(self.image.data, or_mask)
        mask = img.mask

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
        return _to_spectrum1d_pixels(extraction * unit)


@dataclass
class OptimalExtract(HorneExtract):
    """
    An alias for `HorneExtract`.
    """
    __doc__ += HorneExtract.__doc__
    pass
