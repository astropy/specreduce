# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.nddata import VarianceUncertainty
from astropy.utils.decorators import deprecated_attribute

from specreduce.compat import SPECUTILS_LT_2, Spectrum
from specreduce.core import _ImageParser, MaskingOption, ImageLike
from specreduce.extract import _ap_weight_image
from specreduce.tracing import Trace, FlatTrace

__all__ = ["Background"]


@dataclass
class Background(_ImageParser):
    """
    Determine the background from an image for subtraction.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        bg = Background.two_sided(image, trace, bkg_sep, width=bkg_width)
        subtracted_image = image - bg

    Parameters
    ----------
    image
        Image with 2-D spectral image data
    traces : List, `specreduce.tracing.Trace`, int, float
        Individual or list of trace object(s) (or integers/floats to define
        FlatTraces) to extract the background. If None, a ``FlatTrace`` at the
        center of the image (according to ``disp_axis``) will be used.
    width
        Width of extraction aperture in pixels.
    statistic
        Statistic to use when computing the background.  ``average`` will
        account for partial pixel weights, ``median`` will include all partial
        pixels.
    disp_axis
        Dispersion axis.
    crossdisp_axis
        Cross-dispersion axis.
    mask_treatment
        Specifies how to handle masked or non-finite values in the input image.
        The accepted values are:

        - ``apply``: The image remains unchanged, and any existing mask is combined\
            with a mask derived from non-finite values.
        - ``ignore``: The image remains unchanged, and any existing mask is dropped.
        - ``propagate``: The image remains unchanged, and any masked or non-finite pixel\
            causes the mask to extend across the entire cross-dispersion axis.
        - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,\
            and the mask is dropped.
        - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,\
            and the mask is dropped.
        - ``apply_mask_only``: The  image and mask are left unmodified.
        - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a\
            new mask is created based on non-finite values.

"""

    # required so numpy won't call __rsub__ on individual elements
    # https://stackoverflow.com/a/58409215
    __array_ufunc__ = None

    image: ImageLike
    traces: list = field(default_factory=list)
    width: float = 5
    statistic: str = "average"
    disp_axis: int = 1
    crossdisp_axis: int = 0
    mask_treatment: MaskingOption = "apply"
    _valid_mask_treatment_methods = (
        "apply",
        "ignore",
        "propagate",
        "zero_fill",
        "nan_fill",
        "apply_mask_only",
        "apply_nan_only",
    )

    # TO-DO: update bkg_array with Spectrum alternative (is bkg_image enough?)
    bkg_array = deprecated_attribute("bkg_array", "1.3")

    def __post_init__(self):
        self.image = self._parse_image(
            self.image, disp_axis=self.disp_axis, mask_treatment=self.mask_treatment
        )

        # always work with masked array, even if there is no masked
        # or non-finite data, in case padding is needed. if not, mask will be
        # dropped at the end and a regular array will be returned.
        img = np.ma.masked_array(self.image.data, self.image.mask)

        if self.width < 0:
            raise ValueError("width must be positive")
        if self.width == 0:
            self._bkg_array = np.zeros(self.image.shape[self.disp_axis])
            self._bkg_variance = np.zeros(self.image.shape[self.disp_axis])
            self._variance_unit = self.image.unit**2
            self._orig_uncty_type = (
                type(self.image.uncertainty) if self.image.uncertainty is not None
                else VarianceUncertainty
            )
            return

        self._set_traces()

        bkg_wimage = np.zeros_like(self.image.data, dtype=np.float64)
        for trace in self.traces:
            # note: ArrayTrace can have masked values, but if it does a MaskedArray
            # will be returned so this should be reflected in the window size here
            # (i.e, np.nanmax is not required.)
            windows_max = trace.trace.data.max() + self.width / 2
            windows_min = trace.trace.data.min() - self.width / 2

            if windows_max > self.image.shape[self.crossdisp_axis]:
                warnings.warn(
                    "background window extends beyond image boundaries "
                    + f"({windows_max} >= {self.image.shape[self.crossdisp_axis]})"
                )
            if windows_min < 0:
                warnings.warn(
                    "background window extends beyond image boundaries " + f"({windows_min} < 0)"
                )

            # pass trace.trace.data to ignore any mask on the trace
            bkg_wimage += _ap_weight_image(
                trace, self.width, self.disp_axis, self.crossdisp_axis, self.image.shape
            )

        if np.any(bkg_wimage > 1):
            raise ValueError("background regions overlapped")
        if np.any(np.sum(bkg_wimage, axis=self.crossdisp_axis) == 0):
            raise ValueError(
                "background window does not remain in bounds across entire dispersion axis"
            )  # noqa
        # check if image contained within background window is fully-nonfinite and raise an error
        if np.all(img.mask[bkg_wimage > 0]):
            raise ValueError(
                "Image is fully masked within background window determined by `width`."
            )  # noqa

        if self.statistic == "median":
            # make it clear in the expose image that partial pixels are fully-weighted
            bkg_wimage[bkg_wimage > 0] = 1

        self.bkg_wimage = bkg_wimage

        if self.statistic == "average":
            self._bkg_array = np.ma.average(img, weights=self.bkg_wimage, axis=self.crossdisp_axis)

        elif self.statistic == "median":
            # combine where background weight image is 0 with image masked (which already
            # accounts for non-finite data that wasn't already masked)
            img.mask = np.logical_or(self.bkg_wimage == 0, self.image.mask)
            self._bkg_array = np.ma.median(img, axis=self.crossdisp_axis)
        else:
            raise ValueError("statistic must be 'average' or 'median'")

        # Compute background variance for uncertainty propagation
        self._compute_bkg_variance(img)

    def _set_traces(self):
        """Determine `traces` from input. If an integer/float or list if int/float
        is passed in, use these to construct FlatTrace objects. These values
        must be positive. If None (which is initialized to an empty list),
        construct a FlatTrace using the center of image (according to disp.
        axis). Otherwise, any Trace object or list of Trace objects can be
        passed in."""

        if self.traces == []:
            # assume a flat trace at the image center if nothing is passed in.
            trace_pos = self.image.shape[self.crossdisp_axis] / 2.0
            self.traces = [FlatTrace(self.image, trace_pos)]

        if isinstance(self.traces, Trace):
            # if just one trace, turn it into iterable.
            self.traces = [self.traces]
            return

        # finally, if float/int is passed in convert to FlatTrace(s)
        if isinstance(self.traces, (float, int)):  # for a single number
            self.traces = [self.traces]

        if np.all([isinstance(x, (float, int)) for x in self.traces]):
            self.traces = [FlatTrace(self.image, trace_pos) for trace_pos in self.traces]
            return

        else:
            if not np.all([isinstance(x, Trace) for x in self.traces]):
                raise ValueError(
                    "`traces` must be a `Trace` object or list of "
                    "`Trace` objects, a number or list of numbers to "
                    "define FlatTraces, or None to use a FlatTrace in "
                    "the middle of the image."
                )

    def _compute_bkg_variance(self, img):
        """Compute background variance for uncertainty propagation.

        Parameters
        ----------
        img : np.ma.MaskedArray
            The masked image array used for background computation.
        """
        # Get input variance (estimate from flux if not provided)
        self._variance_unit = self.image.unit**2
        if self.image.uncertainty is not None:
            self._orig_uncty_type = type(self.image.uncertainty)
            if self._orig_uncty_type == VarianceUncertainty:
                var = self.image.uncertainty.array
            else:
                var = self.image.uncertainty.represent_as(VarianceUncertainty).array
        else:
            # Estimate variance from flux values in background region
            self._orig_uncty_type = VarianceUncertainty
            bkg_mask = self.bkg_wimage > 0
            valid_mask = bkg_mask & ~img.mask

            # Compute sample variance along cross-dispersion axis for each column
            masked_flux = np.ma.masked_array(self.image.data, ~valid_mask)
            sample_variance = np.ma.var(masked_flux, axis=self.crossdisp_axis)

            # Tile to full image shape (same variance for all pixels in a column)
            var = np.tile(sample_variance.data, (self.image.shape[0], 1))

        # Create masked variance array matching the image mask
        masked_variance = np.ma.masked_array(var, img.mask)

        if self.statistic == "average":
            # Var(weighted_mean) = sum(w^2 * var) / (sum w)^2
            weights_squared = self.bkg_wimage**2
            numerator = np.ma.sum(weights_squared * masked_variance, axis=self.crossdisp_axis)
            denominator = np.ma.sum(self.bkg_wimage, axis=self.crossdisp_axis) ** 2
            self._bkg_variance = (numerator / denominator).data

        elif self.statistic == "median":
            # Var(median) ~ (pi/2) * var / n for approximately normal data
            # Use mean variance of included pixels
            bkg_mask = self.bkg_wimage > 0
            valid_mask = bkg_mask & ~img.mask
            n_pixels = np.sum(valid_mask, axis=self.crossdisp_axis)
            variance_sum = np.sum(np.where(valid_mask, var, 0), axis=self.crossdisp_axis)
            mean_variance = variance_sum / n_pixels
            self._bkg_variance = (np.pi / 2) * mean_variance / n_pixels

    @classmethod
    def two_sided(cls, image, trace_object, separation, **kwargs):
        """
        Determine the background from an image for subtraction centered around
        an input trace.


        Example: ::

            trace = FitTrace(image, guess=trace_pos)
            bg = Background.two_sided(image, trace, bkg_sep, width=bkg_width)

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like
            Image with 2-D spectral image data. Assumes cross-dispersion
            (spatial) direction is axis 0 and dispersion (wavelength)
            direction is axis 1.
        trace_object : `~specreduce.tracing.Trace`
            estimated trace of the spectrum to center the background traces
        separation : float
            separation from ``trace_object`` for the background regions
        width : float
            width of each background aperture in pixels
        statistic : string
            statistic to use when computing the background.  'average' will
            account for partial pixel weights, 'median' will include all partial
            pixels.
        disp_axis : int
            dispersion axis
        crossdisp_axis : int
            cross-dispersion axis
        mask_treatment : string
            Specifies how to handle masked or non-finite values in the input image.
            The accepted values are:

            - ``apply``: The image remains unchanged, and any existing mask is combined\
                with a mask derived from non-finite values.
            - ``ignore``: The image remains unchanged, and any existing mask is dropped.
            - ``propagate``: The image remains unchanged, and any masked or non-finite pixel\
                causes the mask to extend across the entire cross-dispersion axis.
            - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,\
                and the mask is dropped.
            - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,\
                and the mask is dropped.
            - ``apply_mask_only``: The  image and mask are left unmodified.
            - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a\
                new mask is created based on non-finite values.
        """

        image = _ImageParser._get_data_from_image(image) if image is not None else cls.image
        kwargs["traces"] = [trace_object - separation, trace_object + separation]
        return cls(image=image, **kwargs)

    @classmethod
    def one_sided(cls, image, trace_object, separation, **kwargs):
        """
        Determine the background from an image for subtraction above
        or below an input trace.

        Example: ::

            trace = FitTrace(image, guess=trace_pos)
            bg = Background.one_sided(image, trace, bkg_sep, width=bkg_width)

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like
            Image with 2-D spectral image data. Assumes cross-dispersion
            (spatial) direction is axis 0 and dispersion (wavelength)
            direction is axis 1.
        trace_object : `~specreduce.tracing.Trace`
            Estimated trace of the spectrum to center the background traces
        separation : float
            Separation from ``trace_object`` for the background, positive will be
            above the trace, negative below.
        width : float
            Width of each background aperture in pixels
        statistic : string
            Statistic to use when computing the background.  'average' will
            account for partial pixel weights, 'median' will include all partial
            pixels.
        disp_axis : int
            Dispersion axis
        crossdisp_axis : int
            Cross-dispersion axis
        mask_treatment : string
            Specifies how to handle masked or non-finite values in the input image.
            The accepted values are:

            - ``apply``: The image remains unchanged, and any existing mask is combined\
                with a mask derived from non-finite values.
            - ``ignore``: The image remains unchanged, and any existing mask is dropped.
            - ``propagate``: The image remains unchanged, and any masked or non-finite pixel\
                causes the mask to extend across the entire cross-dispersion axis.
            - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,\
                and the mask is dropped.
            - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,\
                and the mask is dropped.
            - ``apply_mask_only``: The  image and mask are left unmodified.
            - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a\
                new mask is created based on non-finite values.
        """
        image = _ImageParser._get_data_from_image(image) if image is not None else cls.image
        kwargs["traces"] = [trace_object + separation]
        return cls(image=image, **kwargs)

    def bkg_image(self, image=None) -> Spectrum:
        """
        Expose the background tiled to the dimension of ``image``.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, optional
            Image with 2-D spectral image data. Assumes cross-dispersion
            (spatial) direction is axis 0 and dispersion (wavelength)
            direction is axis 1. If None, will extract the background
            from ``image`` used to initialize the class. [default: None]

        Returns
        -------
        spec : `~specutils.Spectrum`
            Spectrum object with same shape as ``image``, including uncertainty.
        """
        image = self._parse_image(image)
        arr = np.tile(self._bkg_array, (image.shape[0], 1))
        var_arr = np.tile(self._bkg_variance, (image.shape[0], 1))
        uncertainty = VarianceUncertainty(var_arr * self._variance_unit).represent_as(
            self._orig_uncty_type
        )

        if SPECUTILS_LT_2:
            kwargs = {}
        else:
            kwargs = {"spectral_axis_index": arr.ndim - 1}
        return Spectrum(
            arr * image.unit, spectral_axis=image.spectral_axis, uncertainty=uncertainty, **kwargs
        )

    def bkg_spectrum(self, image=None, bkg_statistic=None) -> Spectrum:
        """
        Expose the 1D spectrum of the background.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, optional
            Image with 2D spectral image data. Assumes cross-dispersion
            (spatial) direction is axis 0 and dispersion (wavelength)
            direction is axis 1. If None, will extract the background
            from ``image`` used to initialize the class. [default: None]

        Returns
        -------
        spec : `~specutils.Spectrum`
            The background 1D spectrum, with flux and uncertainty expressed
            in the same units as the input image (or DN if none were provided).
        """
        if bkg_statistic is not None:
            warnings.warn(
                "'bkg_statistic' is deprecated and will be removed in a future release. "
                "Please use the 'statistic' argument in the Background initializer instead.",  # noqa
                DeprecationWarning,
            )

        uncertainty = VarianceUncertainty(self._bkg_variance * self._variance_unit).represent_as(
            self._orig_uncty_type
        )
        return Spectrum(
            self._bkg_array * self.image.unit, self.image.spectral_axis, uncertainty=uncertainty
        )

    def sub_image(self, image=None) -> Spectrum:
        """
        Subtract the computed background from image.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        spec : `~specutils.Spectrum`
            Spectrum object with same shape as ``image``.
        """
        image = self._parse_image(image)

        if not SPECUTILS_LT_2:
            return image - self.bkg_image(image)

        # a compare_wcs argument is needed for Spectrum.subtract() in order to
        # avoid a TypeError from SpectralCoord when image's spectral axis is in
        # pixels. it is not needed when image's spectral axis has physical units
        kwargs = {"compare_wcs": None} if image.spectral_axis.unit == u.pix else {}

        # https://docs.astropy.org/en/stable/nddata/mixins/ndarithmetic.html
        return image.subtract(self.bkg_image(image), **kwargs)

    def sub_spectrum(self, image=None) -> Spectrum:
        """
        Expose the 1D spectrum of the background-subtracted image.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-compatible image or None
            image with 2D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        spec : `~specutils.Spectrum`
            The background-subtracted 1D spectrum, with flux and uncertainty
            expressed in the same units as the input image (or u.DN if none
            were provided) and the spectral axis expressed in pixel units.
        """
        sub_img = self.sub_image(image=image)

        try:
            result = sub_img.collapse(np.nansum, axis=self.crossdisp_axis)
        except u.UnitTypeError:
            # can't collapse with a spectral axis in pixels because
            # SpectralCoord only allows frequency/wavelength equivalent units...
            ext1d = np.nansum(sub_img.flux, axis=self.crossdisp_axis)
            result = Spectrum(ext1d, spectral_axis=sub_img.spectral_axis)

        # Propagate uncertainty: Var(sum) = sum of variances
        # Must convert to variance before summing, then convert back to original type
        if sub_img.uncertainty is not None:
            var_uncert = sub_img.uncertainty.represent_as(VarianceUncertainty)
            var_sum = np.nansum(var_uncert.array, axis=self.crossdisp_axis)
            uncertainty = VarianceUncertainty(var_sum * var_uncert.unit).represent_as(
                self._orig_uncty_type
            )
            result = Spectrum(
                result.flux, spectral_axis=result.spectral_axis, uncertainty=uncertainty
            )
        return result

    def __rsub__(self, image):
        """
        Subtract the background from an image.
        """
        return self.sub_image(image)
