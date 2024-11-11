# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field, InitVar

import numpy as np
from astropy.utils.decorators import deprecated_attribute
from specutils import Spectrum1D

from specreduce.extract import _ap_weight_image
from specreduce.tracing import Trace, FlatTrace
from specreduce.image import SRImage, as_image

__all__ = ['Background']


@dataclass
class Background:
    """
    Determine the background from an image for subtraction.


    Example: ::

        trace = FlatTrace(image, trace_pos)
        bg = Background.two_sided(image, trace, bkg_sep, width=bkg_width)
        subtracted_image = image - bg

    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like
        image with 2-D spectral image data
    traces : List, `specreduce.tracing.Trace`, int, float
        Individual or list of trace object(s) (or integers/floats to define
        FlatTraces) to extract the background. If None, a ``FlatTrace`` at the
        center of the image (according to `disp_axis`) will be used.
    width : float
        width of extraction aperture in pixels
    statistic: string
        statistic to use when computing the background.  'average' will
        account for partial pixel weights, 'median' will include all partial
        pixels.
    disp_axis : int
        dispersion axis
        [default: 1]
    crossdisp_axis : int
        cross-dispersion axis
        [default: 0]
    mask_treatment : string, optional
        The method for handling masked or non-finite data. Choice of ``filter``,
        ``omit``, or ``zero-fill``. If ``filter`` is chosen, masked and non-finite
        data will not contribute to the background statistic that is calculated
        in each column along `disp_axis`. If `omit` is chosen, columns along
        disp_axis with any masked/non-finite data values will be fully masked
        (i.e, 2D mask is collapsed to 1D and applied). If ``zero-fill`` is chosen,
        masked/non-finite data will be replaced with 0.0 in the input image,
        and the mask will then be dropped. For all three options, the input mask
        (optional on input NDData object) will be combined with a mask generated
        from any non-finite values in the image data.
        [default: ``filter``]
    """
    # required so numpy won't call __rsub__ on individual elements
    # https://stackoverflow.com/a/58409215
    __array_ufunc__ = None

    image: SRImage
    traces: list = field(default_factory=list)
    width: float = 5.
    disp_axis: InitVar[int] = 1
    crossdisp_axis: InitVar[int | None] = None
    statistic: str = 'average'
    mask_treatment: str = 'filter'
    _valid_mask_treatment_methods = ('filter', 'omit', 'zero-fill')

    # TO-DO: update bkg_array with Spectrum1D alternative (is bkg_image enough?)
    bkg_array = deprecated_attribute('bkg_array', '1.3')

    def __post_init__(self, disp_axis, crossdisp_axis):
        """
        Determine the background from an image for subtraction.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like
            image with 2-D spectral image data
        traces : List
            list of trace objects (or integers to define FlatTraces) to
            extract the background
        width : float
            width of each background aperture in pixels
        statistic: string
            statistic to use when computing the background.  'average' will
            account for partial pixel weights, 'median' will include all partial
            pixels.
        disp_axis : int
            dispersion axis
        crossdisp_axis : int
            cross-dispersion axis
        mask_treatment : string
            The method for handling masked or non-finite data. Choice of `filter`,
            `omit`, or `zero-fill`. If `filter` is chosen, masked/non-finite data
            will be filtered during the fit to each bin/column (along disp. axis) to
            find the peak. If `omit` is chosen, columns along disp_axis with any
            masked/non-finite data values will be fully masked (i.e, 2D mask is
            collapsed to 1D and applied). If `zero-fill` is chosen, masked/non-finite
            data will be replaced with 0.0 in the input image, and the mask will then
            be dropped. For all three options, the input mask (optional on input
            NDData object) will be combined with a mask generated from any non-finite
            values in the image data.
        """
        self.image = as_image(self.image,
                              disp_axis=disp_axis,
                              crossdisp_axis=crossdisp_axis).apply_mask(self.mask_treatment)

        # always work with masked array, even if there is no masked
        # or nonfinite data, in case padding is needed. if not, mask will be
        # dropped at the end and a regular array will be returned.
        img = self.image.to_masked_array()

        if self.width < 0:
            raise ValueError("width must be positive")
        if self.width == 0:
            self._bkg_array = np.zeros(self.image.shape[self.image.disp_axis])
            return

        self._set_traces()

        bkg_wimage = np.zeros_like(self.image.flux, dtype=np.float64)
        for trace in self.traces:
            # note: ArrayTrace can have masked values, but if it does a MaskedArray
            # will be returned so this should be reflected in the window size here
            # (i.e, np.nanmax is not required.)
            windows_max = trace.trace.data.max() + self.width/2
            windows_min = trace.trace.data.min() - self.width/2

            if windows_max > self.image.shape[self.image.crossdisp_axis]:
                warnings.warn("background window extends beyond image boundaries " +
                              f"({windows_max} >= {self.image.shape[self.image.crossdisp_axis]})")
            if windows_min < 0:
                warnings.warn("background window extends beyond image boundaries " +
                              f"({windows_min} < 0)")

            # pass trace.trace.data to ignore any mask on the trace
            bkg_wimage += _ap_weight_image(trace,
                                           self.width,
                                           self.image.disp_axis,
                                           self.image.crossdisp_axis,
                                           self.image.shape)

        if np.any(bkg_wimage > 1):
            raise ValueError("background regions overlapped")
        if np.any(np.sum(bkg_wimage, axis=self.image.crossdisp_axis) == 0):
            raise ValueError("background window does not remain in bounds across entire dispersion axis")  # noqa
        # check if image contained within background window is fully-nonfinite and raise an error
        if np.all(img.mask[bkg_wimage > 0]):
            raise ValueError("Image is fully masked within background window determined by `width`.")  # noqa

        if self.statistic == 'median':
            # make it clear in the expose image that partial pixels are fully-weighted
            bkg_wimage[bkg_wimage > 0] = 1

        self.bkg_wimage = bkg_wimage

        if self.statistic == 'average':
            self._bkg_array = np.ma.average(img,
                                            weights=self.bkg_wimage,
                                            axis=self.image.crossdisp_axis)
        elif self.statistic == 'median':
            # combine where background weight image is 0 with image masked (which already
            # accounts for non-finite data that wasn't already masked)
            img.mask = np.logical_or(self.bkg_wimage == 0, self.image.mask)
            self._bkg_array = np.ma.median(img, axis=self.image.crossdisp_axis)
        else:
            raise ValueError("statistic must be 'average' or 'median'")

    def _set_traces(self):
        """Determine `traces` from input. If an integer/float or list if int/float
           is passed in, use these to construct FlatTrace objects. These values
           must be positive. If None (which is initialized to an empty list),
           construct a FlatTrace using the center of image (according to disp.
           axis). Otherwise, any Trace object or list of Trace objects can be
           passed in."""

        if self.traces == []:
            # assume a flat trace at the image center if nothing is passed in.
            trace_pos = self.image.shape[self.image.disp_axis] / 2.
            self.traces = [FlatTrace(self.image.nddata, trace_pos)]

        if isinstance(self.traces, Trace):
            # if just one trace, turn it into iterable.
            self.traces = [self.traces]
            return

        # finally, if float/int is passed in convert to FlatTrace(s)
        if isinstance(self.traces, (float, int)):  # for a single number
            self.traces = [self.traces]

        if np.all([isinstance(x, (float, int)) for x in self.traces]):
            self.traces = [FlatTrace(self.image.nddata, trace_pos) for trace_pos in self.traces]
            return

        else:
            if not np.all([isinstance(x, Trace) for x in self.traces]):
                raise ValueError('`traces` must be a `Trace` object or list of '
                                 '`Trace` objects, a number or list of numbers to '
                                 'define FlatTraces, or None to use a FlatTrace in '
                                 'the middle of the image.')

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
        trace_object: `~specreduce.tracing.Trace`
            estimated trace of the spectrum to center the background traces
        separation: float
            separation from ``trace_object`` for the background regions
        width : float
            width of each background aperture in pixels
        statistic: string
            statistic to use when computing the background.  'average' will
            account for partial pixel weights, 'median' will include all partial
            pixels.
        disp_axis : int
            dispersion axis
        crossdisp_axis : int
            cross-dispersion axis
        mask_treatment : string
            The method for handling masked or non-finite data. Choice of `filter`,
            `omit`, or `zero-fill`. If `filter` is chosen, masked/non-finite data
            will be filtered during the fit to each bin/column (along disp. axis) to
            find the peak. If `omit` is chosen, columns along disp_axis with any
            masked/non-finite data values will be fully masked (i.e, 2D mask is
            collapsed to 1D and applied). If `zero-fill` is chosen, masked/non-finite
            data will be replaced with 0.0 in the input image, and the mask will then
            be dropped. For all three options, the input mask (optional on input
            NDData object) will be combined with a mask generated from any non-finite
            values in the image data.
        """
        kwargs['traces'] = [trace_object-separation, trace_object+separation]
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
        trace_object: `~specreduce.tracing.Trace`
            estimated trace of the spectrum to center the background traces
        separation: float
            separation from ``trace_object`` for the background, positive will be
            above the trace, negative below.
        width : float
            width of each background aperture in pixels
        statistic: string
            statistic to use when computing the background.  'average' will
            account for partial pixel weights, 'median' will include all partial
            pixels.
        disp_axis : int
            dispersion axis
        crossdisp_axis : int
            cross-dispersion axis
        mask_treatment : string
            The method for handling masked or non-finite data. Choice of `filter`,
            `omit`, or `zero-fill`. If `filter` is chosen, masked/non-finite data
            will be filtered during the fit to each bin/column (along disp. axis) to
            find the peak. If `omit` is chosen, columns along disp_axis with any
            masked/non-finite data values will be fully masked (i.e, 2D mask is
            collapsed to 1D and applied). If `zero-fill` is chosen, masked/non-finite
            data will be replaced with 0.0 in the input image, and the mask will then
            be dropped. For all three options, the input mask (optional on input
            NDData object) will be combined with a mask generated from any non-finite
            values in the image data.
        """
        kwargs['traces'] = [trace_object+separation]
        return cls(image=image, **kwargs)

    def bkg_image(self, image=None,
                  disp_axis: int = 1,
                  crossdisp_axis: int | None = None) -> SRImage:
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
        `~specutils.SRImage` object with same shape as ``image``.
        """
        image = self.image if image is None else as_image(image, disp_axis, crossdisp_axis)
        return SRImage(np.tile(self._bkg_array,
                               (image.shape[image.crossdisp_axis], 1)) * image.unit)

    def bkg_spectrum(self, image=None) -> Spectrum1D:
        """
        Expose the 1D spectrum of the background.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, optional
            Image with 2-D spectral image data. Assumes cross-dispersion
            (spatial) direction is axis 0 and dispersion (wavelength)
            direction is axis 1. If None, will extract the background
            from ``image`` used to initialize the class. [default: None]

        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The background 1-D spectrum, with flux expressed in the same
            units as the input image (or u.DN if none were provided) and
            the spectral axis expressed in pixel units.
        """
        img = self.bkg_image(image)
        return Spectrum1D(np.nansum(img.flux, axis=img.crossdisp_axis) * img.unit)

    def sub_image(self, image=None, disp_axis: int = 1, crossdisp_axis: int | None = None):
        """
        Subtract the computed background from ``image``.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        `~specreduce.SRImage` object with same shape as ``image``.
        """
        image = self.image if image is None else as_image(image, disp_axis, crossdisp_axis)
        return image.subtract(self.bkg_image(image).nddata)

    def sub_spectrum(self, image=None) -> Spectrum1D:
        """
        Expose the 1D spectrum of the background-subtracted image.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The background 1-D spectrum, with flux expressed in the same
            units as the input image (or u.DN if none were provided) and
            the spectral axis expressed in pixel units.
        """
        sub_image = self.sub_image(image=image)
        return Spectrum1D(np.nansum(sub_image.flux, axis=sub_image.crossdisp_axis) * sub_image.unit)

    def __rsub__(self, image):
        """
        Subtract the background from an image.
        """
        return self.sub_image(image)
