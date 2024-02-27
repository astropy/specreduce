# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.nddata import NDData
from astropy.utils.decorators import deprecated_attribute
from specutils import Spectrum1D

from specreduce.core import _ImageParser
from specreduce.extract import _ap_weight_image
from specreduce.tracing import Trace, FlatTrace

__all__ = ['Background']


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
    image : `~astropy.nddata.NDData`-like or array-like
        image with 2-D spectral image data
    traces : trace, int, float (single or list)
        Individual or list of trace object(s) (or integers/floats to define
        FlatTraces) to extract the background. If None, a FlatTrace at the
        center of the image (according to `disp_axis`) will be used.
    width : float
        width of extraction aperture in pixels
    statistic: string
        statistic to use when computing the background.  'average' will
        account for partial pixel weights, 'median' will include all partial
        pixels.
    disp_axis : int
        dispersion axis
    crossdisp_axis : int
        cross-dispersion axis
    """
    # required so numpy won't call __rsub__ on individual elements
    # https://stackoverflow.com/a/58409215
    __array_ufunc__ = None

    image: NDData
    traces: list = field(default_factory=list)
    width: float = 5
    statistic: str = 'average'
    disp_axis: int = 1
    crossdisp_axis: int = 0

    # TO-DO: update bkg_array with Spectrum1D alternative (is bkg_image enough?)
    bkg_array = deprecated_attribute('bkg_array', '1.3')

    def __post_init__(self):
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
        """
        self.image = self._parse_image(self.image)

        if self.width < 0:
            raise ValueError("width must be positive")
        if self.width == 0:
            self._bkg_array = np.zeros(self.image.shape[self.disp_axis])
            return

        self._set_traces()

        bkg_wimage = np.zeros_like(self.image.data, dtype=np.float64)
        for trace in self.traces:
            windows_max = trace.trace.data.max() + self.width/2
            windows_min = trace.trace.data.min() - self.width/2
            if windows_max >= self.image.shape[self.crossdisp_axis]:
                warnings.warn("background window extends beyond image boundaries " +
                              f"({windows_max} >= {self.image.shape[self.crossdisp_axis]})")
            if windows_min < 0:
                warnings.warn("background window extends beyond image boundaries " +
                              f"({windows_min} < 0)")

            # pass trace.trace.data to ignore any mask on the trace
            bkg_wimage += _ap_weight_image(trace,
                                           self.width,
                                           self.disp_axis,
                                           self.crossdisp_axis,
                                           self.image.shape)

        if np.any(bkg_wimage > 1):
            raise ValueError("background regions overlapped")
        if np.any(np.sum(bkg_wimage, axis=self.crossdisp_axis) == 0):
            raise ValueError("background window does not remain in bounds across entire dispersion axis")  # noqa

        if self.statistic == 'median':
            # make it clear in the expose image that partial pixels are fully-weighted
            bkg_wimage[bkg_wimage > 0] = 1

        self.bkg_wimage = bkg_wimage

        # mask user-highlighted and invalid values (if any) before taking stats
        or_mask = (np.logical_or(~np.isfinite(self.image.data), self.image.mask)
                   if self.image.mask is not None
                   else ~np.isfinite(self.image.data))

        if self.statistic == 'average':
            image_ma = np.ma.masked_array(self.image.data, mask=or_mask)
            self._bkg_array = np.ma.average(image_ma,
                                            weights=self.bkg_wimage,
                                            axis=self.crossdisp_axis).data
        elif self.statistic == 'median':
            med_mask = np.logical_or(self.bkg_wimage == 0, or_mask)
            image_ma = np.ma.masked_array(self.image.data, mask=med_mask)
            self._bkg_array = np.ma.median(image_ma, axis=self.crossdisp_axis).data
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
            trace_pos = self.image.shape[self.disp_axis] / 2.
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
        """
        image = _ImageParser._get_data_from_image(image) if image is not None else cls.image
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
        """
        image = _ImageParser._get_data_from_image(image) if image is not None else cls.image
        kwargs['traces'] = [trace_object+separation]
        return cls(image=image, **kwargs)

    def bkg_image(self, image=None):
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
        `~specutils.Spectrum1D` object with same shape as ``image``.
        """
        image = self._parse_image(image)
        return Spectrum1D(np.tile(self._bkg_array,
                                  (image.shape[0], 1)) * image.unit,
                          spectral_axis=image.spectral_axis)

    def bkg_spectrum(self, image=None):
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
        bkg_image = self.bkg_image(image)

        try:
            return bkg_image.collapse(np.nansum, axis=self.crossdisp_axis)
        except u.UnitTypeError:
            # can't collapse with a spectral axis in pixels because
            # SpectralCoord only allows frequency/wavelength equivalent units...
            ext1d = np.nansum(bkg_image.flux, axis=self.crossdisp_axis)
            return Spectrum1D(ext1d, bkg_image.spectral_axis)

    def sub_image(self, image=None):
        """
        Subtract the computed background from ``image``.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        `~specutils.Spectrum1D` object with same shape as ``image``.
        """
        image = self._parse_image(image)

        # a compare_wcs argument is needed for Spectrum1D.subtract() in order to
        # avoid a TypeError from SpectralCoord when image's spectral axis is in
        # pixels. it is not needed when image's spectral axis has physical units
        kwargs = ({'compare_wcs': None} if image.spectral_axis.unit == u.pix
                  else {})

        # https://docs.astropy.org/en/stable/nddata/mixins/ndarithmetic.html
        return image.subtract(self.bkg_image(image), **kwargs)

    def sub_spectrum(self, image=None):
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

        try:
            return sub_image.collapse(np.nansum, axis=self.crossdisp_axis)
        except u.UnitTypeError:
            # can't collapse with a spectral axis in pixels because
            # SpectralCoord only allows frequency/wavelength equivalent units...
            ext1d = np.nansum(sub_image.flux, axis=self.crossdisp_axis)
            return Spectrum1D(ext1d, spectral_axis=sub_image.spectral_axis)

    def __rsub__(self, image):
        """
        Subtract the background from an image.
        """
        return self.sub_image(image)
