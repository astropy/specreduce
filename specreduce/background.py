# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.nddata import NDData
from astropy import units as u

from specreduce.extract import _ap_weight_image, _to_spectrum1d_pixels
from specreduce.tracing import Trace, FlatTrace

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
    image : `~astropy.nddata.NDData` or array-like
        image with 2-D spectral image data
    traces : List
        list of trace objects (or integers to define FlatTraces) to
        extract the background
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

    def __post_init__(self):
        """
        Determine the background from an image for subtraction.

        Parameters
        ----------
        image : `~astropy.nddata.NDData` or array-like
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
        def _to_trace(trace):
            if not isinstance(trace, Trace):
                trace = FlatTrace(self.image, trace)

            # TODO: this check can be removed if/when implemented as a check in FlatTrace
            if isinstance(trace, FlatTrace):
                if trace.trace_pos < 1:
                    raise ValueError('trace_object.trace_pos must be >= 1')
            return trace

        if self.width < 0:
            raise ValueError("width must be positive")

        if self.width == 0:
            self.bkg_array = np.zeros(self.image.shape[self.disp_axis])
            return

        bkg_wimage = np.zeros_like(self.image, dtype=np.float64)
        for trace in self.traces:
            trace = _to_trace(trace)
            windows_max = trace.trace.data.max() + self.width/2
            windows_min = trace.trace.data.min() - self.width/2
            if windows_max >= self.image.shape[self.crossdisp_axis]:
                warnings.warn("background window extends above image boundaries " +
                              f"({windows_max} >= {self.image.shape[self.crossdisp_axis]})")
            if windows_min < 0:
                warnings.warn("background window extends below image boundaries " +
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

        if self.statistic == 'average':
            self.bkg_array = np.average(self.image, weights=self.bkg_wimage,
                                        axis=self.crossdisp_axis)
        elif self.statistic == 'median':
            med_image = self.image.copy()
            med_image[np.where(self.bkg_wimage) == 0] = np.nan
            self.bkg_array = np.nanmedian(med_image, axis=self.crossdisp_axis)
        else:
            raise ValueError("statistic must be 'average' or 'median'")

    @classmethod
    def two_sided(cls, image, trace_object, separation, **kwargs):
        """
        Determine the background from an image for subtraction centered around
        an input trace.


        Example: ::

            trace = KosmosTrace(image, guess=trace_pos)
            bg = Background.two_sided(image, trace, bkg_sep, width=bkg_width)

        Parameters
        ----------
        image : nddata-compatible image
            image with 2-D spectral image data
        trace_object: Trace
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
        kwargs['traces'] = [trace_object-separation, trace_object+separation]
        return cls(image=image, **kwargs)

    @classmethod
    def one_sided(cls, image, trace_object, separation, **kwargs):
        """
        Determine the background from an image for subtraction above
        or below an input trace.

        Example: ::

            trace = KosmosTrace(image, guess=trace_pos)
            bg = Background.one_sided(image, trace, bkg_sep, width=bkg_width)

        Parameters
        ----------
        image : nddata-compatible image
            image with 2-D spectral image data
        trace_object: Trace
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
        kwargs['traces'] = [trace_object+separation]
        return cls(image=image, **kwargs)

    def bkg_image(self, image=None):
        """
        Expose the background tiled to the dimension of ``image``.

        Parameters
        ----------
        image : nddata-compatible image or None
            image with 2-D spectral image data.  If None, will extract
            the background from ``image`` used to initialize the class.

        Returns
        -------
        array with same shape as ``image``.
        """
        if image is None:
            image = self.image

        return np.tile(self.bkg_array, (image.shape[0], 1))

    def bkg_spectrum(self, image=None):
        """
        Expose the 1D spectrum of the background.

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
        bkg_image = self.bkg_image(image=image)

        ext1d = np.sum(bkg_image, axis=self.crossdisp_axis)
        return _to_spectrum1d_pixels(ext1d * getattr(image, 'unit', u.DN))

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
        array with same shape as ``image``
        """
        if image is None:
            image = self.image

        if isinstance(image, NDData):
            # https://docs.astropy.org/en/stable/nddata/mixins/ndarithmetic.html
            return image.subtract(self.bkg_image(image)*image.unit)
        else:
            return image - self.bkg_image(image)

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

        ext1d = np.sum(sub_image, axis=self.crossdisp_axis)
        return _to_spectrum1d_pixels(ext1d * getattr(image, 'unit', u.DN))

    def __rsub__(self, image):
        """
        Subtract the background from an image.
        """
        return self.sub_image(image)
