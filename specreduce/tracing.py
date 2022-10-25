# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
from dataclasses import dataclass, field
import warnings

from astropy.modeling import fitting, models
from astropy.nddata import CCDData, NDData
from astropy.stats import gaussian_sigma_to_fwhm
from scipy.interpolate import UnivariateSpline
import numpy as np

__all__ = ['BaseTrace', 'Trace', 'FlatTrace', 'ArrayTrace', 'KosmosTrace']


@dataclass(frozen=True)
class BaseTrace:
    """
    A dataclass common to all Trace objects.
    """
    image: CCDData
    _trace_pos: (float, np.ndarray) = field(repr=False)
    _trace: np.ndarray = field(repr=False)

    def __post_init__(self):
        # this class only exists to catch __post_init__ calls in its
        # subclasses, so that super().__post_init__ calls work correctly.
        pass

    def __getitem__(self, i):
        return self.trace[i]

    def _bound_trace(self):
        """
        Mask trace positions that are outside the upper/lower bounds of the image.
        """
        ny = self.image.shape[0]
        object.__setattr__(self, '_trace', np.ma.masked_outside(self._trace, 0, ny - 1))

    def __add__(self, delta):
        """
        Return a copy of the trace shifted "forward" by delta pixels perpendicular to the axis
        being traced
        """
        copy = deepcopy(self)
        copy.shift(delta)
        return copy

    def __sub__(self, delta):
        """
        Return a copy of the trace shifted "backward" by delta pixels perpendicular to the axis
        being traced
        """
        return self.__add__(-delta)

    def shift(self, delta):
        """
        Shift the trace by delta pixels perpendicular to the axis being traced

        Parameters
        ----------
        delta : float
            Shift to be applied to the trace
        """
        # act on self._trace.data to ignore the mask and then re-mask when calling _bound_trace
        object.__setattr__(self, '_trace', np.asarray(self._trace.data) + delta)
        object.__setattr__(self, '_trace_pos', self._trace_pos + delta)
        self._bound_trace()

    @property
    def shape(self):
        return self._trace.shape

    @property
    def trace(self):
        return self._trace

    @property
    def trace_pos(self):
        return self._trace_pos

    @staticmethod
    def _default_trace_attrs(image):
        """
        Compute a default trace position and trace array using only
        the image dimensions.
        """
        trace_pos = image.shape[0] / 2
        trace = np.ones_like(image[0]) * trace_pos
        return trace_pos, trace


@dataclass(init=False, frozen=True)
class Trace(BaseTrace):
    """
    Basic tracing class that by default traces the middle of the image.

    Parameters
    ----------
    image : `~astropy.nddata.CCDData`
        Image to be traced
    """
    def __init__(self, image):
        trace_pos, trace = self._default_trace_attrs(image)
        super().__init__(image, trace_pos, trace)


@dataclass(init=False, frozen=True)
class FlatTrace(BaseTrace):
    """
    Trace that is constant along the axis being traced

    Example: ::

        trace = FlatTrace(image, trace_pos)

    Parameters
    ----------
    trace_pos : float
        Position of the trace
    """

    def __init__(self, image, trace_pos):
        _, trace = self._default_trace_attrs(image)
        super().__init__(image, trace_pos, trace)
        self.set_position(trace_pos)

    def set_position(self, trace_pos):
        """
        Set the trace position within the image

        Parameters
        ----------
        trace_pos : float
            Position of the trace
        """
        object.__setattr__(self, '_trace_pos', trace_pos)
        object.__setattr__(self, '_trace', np.ones_like(self.image[0]) * trace_pos)
        self._bound_trace()


@dataclass(init=False, frozen=True)
class ArrayTrace(BaseTrace):
    """
    Define a trace given an array of trace positions

    Parameters
    ----------
    trace : `numpy.ndarray`
        Array containing trace positions
    """
    def __init__(self, image, trace):
        trace_pos, _ = self._default_trace_attrs(image)
        super().__init__(image, trace_pos, trace)

        nx = self.image.shape[1]
        nt = len(trace)
        if nt != nx:
            if nt > nx:
                # truncate trace to fit image
                trace = trace[0:nx]
            else:
                # assume trace starts at beginning of image and pad out trace to fit.
                # padding will be the last value of the trace, but will be masked out.
                padding = np.ma.MaskedArray(np.ones(nx - nt) * trace[-1], mask=True)
                trace = np.ma.hstack([trace, padding])
        object.__setattr__(self, '_trace', trace)
        self._bound_trace()


@dataclass(init=False, frozen=True)
class KosmosTrace(BaseTrace):
    """
    Trace the spectrum aperture in an image.

    Chops image up in bins along the dispersion (wavelength) direction,
    fits a Gaussian within each bin to determine the trace's spatial
    center. Finally, draws a cubic spline through the bins to up-sample
    trace along every pixel in the dispersion direction.

    (The original version of this algorithm is sourced from James
    Davenport's ``kosmos`` repository.)


    Example: ::

        trace = KosmosTrace(image, guess=trace_pos)

    Parameters
    ----------
    image : `~astropy.nddata.NDData` or array-like, required
        The image over which to run the trace. Assumes cross-dispersion
        (spatial) direction is axis 0 and dispersion (wavelength)
        direction is axis 1.
    bins : int, optional
        The number of bins in the dispersion (wavelength) direction
        into which to divide the image. Use fewer if KosmosTrace is
        having difficulty, such as with faint targets.
        Minimum bin size is 4. [default: 20]
    guess : int, optional
        A guess at the trace's location in the cross-dispersion
        (spatial) direction. If set, overrides the normal max peak
        finder. Good for tracing a fainter source if multiple traces
        are present. [default: None]
    window : int, optional
        Fit the trace to a region with size ``window * 2`` around the
        guess position. Useful for tracing faint sources if multiple
        traces are present, but potentially bad if the trace is
        substantially bent or warped. [default: None]
    peak_method : string, optional
        One of ``gaussian`` (default), ``centroid``, or ``max``.
        gaussian: fits a gaussian to the window within each bin and
        adopts the central value.  centroid: takes the centroid of the
        window within in bin.  smooth_max: takes the position with the
        maximum flux after smoothing over the window within each bin.

    Improvements Needed
    -------------------
    1) switch to astropy models for Gaussian (done)
    2) return info about trace width (?)
    3) add re-fit trace functionality (or break off into other method)
    4) add other interpolation modes besides spline, maybe via
        specutils.manipulation methods?
    """
    bins: int
    guess: float
    window: int
    peak_method: str
    _crossdisp_axis = 0
    _disp_axis = 1

    def _process_init_kwargs(self, **kwargs):
        for attr, value in kwargs.items():
            object.__setattr__(self, attr, value)

    def __init__(self, image, bins=20, guess=None, window=None, peak_method='gaussian'):
        # This method will assign the user supplied value (or default) to the attrs:
        self._process_init_kwargs(
            bins=bins, guess=guess, window=window, peak_method=peak_method
        )
        trace_pos, trace = self._default_trace_attrs(image)
        super().__init__(image, trace_pos, trace)

        # handle multiple image types and mask uncaught invalid values
        if isinstance(self.image, NDData):
            img = np.ma.masked_invalid(np.ma.masked_array(self.image.data,
                                                          mask=self.image.mask))
        else:
            img = np.ma.masked_invalid(self.image)

        # validate arguments
        valid_peak_methods = ('gaussian', 'centroid', 'max')
        if self.peak_method not in valid_peak_methods:
            raise ValueError(f"peak_method must be one of {valid_peak_methods}")

        if img.mask.all():
            raise ValueError('image is fully masked. Check for invalid values')

        if self._crossdisp_axis != 0:
            raise ValueError('cross-dispersion axis must equal 0')

        if self._disp_axis != 1:
            raise ValueError('dispersion axis must equal 1')

        if not isinstance(self.bins, int):
            warnings.warn('TRACE: Converting bins to int')
            object.__setattr__(self, 'bins', int(self.bins))

        if self.bins < 4:
            raise ValueError('bins must be >= 4')

        cols = img.shape[self._disp_axis]
        if self.bins >= cols:
            raise ValueError(f"bins must be < {cols}, the length of the "
                             "image's spatial direction")

        if (self.window is not None
            and (self.window > img.shape[self._disp_axis]
                 or self.window < 1)):
            raise ValueError(f"window must be >= 2 and less than {cols}, the "
                             "length of the image's spatial direction")
        elif self.window is not None and not isinstance(self.window, int):
            warnings.warn('TRACE: Converting window to int')
            object.__setattr__(self, 'window', int(self.window))

        # set max peak location by user choice or wavelength with max avg flux
        ztot = img.sum(axis=self._disp_axis) / img.shape[self._disp_axis]
        peak_y = self.guess if self.guess is not None else ztot.argmax()
        # NOTE: peak finder can be bad if multiple objects are on slit

        yy = np.arange(img.shape[self._crossdisp_axis])

        if self.peak_method == 'gaussian':
            # guess the peak width as the FWHM, roughly converted to gaussian sigma
            yy_above_half_max = np.sum(ztot > (ztot.max() / 2))
            width_guess = yy_above_half_max / gaussian_sigma_to_fwhm

            # enforce some (maybe sensible?) rules about trace peak width
            width_guess = (2 if width_guess < 2
                           else 25 if width_guess > 25
                           else width_guess)

            # fit a Gaussian to peak for fall-back answer, but don't use yet
            g1d_init = models.Gaussian1D(amplitude=ztot.max(),
                                         mean=peak_y, stddev=width_guess)
            offset_init = models.Const1D(np.ma.median(ztot))
            profile = g1d_init + offset_init

            fitter = fitting.LevMarLSQFitter()
            popt_tot = fitter(profile, yy, ztot)

        # restrict fit to window (if one exists)
        ilum2 = (yy if self.window is None
                 else yy[np.arange(peak_y - self.window,
                                   peak_y + self.window, dtype=int)])
        if img[ilum2].mask.all():
            raise ValueError('All pixels in window region are masked. Check '
                             'for invalid values or use a larger window value.')

        x_bins = np.linspace(0, img.shape[self._disp_axis],
                             self.bins + 1, dtype=int)
        y_bins = np.tile(np.nan, self.bins)

        for i in range(self.bins):
            # repeat earlier steps to create gaussian fit for each bin
            z_i = img[ilum2, x_bins[i]:x_bins[i+1]].sum(axis=self._disp_axis)
            if not z_i.mask.all():
                peak_y_i = ilum2[z_i.argmax()]
            else:
                warnings.warn(f"All pixels in bin {i} are masked. Falling "
                              'to trace value from all-bin fit.')
                peak_y_i = peak_y

            if self.peak_method == 'gaussian':
                yy_i_above_half_max = np.sum(z_i > (z_i.max() / 2))
                width_guess_i = yy_i_above_half_max / gaussian_sigma_to_fwhm

                # NOTE: original KOSMOS code mandated width be greater than 2
                # (to avoid cosmic rays) and less than 25 (to avoid fitting noise).
                # we should extract values from img to derive similar limits
                # width_guess_i = (2 if width_guess_i < 2
                #                  else 25 if width_guess_i > 25
                #                  else width_guess_i)

                g1d_init_i = models.Gaussian1D(amplitude=z_i.max(),
                                               mean=peak_y_i,
                                               stddev=width_guess_i)
                offset_init_i = models.Const1D(np.ma.median(z_i))

                profile_i = g1d_init_i + offset_init_i
                popt_i = fitter(profile_i, ilum2, z_i)

                # if gaussian fits off chip, then fall back to previous answer
                if not ilum2.min() <= popt_i.mean_0 <= ilum2.max():
                    y_bins[i] = popt_tot.mean_0.value
                else:
                    y_bins[i] = popt_i.mean_0.value
                    popt_tot = popt_i

            elif self.peak_method == 'centroid':
                z_i_cumsum = np.cumsum(z_i)
                # find the interpolated index where the cumulative array reaches half the total
                # cumulative values
                y_bins[i] = np.interp(z_i_cumsum[-1]/2., z_i_cumsum, ilum2)

            elif self.peak_method == 'max':
                # TODO: implement smoothing with provided width
                y_bins[i] = ilum2[z_i.argmax()]

        # recenter bin positions
        x_bins = (x_bins[:-1] + x_bins[1:]) / 2

        # interpolate the fitted trace over the entire wavelength axis
        y_finite = np.where(np.isfinite(y_bins))[0]
        if y_finite.size > 0:
            x_bins = x_bins[y_finite]
            y_bins = y_bins[y_finite]

            # run a cubic spline through the bins; interpolate over wavelengths
            ap_spl = UnivariateSpline(x_bins, y_bins, k=3, s=0)
            trace_x = np.arange(img.shape[self._disp_axis])
            trace_y = ap_spl(trace_x)
        else:
            warnings.warn("TRACE ERROR: No valid points found in trace")
            trace_y = np.tile(np.nan, len(x_bins))

        object.__setattr__(self, '_trace', np.ma.masked_invalid(trace_y))
