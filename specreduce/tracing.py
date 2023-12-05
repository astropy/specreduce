# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from astropy.modeling import Model, fitting, models
from astropy.nddata import NDData
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.utils.decorators import deprecated

from specreduce.core import _ImageParser

__all__ = ['Trace', 'FlatTrace', 'ArrayTrace', 'FitTrace']


@dataclass
class Trace:
    """
    Basic tracing class that by default traces the middle of the image.

    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like, required
        Image to be traced

    Properties
    ----------
    shape : tuple
        Shape of the array describing the trace
    """
    image: NDData

    def __post_init__(self):
        self.trace_pos = self.image.shape[0] / 2
        self.trace = np.ones_like(self.image[0]) * self.trace_pos

    def __getitem__(self, i):
        return self.trace[i]

    @property
    def shape(self):
        return self.trace.shape

    def shift(self, delta):
        """
        Shift the trace by delta pixels perpendicular to the axis being traced

        Parameters
        ----------
        delta : float
            Shift to be applied to the trace
        """
        # act on self.trace.data to ignore the mask and then re-mask when calling _bound_trace
        self.trace = np.asarray(self.trace.data) + delta
        self._bound_trace()

    def _bound_trace(self):
        """
        Mask trace positions that are outside the upper/lower bounds of the image.
        """
        ny = self.image.shape[0]
        self.trace = np.ma.masked_outside(self.trace, 0, ny-1)

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


@dataclass
class FlatTrace(Trace, _ImageParser):
    """
    Trace that is constant along the axis being traced.

    Example: ::

        trace = FlatTrace(image, trace_pos)

    Parameters
    ----------
    trace_pos : float
        Position of the trace
    """
    trace_pos: float

    def __post_init__(self):
        self.image = self._parse_image(self.image)

        self.set_position(self.trace_pos)

    def set_position(self, trace_pos):
        """
        Set the trace position within the image

        Parameters
        ----------
        trace_pos : float
            Position of the trace
        """
        self.trace_pos = trace_pos
        self.trace = np.ones_like(self.image.data[0]) * self.trace_pos
        self._bound_trace()


@dataclass
class ArrayTrace(Trace, _ImageParser):
    """
    Define a trace given an array of trace positions.

    Parameters
    ----------
    trace : `numpy.ndarray`
        Array containing trace positions
    """
    trace: np.ndarray

    def __post_init__(self):
        self.image = self._parse_image(self.image)

        nx = self.image.shape[1]
        nt = len(self.trace)
        if nt != nx:
            if nt > nx:
                # truncate trace to fit image
                self.trace = self.trace[0:nx]
            else:
                # assume trace starts at beginning of image and pad out trace to fit.
                # padding will be the last value of the trace, but will be masked out.
                padding = np.ma.MaskedArray(np.ones(nx - nt) * self.trace[-1], mask=True)
                self.trace = np.ma.hstack([self.trace, padding])
        self._bound_trace()


@dataclass
class FitTrace(Trace, _ImageParser):
    """
    Trace the spectrum aperture in an image.

    Bins along the image's dispersion (wavelength) direction, finds each
    bin's peak cross-dispersion (spatial) pixel, and uses a model to
    interpolate the function fitted to the peaks as a final trace. The
    number of bins, peak finding algorithm, and model used for fitting
    are customizable by the user.


    Example: ::

        trace = FitTrace(image, peak_method='gaussian', guess=trace_pos)

    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like, required
        The image over which to run the trace. Assumes cross-dispersion
        (spatial) direction is axis 0 and dispersion (wavelength)
        direction is axis 1.
    bins : int, optional
        The number of bins in the dispersion (wavelength) direction
        into which to divide the image. If not set, defaults to one bin
        per dispersion (wavelength) pixel in the given image. If set,
        requires at least 4 or N bins for a degree N ``trace_model``,
        whichever is greater. [default: None]
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
    trace_model : one of `~astropy.modeling.polynomial.Chebyshev1D`,\
            `~astropy.modeling.polynomial.Legendre1D`,\
            `~astropy.modeling.polynomial.Polynomial1D`,\
            or `~astropy.modeling.spline.Spline1D`, optional
        The 1-D polynomial model used to fit the trace to the bins' peak
        pixels. Spline1D models are fit with Astropy's
        'SplineSmoothingFitter', while the other models are fit with the
        'LevMarLSQFitter'. [default: ``models.Polynomial1D(degree=1)``]
    peak_method : string, optional
        One of ``gaussian``, ``centroid``, or ``max``.
        ``gaussian``: Fits a gaussian to the window within each bin and
        adopts the central value as the peak. May work best with fewer
        bins on faint targets. (Based on the "kosmos" algorithm from
        James Davenport's same-named repository.)
        ``centroid``: Takes the centroid of the window within in bin.
        ``max``: Saves the position with the maximum flux in each bin.
        [default: ``max``]
    """
    bins: int = None
    guess: float = None
    window: int = None
    trace_model: Model = field(default=models.Polynomial1D(degree=1))
    peak_method: str = 'max'
    _crossdisp_axis = 0
    _disp_axis = 1

    def __post_init__(self):
        self.image = self._parse_image(self.image)

        # mask any previously uncaught invalid values
        or_mask = np.logical_or(self.image.mask, ~np.isfinite(self.image.data))
        img = np.ma.masked_array(self.image.data, or_mask)

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

        valid_models = (models.Spline1D, models.Legendre1D,
                        models.Chebyshev1D, models.Polynomial1D)
        if not isinstance(self.trace_model, valid_models):
            raise ValueError("trace_model must be one of "
                             f"{', '.join([m.name for m in valid_models])}.")

        cols = img.shape[self._disp_axis]
        model_deg = self.trace_model.degree
        if self.bins is None:
            self.bins = cols
        elif self.bins < 4:
            # many of the Astropy model fitters require four points at minimum
            raise ValueError('bins must be >= 4')
        elif self.bins <= model_deg:
            raise ValueError(f"bins must be > {model_deg} for "
                             f"a degree {model_deg} model.")
        elif self.bins > cols:
            raise ValueError(f"bins must be <= {cols}, the length of the "
                             "image's spatial direction")

        if not isinstance(self.bins, int):
            warnings.warn('TRACE: Converting bins to int')
            self.bins = int(self.bins)

        if (self.window is not None
            and (self.window > img.shape[self._disp_axis]
                 or self.window < 1)):
            raise ValueError(f"window must be >= 2 and less than {cols}, the "
                             "length of the image's spatial direction")
        elif self.window is not None and not isinstance(self.window, int):
            warnings.warn('TRACE: Converting window to int')
            self.window = int(self.window)

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

            # use given model to bin y-values; interpolate over all wavelengths
            fitter = (fitting.SplineSmoothingFitter()
                      if isinstance(self.trace_model, models.Spline1D)
                      else fitting.LevMarLSQFitter())
            self.trace_model_fit = fitter(self.trace_model, x_bins, y_bins)

            trace_x = np.arange(img.shape[self._disp_axis])
            trace_y = self.trace_model_fit(trace_x)
        else:
            warnings.warn("TRACE ERROR: No valid points found in trace")
            trace_y = np.tile(np.nan, img.shape[self._disp_axis])

        self.trace = np.ma.masked_invalid(trace_y)


@deprecated('1.3', alternative='FitTrace')
@dataclass
class KosmosTrace(FitTrace):
    """
    This class is pending deprecation. Please use `FitTrace` instead.
    """
    __doc__ += FitTrace.__doc__
    pass
