# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass
import warnings

from astropy.modeling import CompoundModel, fitting, models
from astropy.nddata import CCDData
from astropy.stats import gaussian_sigma_to_fwhm
from scipy.interpolate import UnivariateSpline
import numpy as np

__all__ = ['Trace', 'FlatTrace', 'ArrayTrace', 'KosmosTrace']


@dataclass
class Trace:
    """
    Basic tracing class that by default traces the middle of the image.

    Parameters
    ----------
    image : `~astropy.nddata.CCDData`
        Image to be traced

    Properties
    ----------
    shape : tuple
        Shape of the array describing the trace
    """
    image: CCDData

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
        self.trace += delta
        self._bound_trace()

    def _bound_trace(self):
        """
        Mask trace positions that are outside the upper/lower bounds of the image.
        """
        ny = self.image.shape[0]
        self.trace = np.ma.masked_outside(self.trace, 0, ny-1)


@dataclass
class FlatTrace(Trace):
    """
    Trace that is constant along the axis being traced

    Parameters
    ----------
    trace_pos : float
        Position of the trace
    """
    trace_pos: float

    def __post_init__(self):
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
        self.trace = np.ones_like(self.image[0]) * self.trace_pos
        self._bound_trace()


@dataclass
class ArrayTrace(Trace):
    """
    Define a trace given an array of trace positions

    Parameters
    ----------
    trace : `numpy.ndarray`
        Array containing trace positions
    """
    trace: np.ndarray

    def __post_init__(self):
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
class KosmosTrace(Trace):
    """
    Trace the spectrum aperture in an image.

    Chops image up in bins along the dispersion (wavelength) direction,
    fits a Gaussian within each bin to determine the trace's spatial
    center. Finally, draws a cubic spline through the bins to up-sample
    trace along every pixel in the dispersion direction.

    (The original version of this algorithm is sourced from James
     Davenport's `kosmos` repository.)

    Parameters
    ----------
    image : CCDData or array-like, required
        The image over which to run the trace. Assumes cross-dispersion
        (spatial) direction is axis 0 and dispersion (wavelength)
        direction is axis 1.
    bins : int, optional
        The number of bins in the dispersion (wavelength) into which to
        divide the image. Use fewer if KosmosTrace is having
        difficulty, such as with faint targets. Minimum bin size is 4.
        [default: 20]
    guess : int, optional
        A guess at the trace's location in the cross-dispersion
        (spatial) direction. If set, overrides the normal max peak
        finder. Good for tracing a fainter source if multiple traces
        are present. [default: None]
    window : int, optional
        Fit the trace to a region with size `window * 2` around the
        guess position. Useful for tracing faint sources if multiple
        traces are present, but potentially bad if the trace is
        substantially bent or warped. [default: None]

    Improvements Needed
    -------------------
    1) switch to astropy models for Gaussian (done)
    2) return info about trace width (?)
    3) add re-fit trace functionality (or break off into other method)
    4) add other interpolation modes besides spline, maybe via
        specutils.manipulation methods?
    """
    image: CCDData
    bins: int = 20
    guess: float = None
    window: int = None
    disp_axis = 1

    def __post_init__(self,
                      # Saxis=0, Waxis=1, display=False
                      ):
        if not isinstance(self.bins, int):
            warnings.warn('TRACE: Converting bins to int')
            self.bins = int(self.bins)
        if self.bins < 4:
            raise ValueError('bins must be >= 4')

        if (self.window is not None
            and (self.window > self.image.shape[self.disp_axis]
                 or self.window < 1)):
            raise ValueError("window must be >= 2 and less than the length of "
                             "the image's spatial direction")
        elif self.window is not None and not isinstance(self.window, int):
            warnings.warn('TRACE: Converting window to int')
            self.window = int(self.window)

        # set max peak location by user choice or wavelength with max avg flux
        ztot = np.nansum(self.image, axis=1) / self.image.shape[1]
        peak_y = self.guess if self.guess is not None else np.nanargmax(ztot)
        # (peak finder can be bad if multiple objects are on slit...)

        # guess the peak width as the FWHM, roughly converted to gaussian sigma
        yy = np.arange(len(ztot))
        yy_above_half_max = np.sum(ztot > (np.nanmax(ztot) / 2))
        width_guess = yy_above_half_max / gaussian_sigma_to_fwhm

        # enforce some (maybe sensible?) rules about trace peak width
        width_guess = (2 if width_guess < 2
                       else 25 if width_guess > 25
                       else width_guess)

        # fit a Gaussian to peak for fall-back answer, but don't use yet
        g1d_init = models.Gaussian1D(amplitude=np.nanmax(ztot),
                                     mean=peak_y, stddev=width_guess)
        offset_init = models.Const1D(np.nanmedian(ztot))
        profile = g1d_init + offset_init

        fitter = fitting.LevMarLSQFitter()
        popt_tot = fitter(profile, yy, ztot)

        # restrict fit to window (if one exists)
        ilum2 = (yy if self.window is None
                 else yy[np.arange(peak_y - self.window,
                                   peak_y + self.window, dtype=int)])

        x_bins = np.linspace(0, self.image.shape[self.disp_axis],
                             self.bins + 1, dtype=int)
        y_bins = np.tile(np.nan, self.bins)

        for i in range(self.bins):
            # repeat earlier steps to create gaussian fit for each bin
            z_i = np.nansum(self.image[ilum2, x_bins[i]:x_bins[i+1]],
                            axis=self.disp_axis)
            peak_y_i = ilum2[np.nanargmax(z_i)]

            yy_i_above_half_max = np.sum(z_i > (np.nanmax(z_i) / 2))
            width_guess_i = yy_i_above_half_max / gaussian_sigma_to_fwhm
            width_guess_i = (2 if width_guess_i < 2
                             else 25 if width_guess_i > 25
                             else width_guess_i)

            # save initial parameter guesses in case off issues with fits
            pguess = CompoundModel(
                '+',
                models.Gaussian1D(amplitude=np.nanmax(z_i),
                                  mean=peak_y_i, stddev=width_guess_i),
                models.Const1D(np.nanmedian(z_i)))

            try:
                g1d_init_i = models.Gaussian1D(amplitude=np.nanmax(z_i),
                                               mean=peak_y_i,
                                               stddev=width_guess_i)
                offset_init_i = models.Const1D(np.nanmedian(z_i))

                profile_i = g1d_init_i + offset_init_i
                popt_i = fitter(profile_i, ilum2, z_i)

                # if gaussian fits off chip, then fall back to previous answer
                if not ilum2.min() <= popt_i.mean_0 <= ilum2.max():
                    y_bins[i] = popt_tot.mean_0.value
                else:
                    y_bins[i] = popt_i.mean_0.value
                    popt_tot = popt_i

            except RuntimeError:
                # NOTE: would this happen? if not, pguess and try/except are unneeded
                warnings.warn(f"TRACE: Fitting bin {i} caused RuntimeError; "
                              "reverting to initial guess")
                popt_i = pguess

        # recenter bin positions
        x_bins = (x_bins[:-1] + x_bins[1:]) / 2

        # interpolate the fitted trace over the entire wavelength axis
        y_finite = np.where(np.isfinite(y_bins))[0]
        if y_finite.size > 0:
            x_bins = x_bins[y_finite]
            y_bins = y_bins[y_finite]

            # run a cubic spline through the bins; interpolate over wavelengths
            ap_spl = UnivariateSpline(x_bins, y_bins, k=3, s=0)
            trace_x = np.arange(self.image.shape[self.disp_axis])
            trace_y = ap_spl(trace_x)
        else:
            warnings.warn("TRACE ERROR: No valid points found in trace")
            trace_y = np.tile(np.nan, len(x_bins))

        self.trace = np.ma.masked_invalid(trace_y)
