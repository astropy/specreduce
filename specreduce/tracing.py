# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

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

    Attributes
    ----------
    shape : tuple
        Shape of the array describing the trace
    """
    image: NDData

    def __post_init__(self):
        self.trace_pos = self.image.shape[0] / 2
        self.trace = np.ones_like(self.image[0]) * self.trace_pos

        # masking options not relevant for basic Trace
        self._mask_treatment = None
        self._valid_mask_treatment_methods = [None]

        # eventually move this to common SpecreduceOperation base class
        self.validate_masking_options()

    def __getitem__(self, i):
        return self.trace[i]

    @property
    def shape(self):
        return self.trace.shape

    def validate_masking_options(self):
        if self.mask_treatment not in self.valid_mask_treatment_methods:
            raise ValueError(
                f'`mask_treatment` {self.mask_treatment} not one of {self.valid_mask_treatment_methods}')  # noqa

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
        self.trace = np.ma.masked_outside(self.trace, 0, ny - 1)

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

    @property
    def mask_treatment(self):
        return self._mask_treatment

    @property
    def valid_mask_treatment_methods(self):
        return self._valid_mask_treatment_methods


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

        # masking options not relevant for basic Trace
        self._mask_treatment = None
        self._valid_mask_treatment_methods = [None]

    def set_position(self, trace_pos):
        """
        Set the trace position within the image

        Parameters
        ----------
        trace_pos : float
            Position of the trace
        """
        if trace_pos < 1:
            raise ValueError('`trace_pos` must be positive.')
        self.trace_pos = trace_pos
        self.trace = np.ones_like(self.image.data[0]) * self.trace_pos
        self._bound_trace()


@dataclass
class ArrayTrace(Trace, _ImageParser):
    """
    Define a trace given an array of trace positions.

    Parameters
    ----------
    trace : `numpy.ndarray` or `numpy.ma.MaskedArray`
        Array containing trace positions.
    """
    trace: np.ndarray

    def __post_init__(self):

        # masking options not relevant for ArrayTrace. any non-finite or masked
        # data in `image` will not affect array trace
        self._mask_treatment = None
        self._valid_mask_treatment_methods = [None]

        # masked array will have a .data, regular array will not.
        trace_data = getattr(self.trace, 'data', self.trace)

        # but we do need to mask uncaught non-finite values in input trace array
        # which should also be combined with any existing mask in the input `trace`
        if hasattr(self.trace, 'mask'):
            total_mask = np.logical_or(self.trace.mask, ~np.isfinite(trace_data))
        else:
            total_mask = ~np.isfinite(trace_data)

        # always work with masked array, even if there is no masked
        # or nonfinite data, in case padding is needed. if not, mask will be
        # dropped at the end and a regular array will be returned.
        self.trace = np.ma.MaskedArray(trace_data, total_mask)

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

        # warn if entire trace is masked
        if np.all(self.trace.mask):
            warnings.warn("Entire trace array is masked.")

        # and return plain array if nothing is masked
        if not np.any(self.trace.mask):
            self.trace = self.trace.data


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
        'SplineSmoothingFitter', generic linear models are fit with the
        'LinearLSQFitter', while the other models are fit with the
        'LMLSQFitter'. [default: ``models.Polynomial1D(degree=1)``]
    peak_method : string, optional
        One of ``gaussian``, ``centroid``, or ``max``.
        ``gaussian``: Fits a gaussian to the window within each bin and
        adopts the central value as the peak. May work best with fewer
        bins on faint targets. (Based on the "kosmos" algorithm from
        James Davenport's same-named repository.)
        ``centroid``: Takes the centroid of the window within in bin.
        ``max``: Saves the position with the maximum flux in each bin.
        [default: ``max``]
    mask_treatment : string, optional
        The method for handling masked or non-finite data. Choice of ``filter`` or
        ``omit``. If `filter` is chosen, masked/non-finite data will be filtered
        during the fit to each bin/column (along disp. axis) to find the peak.
        If ``omit`` is chosen, columns along disp_axis with any masked/non-finite
        data values will be fully masked (i.e, 2D mask is collapsed to 1D and applied).
        For both options, the input mask (optional on input NDData object) will
        be combined with a mask generated from any non-finite values in the image
        data. Also note that because binning is an option in FitTrace, that masked
        data will contribute zero to the sum when binning adjacent columns.
        [default: ``filter``]

    """
    bins: int | None = None
    guess: float | None = None
    window: int | None = None
    trace_model: Model = field(default=models.Polynomial1D(degree=1))
    peak_method: Literal['gaussian', 'centroid', 'max'] = 'max'
    _crossdisp_axis: int = 0
    _disp_axis: int = 1
    mask_treatment: Literal['filter', 'omit'] = 'filter'
    _valid_mask_treatment_methods = ('filter', 'omit')
    # for testing purposes only, save bin peaks if requested
    _save_bin_peaks_testing: bool = False

    def __post_init__(self):

        # Parse image, including masked/nonfinite data handling based on
        # choice of `mask_treatment`. returns a Spectrum1D
        if self.mask_treatment not in self._valid_mask_treatment_methods:
            raise ValueError("`mask_treatment` must be one of "
                             f"{self._valid_mask_treatment_methods}")

        self.image = self._parse_image(self.image, disp_axis=self._disp_axis,
                                       mask_treatment=self.mask_treatment)

        # _parse_image returns a Spectrum1D. convert this to a masked array
        # for ease of calculations here (even if there is no masked data).
        # Note: uncertainties are dropped, this should also be addressed at
        # some point probably across the package.
        img = np.ma.masked_array(self.image.data, self.image.mask)
        self._mask_temp = self.image.mask

        # validate input arguments
        valid_peak_methods = ('gaussian', 'centroid', 'max')
        if self.peak_method not in valid_peak_methods:
            raise ValueError(f"peak_method must be one of {valid_peak_methods}")

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

        # fit the trace
        self._fit_trace(img)

    def _fit_trace(self, img):

        yy = np.arange(img.shape[self._crossdisp_axis])

        # set max peak location by user choice or wavelength with max avg flux
        ztot = img.mean(axis=self._disp_axis)
        peak_y = self.guess if self.guess is not None else ztot.argmax()
        # NOTE: peak finder can be bad if multiple objects are on slit

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

            fitter = fitting.DogBoxLSQFitter()
            popt_tot = fitter(profile, yy, ztot)

        # restrict fit to window (if one exists)
        ilum2 = (yy if self.window is None
                 else yy[np.arange(peak_y - self.window,
                                   peak_y + self.window, dtype=int)])

        # check if everything in window region is masked
        if img[ilum2].mask.all():
            raise ValueError('All pixels in window region are masked. Check '
                             'for invalid values or use a larger window value.')

        x_bins = np.linspace(0, img.shape[self._disp_axis],
                             self.bins + 1, dtype=int)
        y_bins = np.tile(np.nan, self.bins)

        warn_bins = []
        for i in range(self.bins):

            # binned columns, averaged along disp. axis.
            # or just a single, unbinned column if no bins
            z_i = img[ilum2, x_bins[i]:x_bins[i + 1]].mean(axis=self._disp_axis)

            # if this bin is fully masked, set bin peak to NaN so it can be
            # filtered in the final fit to all bin peaks for the trace
            if z_i.mask.all():
                warn_bins.append(i)
                y_bins[i] = np.nan
                continue

            if self.peak_method == 'gaussian':

                peak_y_i = ilum2[z_i.argmax()]

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
                popt_i = fitter(profile_i, ilum2[~z_i.mask], z_i.data[~z_i.mask])

                # if gaussian fits off chip, then fall back to previous answer
                if not ilum2.min() <= popt_i.mean_0 <= ilum2.max():
                    y_bins[i] = popt_tot.mean_0.value
                else:
                    y_bins[i] = popt_i.mean_0.value
                    popt_tot = popt_i

            elif self.peak_method == 'centroid':
                z_i_cumsum = np.cumsum(z_i)
                # find the interpolated index where the cumulative array reaches
                # half the total cumulative values
                y_bins[i] = np.interp(z_i_cumsum[-1] / 2., z_i_cumsum, ilum2)

                # NOTE this reflects current behavior, should eventually be changed
                # to set to nan by default (or zero fill / interpoate option once
                # available)

            elif self.peak_method == 'max':
                # TODO: implement smoothing with provided width
                y_bins[i] = ilum2[z_i.argmax()]

                # NOTE: a fully masked should eventually be changed to set to
                # nan by default (or zero fill / interpoate option once available)

        # warn about fully-masked bins
        if len(warn_bins) > 0:

            # if there are a ton of bins, we don't want to print them all out
            if len(warn_bins) > 20:
                warn_bins = warn_bins[0: 10] + ['...'] + [warn_bins[-1]]

            warnings.warn(f"All pixels in {'bins' if len(warn_bins) else 'bin'} "
                          f"{', '.join([str(x) for x in warn_bins])}"
                          " are fully masked. Setting bin"
                          f" peak{'s' if len(warn_bins) else ''} to NaN.")

        # recenter bin positions
        x_bins = (x_bins[:-1] + x_bins[1:]) / 2

        # interpolate the fitted trace over the entire wavelength axis

        # for testing purposes only, save bin peaks if requested
        if self._save_bin_peaks_testing:
            self._bin_peaks_testing = (x_bins, y_bins)

        # filter non-finite bin peaks before filtering to all bin peaks
        y_finite = np.where(np.isfinite(y_bins))[0]
        if y_finite.size > 0:
            x_bins = x_bins[y_finite]
            y_bins = y_bins[y_finite]

            # use given model to bin y-values; interpolate over all wavelengths
            if isinstance(self.trace_model, models.Spline1D):
                fitter = fitting.SplineSmoothingFitter()
            elif self.trace_model.linear:
                fitter = fitting.LinearLSQFitter()
            else:
                fitter = fitting.LMLSQFitter()

            self._y_bins = y_bins
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
