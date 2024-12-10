from numbers import Number
from typing import Literal

import numpy as np
from astropy.nddata import VarianceUncertainty, NDData
from specutils import Spectrum1D

from specreduce.core import _ImageParser
from specreduce.tracing import Trace, FlatTrace
from specreduce.extract import _ap_weight_image, _align_along_trace

__all__ = ['measure_cross_dispersion_profile', '_align_along_trace', 'align_2d_spectrum_along_trace']


def _get_image_ndim(image):
    if isinstance(image, np.ndarray):
        return image.ndim
    elif isinstance(image, NDData):
        return image.data.ndim
    else:
        raise ValueError('Unrecognized image data format.')


def align_2d_spectrum_along_trace(image: NDData | np.ndarray,
                                  trace: Trace | np.ndarray | Number,
                                  method: Literal['interpolate', 'shift'] = 'interpolate',
                                  disp_axis: int = 1) -> Spectrum1D:
    """
    Align a 2D spectrum image along a trace either with an integer or sub-pixel precision.

    This function rectifies a 2D spectrum by aligning its cross-dispersion profile along a given
    trace. The function also updates the mask to reflect alignment operations and propagates
    uncertainties when using the 'interpolate' method.  The rectification process can use either
    sub-pixel precision through interpolation or integer shifts for simplicity. The method assumes
    the input spectrum is rectilinear, meaning the dispersion direction and spatial direction are
    aligned with the pixel grid.

    Parameters
    ----------
    image
        The 2D image to align.
    trace
        Either a ``Trace`` object, a 1D ndarray, or a single value that defines the center
        of the cross-dispersion profile.
    method
        The method used to align the image: ``interpolate`` aligns the image
        with a sub-pixel precision using linear interpolation while ``shift``
        aligns the image using integer shifts.
    disp_axis
        The index of the image's dispersion axis. [default: 1]

    Returns
    -------
    Spectrum1D
        A rectified version of the image aligned along the specified trace.

    Notes
    -----
    - This function is intended only for rectilinear spectra, where the dispersion
      and spatial axes are already aligned with the image grid. Non-rectilinear spectra
      require additional pre-processing (e.g., geometric rectification) before using
      this function.
    """
    if _get_image_ndim(image) > 2:
        raise ValueError('The number of image dimensions must be 2.')
    if not (0 <= disp_axis <= 1):
        raise ValueError('Displacement axis must be either 0 or 1.')

    if isinstance(trace, Trace):
        trace = trace.trace.data
    elif isinstance(trace, (np.ndarray, Number)):
        pass
    else:
        raise ValueError('Unrecognized trace format.')

    image = _ImageParser()._parse_image(image, disp_axis=disp_axis)
    data = image.data
    mask = image.mask | ~np.isfinite(data)
    ucty = image.uncertainty.represent_as(VarianceUncertainty).array

    if disp_axis == 0:
        data = data.T
        mask = mask.T
        ucty = ucty.T

    n_rows = data.shape[0]
    n_cols = data.shape[1]

    rows = np.broadcast_to(np.arange(n_rows)[:, None], data.shape)
    cols = np.broadcast_to(np.arange(n_cols), data.shape)

    if method == 'interpolate':
        # Calculate the real and integer shifts
        # and the interpolation weights.
        shifts = trace - n_rows / 2.0
        k = np.floor(shifts).astype(int)
        a = shifts - k

        # Calculate the shifted indices and mask the
        # edge pixels without information.
        ix1 = rows + k
        ix2 = ix1 + 1
        shift_mask = (ix1 < 0) | (ix1 > n_rows - 2)
        ix1 = np.clip(ix1, 0, n_rows - 1)
        ix2 = np.clip(ix2, 0, n_rows - 1)

        # Shift the data, uncertainties, and the mask using linear
        # interpolation.
        data_r = (1.0-a)*data[ix1, cols] + a*data[ix2, cols]
        ucty_r = (1.0-a)**2*ucty[ix1, cols] + a**2*ucty[ix2, cols]
        mask_r = mask[ix1, cols] | mask[ix2, cols] | shift_mask

    elif method == 'shift':
        shifts = trace.astype(int) - n_rows // 2
        ix = rows + shifts
        shift_mask = (ix < 0) | (ix > n_rows - 1)
        ix = np.clip(ix, 0, n_rows - 1)

        data_r = data[ix, cols]
        ucty_r = ucty[ix, cols]
        mask_r = mask[ix, cols] | shift_mask

    else:
        raise ValueError("method must be either 'interpolate' or 'shift'.")

    if disp_axis == 0:
        data_r = data_r.T
        mask_r = mask_r.T
        ucty_r = ucty_r.T

    return Spectrum1D(data_r * image.unit, mask=mask_r, meta=image.meta,
                      uncertainty=VarianceUncertainty(ucty_r).represent_as(image.uncertainty))


def measure_cross_dispersion_profile(image, trace=None, crossdisp_axis=0,
                                     width=None, pixel=None, pixel_range=None,
                                     statistic='median', align_along_trace=True):
    """
    Return a 1D (quantity) array of the cross-dispersion profile measured at a
    specified pixel value ('wavelength', but potentially before calibration),
    or the average profile across several pixel values (or range of pixel values)
    along the dispersion axis.

    If a single number is specified for `pixel`, then the profile at that pixel
    (i.e wavelength) will be returned. If several pixels are specified in a list or
    array, then they will be averaged (median or mean, set by `statistic` which
    defaults to median). Alternatively, `pixel_range` can be specified as a tuple
    of integers specifying the minimum and maximum pixel range to average the
    profile. `pixel` and `pixel_range` cannot be set simultaneously, and the default
    is `pixel_range`=(min_image, max_image) to return an average profile across the
    entire wavelength range of the image.

    The window in the cross dispersion axis for measuring the profile at
    the pixel(s) specified is determined by `width` and `trace`. If a trace is
    provided (either as a Trace object, or as a number specifying the location
    on the cross-dispersion axis of a FlatTrace object) that will determine the
    center of the profile on the cross-dispersion axis. Otherwise, if `trace`
    is None, the center of the image will be the center of the returned profile
    (i.e., the center row assuming a horizontal trace).

    If `width` is none, a window size of half the image in the cross-dispersion
    axis will be used to measure the cross dispersion profile. Otherwise, an
    integer value can be set for `width` which will determine the size of the
    window around the trace used to measure the profile (this window moves with
    the trace if trace is not flat).

    By default, if a non-flat trace is used the image will be aligned along the
    trace. This can be controlled with the 'align_along_trace' parameter.

    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like, required
        2D image to measure cross-dispersion profile.
    trace : Trace object, int, float, or None
        A Trace object, a number to specify a FlatTrace, or None to use the
        middle of the image. This is the position that defines the center of the
        cross-dispersion profile. [default: None]
    crossdisp_axis : int, optional
        The index of the image's cross-dispersion axis. [default: 0]
    width : tuple of int or None
        Width around 'trace' to calculate profile. If None, then all rows in the
        cross-dispersion axis will be used. [default: None]
    pixel: int, list of int, or None
        Pixel value(s) along the dispersion axis to return cross-dispersion profile.
        If several specified in list, then the average (method set by `statistic`)
        profile will be calculated. If None, and `pixel_range` is set, then
        `pixel_range` will be used. [default: None]
    pixel_range: tuple of int or None
        Tuple of (min, max) defining the pixel range (along dispersion axis) to
        calculate average cross-dispersion profile, up to and not inclusive of max.
        If None, and `pixel` is not None, `pixel` will be used. If None and `pixel`
        is also None, this will be interpreted as using the entire dispersion axis
        to generate an average profile for the whole image. [default: None]
    statistic: 'median' or 'mean'
        If `pixel` specifies multiple pixels, or `pixel_range` is specified, an
        average profile will be returned. This can be either `median` (default)
        or `mean`. This is ignored if only one pixel is specified. [default: median]
    align_along_trace: bool
        Relevant only for non-flat traces. If True, "roll" each column so that
        the trace sits in the central row before calculating average profile. This
        will prevent any 'blurring' from averaging a non-flat trace at different
        pixel/wavelengths. [default: True]

    """

    if crossdisp_axis != 0 and crossdisp_axis != 1:
        raise ValueError('`crossdisp_axis` must be 0 or 1.')
    crossdisp_axis = int(crossdisp_axis)
    disp_axis = 1 if crossdisp_axis == 0 else 0

    unit = getattr(image, 'unit', None)

    # parse image, which will return a spectrum1D (note: this is not ideal,
    # but will be addressed at some point)
    parser = _ImageParser()
    image = parser._parse_image(image, disp_axis=disp_axis)

    # which we then need to make back into a masked array
    # again this way of parsing the image is not ideal but
    # thats just how it is for now.
    image = np.ma.MaskedArray(image.data, mask=image.mask)

    # transpose if disp_axis = 0 just for simplicity of calculations
    # image is already copied so this won't modify input
    if disp_axis == 0:
        image = image.T

    nrows = image.shape[crossdisp_axis]
    ncols = image.shape[disp_axis]

    if not isinstance(trace, Trace):  # `trace` can be a trace obj
        if trace is None:  # if None, make a FlatTrace in the center of image
            trace_pos = nrows / 2
            trace = FlatTrace(image, trace_pos)
        elif isinstance(trace, (float, int)):  # if float/int make a FlatTrace
            trace = FlatTrace(image, trace)
        else:
            raise ValueError('`trace` must be Trace object, number to specify '
                             'the location of a FlatTrace, or None to use center'
                             ' of image.')

    if statistic not in ['median', 'mean']:
        raise ValueError("`statistic` must be 'median' or 'mean'.")

    # determine if there is one pixel/wavelength selected or many as either a
    # list or a tuple to specify a range
    if pixel is not None:
        if pixel_range is not None:
            raise ValueError('Both `pixel` and `pixel_range` can not be set'
                             ' simultaneously.')
        if isinstance(pixel, (int, float)):
            pixels = np.array([int(pixel)])
        elif np.all([isinstance(x, (int, float)) for x in pixel]):
            pixels = np.array([int(x) for x in pixel])
        else:
            raise ValueError('`pixels` must be an integer, or list of integers '
                             'to specify where the crossdisperion profile should '
                             'be measured.')
    else:  # range is specified
        if pixel_range is None:
            pixels = np.arange(0, ncols)
        else:  # if not None, it should be a lower and upper bound
            if len(pixel_range) != 2:
                raise ValueError('`pixel_range` must be a tuple of integers.')
            pixels = np.arange(min(pixel_range), max(pixel_range))

    # now that we have all pixels that should be included in the profile, make
    # sure that they are within image bounds.
    # note: Should just warn instead and clip out out-of-bounds pixels, and only
    # warn if there are none left?
    if np.any(pixels < 0) or np.any(pixels > ncols - 1):
        raise ValueError('Pixels chosen to measure cross dispersion profile are'
                         ' out of image bounds.')

    # now that we know which pixel(s) on the disp. axis we want to include
    # figure out the range/window of pixels along the crossdisp axis to measure
    # the profile
    if width is None:  # if None, use all rows
        width = nrows
    elif isinstance(width, (float, int)):
        width = int(width)
    else:
        raise ValueError('`width` must be an integer, or None to use all '
                         'cross-dispersion pixels.')

    # rectify trace, if _align_along_trace is True and trace is not flat
    aligned_trace = None
    if align_along_trace:
        if not isinstance(trace, FlatTrace):
            # note: img was transposed according to `crossdisp_axis`: disp_axis will always be 1
            aligned_trace = _align_along_trace(image, trace.trace,
                                               disp_axis=1,
                                               crossdisp_axis=0)

            # new trace will be a flat trace at the center of the image
            trace_pos = nrows / 2
            trace = FlatTrace(aligned_trace, trace_pos)

    # create a weight image based on the trace and 'width' to mask around trace

    if width == nrows:
        wimg = np.zeros(image.shape)
    else:
        wimg = _ap_weight_image(trace, width, disp_axis, crossdisp_axis, image.shape)
        # invert mask to include, not exclude, pixels around trace
        wimg = (1 - wimg).astype(int)

    # now that we have figured out the mask for the window in cross-disp. axis,
    # select only the pixel(s) we want to include in measuring the avg. profile
    pixel_mask = np.ones(image.shape)
    pixel_mask[:, pixels] = 0

    # combine these masks to isolate the rows and cols used to measure profile
    combined_mask = np.logical_or(pixel_mask, wimg)

    if aligned_trace is not None:
        masked_arr = np.ma.MaskedArray(aligned_trace, combined_mask)
    else:
        masked_arr = np.ma.MaskedArray(image.data, combined_mask)

    # and measure the cross dispersion profile. if multiple pixels/wavelengths,
    # this will be an average. we already transposed data based on disp_axis so
    # axis is always 1 for this calculation
    if statistic == 'mean':
        avg_prof = np.ma.mean(masked_arr, axis=1)
    else:  # must be median, we already checked.
        avg_prof = np.ma.median(masked_arr, axis=1)

    # and get profile
    avg_prof = avg_prof.data[~avg_prof.mask]

    # and re-apply original unit, if there was one
    if unit is not None:
        avg_prof *= unit

    return avg_prof
