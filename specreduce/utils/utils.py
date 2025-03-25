import numpy as np

from specreduce.core import _ImageParser
from specreduce.tracing import Trace, FlatTrace
from specreduce.extract import _ap_weight_image, _align_along_trace

__all__ = ['measure_cross_dispersion_profile', '_align_along_trace']


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
    array, then they will be averaged (median or average, set by `statistic` which
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
    statistic: 'median' or 'average'
        If `pixel` specifies multiple pixels, or `pixel_range` is specified, an
        average profile will be returned. This can be either `median` (default)
        or `average` (mean). This is ignored if only one pixel is specified.
        [default: median]
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

    if statistic not in ['median', 'average']:
        raise ValueError("`statistic` must be 'median' or 'average'.")

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
        width = int(width)

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
    pixel_mask = np.ones((image.shape))
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
    if statistic == 'average':
        avg_prof = np.ma.mean(masked_arr, axis=1)
    else:  # must be median, we already checked.
        avg_prof = np.ma.median(masked_arr, axis=1)

    # and get profile
    avg_prof = avg_prof.data[~avg_prof.mask]

    # and re-apply original unit, if there was one
    if unit is not None:
        avg_prof *= unit

    return avg_prof
