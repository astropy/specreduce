# Licensed under a 3-clause BSD style license - see ../../licenses/LICENSE.rst

import numpy as np

from photutils.datasets import apply_poisson_noise

import astropy.units as u
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma

from specreduce.calibration_data import load_pypeit_calibration_lines


def make_2dspec_image(
    nx=3000,
    ny=1000,
    background=5,
    trace_center=None,
    trace_order=3,
    trace_coeffs={'c0': 0, 'c1': 50, 'c2': 100},
    source_amplitude=10,
    source_alpha=0.1
):
    """
    Create synthetic 2D spectroscopic image with a single source. The spatial (y-axis) position
    of the source along the dispersion (x-axis) direction is modeled using a Chebyshev polynomial.
    The flux units are counts and the noise is modeled as Poisson.

    Parameters
    ----------
    nx : int (default=3000)
        Size of image in X axis which is assumed to be the dispersion axis
    ny : int (default=1000)
        Size of image in Y axis which is assumed to be the spatial axis
    background : int (default=5)
        Level of constant background in counts
    trace_center : int (default=None)
        Zeropoint of the trace. If None, then use center of Y (spatial) axis.
    trace_order : int (default=3)
        Order of the Chebyshev polynomial used to model the source's trace
    trace_coeffs : dict (default={'c0': 0, 'c1': 50, 'c2': 100})
        Dict containing the Chebyshev polynomial coefficients to use in the trace model
    source_amplitude : int (default=10)
        Amplitude of modeled source in counts
    source_alpha : float (default=0.1)
        Power index of the source's Moffat profile. Use small number here to emulate
        extended source.

    Returns
    -------
    ccd_im : `~astropy.nddata.CCDData`
        CCDData instance containing synthetic 2D spectroscopic image
    """
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    profile = models.Moffat1D()
    profile.amplitude = source_amplitude
    profile.alpha = source_alpha

    if trace_center is None:
        trace_center = ny / 2

    trace_mod = models.Chebyshev1D(degree=trace_order, **trace_coeffs)
    trace = yy - trace_center + trace_mod(xx/nx)
    z = background + profile(trace)
    noisy_image = apply_poisson_noise(z)

    ccd_im = CCDData(noisy_image, unit=u.count)

    return ccd_im


def make_2d_arc_image(
    nx=3000,
    ny=1000,
    wcs=None,
    extent=[3500, 7000],
    wave_unit=u.Angstrom,
    background=5,
    line_fwhm=5.,
    linelists=['HeI'],
    amplitude_scale=1.,
    tilt_func=None
):
    """
    Create synthetic 2D spectroscopic image of reference emission lines, e.g. a calibration arc lamp. Currently,
    linelists from ``pypeit`` are supported and are selected by string or list of strings that is passed to
    `~specreduce.calibration_data.load_pypeit_calibration_lines`. If a ``wcs`` is not provided, one is created
    using ``extent`` and ``wave_unit`` with dispersion along the X axis.

    Parameters
    ----------
    nx : int (default=3000)
        Size of image in X axis which is assumed to be the dispersion axis
    ny : int (default=1000)
        Size of image in Y axis which is assumed to be the spatial axis
    wcs : `~astropy.wcs.WCS` instance or None (default: None)
        2D WCS to apply to the image. Must have a spectral axis defined along with appropriate spectral wavelength units.
    extent : 2-element list-like
        If ``wcs`` is not provided, this defines the beginning and end wavelengths of the dispersion axis.
    wave_unit : `~astropy.unit.Quantity`
        If ``wcs`` is not provides, this defines the wavelength units of the dispersion axis.
    background : int (default=5)
        Level of constant background in counts
    line_fwhm : float (default=5)
        Gaussian FWHM of the lines in pixels
    linelists : str or list of str (default: ['HeI'])
        Specification for linelists to load from ``pypeit``
    amplitude_scale : float (default: 1)
        Scale factor to apply to amplitudes provides in the linelists
    tilt_func : `~astropy.modeling.polynomial.Legendre1D` or `~astropy.modeling.polynomial.Chebyshev1D`
        The tilt function to apply along the cross-dispersion axis to simulate tilted or curved emission lines.

    Returns
    -------
    ccd_im : `~astropy.nddata.CCDData`
        CCDData instance containing synthetic 2D spectroscopic image
    """
    if wcs is None:
        if extent is None:
            raise ValueError("Must specify either a wavelength extent or a WCS.")
        if len(extent) != 2:
            raise ValueError("Wavelength extent must be of length 2.")
        wcs = WCS(naxis=2)
        wcs.wcs.ctype[0] = 'WAVE'
        wcs.wcs.ctype[1] = 'PIXEL'
        wcs.wcs.cunit[0] = wave_unit
        wcs.wcs.cunit[1] = u.pixel
        wcs.wcs.crval[0] = extent[0]
        wcs.wcs.cdelt[0] = (extent[1] - extent[0]) / nx
        wcs.wcs.crval[1] = 0
        wcs.wcs.cdelt[1] = 1
    else:
        if not wcs.has_spectral:
            raise ValueError("Provided WCS must have a spectral axis.")
        if wcs.naxis != 2:
            raise ValueError("WCS must have NAXIS=2 for a 2D image.")

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)

    is_spectral = [a['coordinate_type'] == "spectral" for a in wcs.get_axis_types()]
    if is_spectral[0]:
        disp_axis = 0
    else:
        disp_axis = 1

    if tilt_func is not None:
        if not isinstance(tilt_func, (models.Legendre1D, models.Chebyshev1D)):
            raise ValueError("The only tilt functions currently supported are Legendre1D and Chebyshev1D from astropy.models.")

        if disp_axis == 0:
            xx = xx + tilt_func((yy - ny/2)/ny)
        else:
            yy = yy + tilt_func((xx - nx/2)/nx)

    linelist = load_pypeit_calibration_lines(linelists)
    line_disp_positions = wcs.spectral.world_to_pixel(linelist['wave'])

    z = background + np.zeros((ny, nx))

    line_sigma = gaussian_fwhm_to_sigma * line_fwhm
    for line_pos, ampl in zip(line_disp_positions, linelist['amplitude']):
        line_mod = models.Gaussian1D(
            amplitude=ampl * amplitude_scale,
            mean=line_pos,
            stddev=line_sigma
        )
        if disp_axis == 0:
            z += line_mod(xx)
        else:
            z += line_mod(yy)

    noisy_image = apply_poisson_noise(z)

    ccd_im = CCDData(noisy_image, unit=u.count, wcs=wcs)

    return ccd_im
