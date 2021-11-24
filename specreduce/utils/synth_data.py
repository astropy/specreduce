# Licensed under a 3-clause BSD style license - see ../../licenses/LICENSE.rst

import numpy as np

from photutils.datasets import apply_poisson_noise

import astropy.units as u
from astropy.modeling import models
from astropy.nddata import CCDData


def make_2dspec_image(
    nx = 3000,
    ny = 1000,
    background = 5,
    trace_center = None,
    trace_order = 3,
    trace_coeffs = {'c0': 0, 'c1': 50, 'c2': 100},
    source_amplitude = 10,
    source_alpha = 0.1
):
    """
    Create synthetic 2D spectroscopic image with a single source.

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
        Power index of the sources Moffat profile. Use small number here to emulate extended source.

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
