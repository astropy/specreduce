import numpy as np

import astropy.units as u

from ..calibration_data import (
    AtmosphericExtinction,
    AtmosphericIRExtinction,
    SUPPORTED_EXTINCTION_MODELS
)


def test_supported_models():
    """
    Test loading of supported models
    """
    for model in SUPPORTED_EXTINCTION_MODELS:
        ext = AtmosphericExtinction(model=model)
        assert(len(ext.extinction_mag) > 0)
        assert(len(ext.extinction_frac) > 0)


def test_custom_mag_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = u.Magnitude(1. / wave, u.MagUnit(u.dimensionless_unscaled))
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert(len(ext.extinction_mag) > 0)
    assert(len(ext.extinction_frac) > 0)


def test_custom_linear_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave * u.dimensionless_unscaled
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert(len(ext.extinction_mag) > 0)
    assert(len(ext.extinction_frac) > 0)


def test_missing_extinction_unit():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert(len(ext.extinction_mag) > 0)
    assert(len(ext.extinction_frac) > 0)


def test_ir_extinction():
    ext = AtmosphericIRExtinction()
    assert(len(ext.extinction_mag) > 0)
    assert(len(ext.extinction_frac) > 0)
