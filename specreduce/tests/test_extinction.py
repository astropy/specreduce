import numpy as np
import pytest
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

from specreduce.calibration_data import (
    AtmosphericExtinction,
    AtmosphericTransmission,
    SUPPORTED_EXTINCTION_MODELS
)


@pytest.mark.remote_data
def test_supported_models():
    """
    Test loading of supported models
    """
    for model in SUPPORTED_EXTINCTION_MODELS:
        ext = AtmosphericExtinction(model=model)
        assert len(ext.extinction_mag) > 0
        assert len(ext.transmission) > 0


@pytest.mark.remote_data
def test_custom_mag_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = u.Magnitude(1. / wave, u.MagUnit(u.dimensionless_unscaled))
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert len(ext.extinction_mag) > 0
    assert len(ext.transmission) > 0


@pytest.mark.remote_data
def test_custom_raw_mag_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave * u.mag
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert len(ext.extinction_mag) > 0
    assert len(ext.transmission) > 0


@pytest.mark.remote_data
def test_custom_linear_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave * u.dimensionless_unscaled
    ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)
    assert len(ext.extinction_mag) > 0
    assert len(ext.transmission) > 0


@pytest.mark.remote_data
def test_missing_extinction_unit():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave
    with pytest.warns(AstropyUserWarning):
        ext = AtmosphericExtinction(extinction=extinction, spectral_axis=wave * u.um)

    assert len(ext.extinction_mag) > 0
    assert len(ext.transmission) > 0


@pytest.mark.remote_data
def test_transmission_model():
    """
    Test creating of default atmospheric transmission model
    """
    ext = AtmosphericTransmission()
    assert len(ext.transmission) > 0

    with pytest.warns(RuntimeWarning):
        assert len(ext.extinction_mag) > 0
