import numpy as np

import astropy.units as u

from ..calibration_data import AtmosphericExtinction, SUPPORTED_EXTINCTION_MODELS


def test_supported_models():
    """
    Test loading of supported models
    """
    for model in SUPPORTED_EXTINCTION_MODELS:
        ext = AtmosphericExtinction(model=model)
        assert(len(ext.extinction) > 0)


def test_custom_model():
    """
    Test creation of custom model from Quantity arrays
    """
    wave = np.linspace(0.3, 2.0, 50)
    extinction = 1. / wave
    ext = AtmosphericExtinction(extinction=extinction * u.mag, spectral_axis=wave * u.um)
    assert(len(ext.extinction) > 0)
