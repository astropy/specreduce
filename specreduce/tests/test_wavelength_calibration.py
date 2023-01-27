import numpy as np
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.exceptions import AstropyUserWarning
from specutils import Spectrum1D

from specreduce.wavelength_calibration import CalibrationLine, WavelengthCalibration1D

def test_linear_from_list():
    np.random.seed(7)
    flux = np.random.random(50)*u.Jy
    sa = np.arange(0,50)*u.pix
    spec = Spectrum1D(flux, spectral_axis = sa)
    test = WavelengthCalibration1D(spec, [(5000*u.AA, 0),(5100*u.AA, 10),(5198*u.AA, 20),(5305*u.AA, 30)])
    with pytest.warns(AstropyUserWarning, match="Model is linear in parameters"):
        spec2 = test.apply_to_spectrum(spec)
    
    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_linear_from_calibrationline():
    np.random.seed(7)
    flux = np.random.random(50)*u.Jy
    sa = np.arange(0,50)*u.pix
    spec = Spectrum1D(flux, spectral_axis = sa)
    lines = [CalibrationLine(spec, 5000*u.AA, 0), CalibrationLine(spec, 5100*u.AA, 10),
             CalibrationLine(spec, 5198*u.AA, 20), CalibrationLine(spec, 5305*u.AA, 30)]
    test = WavelengthCalibration1D(spec, lines)
    with pytest.warns(AstropyUserWarning, match="Model is linear in parameters"):
        spec2 = test.apply_to_spectrum(spec)
    
    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)