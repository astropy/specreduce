import numpy as np
from numpy.testing import assert_allclose
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.exceptions import AstropyUserWarning
from specutils import Spectrum1D

from specreduce.wavelength_calibration import CalibrationLine, WavelengthCalibration1D

def test_linear_from_list(spec1d):
    test = WavelengthCalibration1D(spec1d, [(5000*u.AA, 0), (5100*u.AA, 10),
                                          (5198*u.AA, 20), (5305*u.AA, 30)])
    with pytest.warns(AstropyUserWarning, match="Model is linear in parameters"):
        spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_linear_from_calibrationline(spec1d):
    lines = [CalibrationLine(spec1d, 5000*u.AA, 0), CalibrationLine(spec1d, 5100*u.AA, 10),
             CalibrationLine(spec1d, 5198*u.AA, 20), CalibrationLine(spec1d, 5305*u.AA, 30)]
    test = WavelengthCalibration1D(spec1d, lines)
    with pytest.warns(AstropyUserWarning, match="Model is linear in parameters"):
        spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)

def test_calibrationline(spec1d_with_emission_line, spec1d_with_absorption_line):
    with pytest.raises(ValueError, match="You must define 'range' in refinement_kwargs"):
        line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128,
                               refinement_method='gaussian')

    with pytest.raises(ValueError, match="You must define 'direction' in refinement_kwargs"):
        line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128,
                               refinement_method='gradient')

    line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128, refinement_method='gaussian',
                           refinement_kwargs={'range': 25})
    assert_allclose(line.refine(), 129.44371)

    line2 = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 130, refinement_method='max',
                            refinement_kwargs={'range': 10})
    assert line2.refine() == 128

    line3 = CalibrationLine(spec1d_with_absorption_line, 5000*u.AA, 128, refinement_method='min',
                            refinement_kwargs={'range': 10})
    assert line3.refine() == 130