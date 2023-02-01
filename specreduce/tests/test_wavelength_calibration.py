from numpy.testing import assert_allclose
import pytest

import astropy.units as u
from astropy.modeling.models import Polynomial1D
from astropy.tests.helper import assert_quantity_allclose

from specreduce.wavelength_calibration import CalibrationLine, WavelengthCalibration1D


def test_linear_from_list(spec1d):
    test = WavelengthCalibration1D(spec1d, [(5000*u.AA, 0), (5100*u.AA, 10),
                                            (5198*u.AA, 20), (5305*u.AA, 30)])
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_linear_from_calibrationline(spec1d):
    lines = [CalibrationLine(spec1d, 5000*u.AA, 0), CalibrationLine(spec1d, 5100*u.AA, 10),
             CalibrationLine(spec1d, 5198*u.AA, 20), CalibrationLine(spec1d, 5305*u.AA, 30)]
    test = WavelengthCalibration1D(spec1d, lines)
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_poly_from_calibrationline(spec1d):
    # This test is mostly to prove that you can use other models
    lines = [CalibrationLine(spec1d, 5005*u.AA, 0), CalibrationLine(spec1d, 5110*u.AA, 10),
             CalibrationLine(spec1d, 5214*u.AA, 20), CalibrationLine(spec1d, 5330*u.AA, 30),
             CalibrationLine(spec1d, 5438*u.AA, 40)]
    test = WavelengthCalibration1D(spec1d, lines, model=Polynomial1D(2))
    test.apply_to_spectrum(spec1d)

    assert_allclose(test.model.parameters, [5.00477143e+03, 1.03457143e+01, 1.28571429e-02])


def test_calibrationline(spec1d_with_emission_line, spec1d_with_absorption_line):
    with pytest.raises(ValueError, match="You must define 'range' in refinement_kwargs"):
        line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128,
                               refinement_method='gaussian')

    with pytest.raises(ValueError, match="You must define 'direction' in refinement_kwargs"):
        line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128,
                               refinement_method='gradient')

    line = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 128, refinement_method='gaussian',
                           refinement_kwargs={'range': 25})
    assert_allclose(line.refine(), 129.44371, atol=0.01)

    line2 = CalibrationLine(spec1d_with_emission_line, 5000*u.AA, 130, refinement_method='max',
                            refinement_kwargs={'range': 10})
    assert line2.refine() == 128

    line3 = CalibrationLine(spec1d_with_absorption_line, 5000*u.AA, 128, refinement_method='min',
                            refinement_kwargs={'range': 10})
    assert line3.refine() == 130


def test_replace_spectrum(spec1d, spec1d_with_emission_line):
    lines = [CalibrationLine(spec1d, 5000*u.AA, 0), CalibrationLine(spec1d, 5100*u.AA, 10),
             CalibrationLine(spec1d, 5198*u.AA, 20), CalibrationLine(spec1d, 5305*u.AA, 30)]
    test = WavelengthCalibration1D(spec1d, lines)
    # Accessing this property causes fits the model and caches the resulting WCS
    test.wcs
    assert "wcs" in test.__dict__
    for line in test.lines:
        assert "refined_pixel" in line.__dict__

    # Replace the input spectrum, which should clear the cached properties
    test.input_spectrum = spec1d_with_emission_line
    assert "wcs" not in test.__dict__
    for line in test.lines:
        assert "refined_pixel" not in line.__dict__
