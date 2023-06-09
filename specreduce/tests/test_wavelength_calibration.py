from numpy.testing import assert_allclose
import numpy as np
import pytest

from astropy.table import QTable
import astropy.units as u
from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LinearLSQFitter
from astropy.tests.helper import assert_quantity_allclose

from specreduce import WavelengthCalibration1D


def test_linear_from_list(spec1d):
    centers = [0, 10, 20, 30]
    w = [5000, 5100, 5198, 5305]*u.AA
    test = WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w)
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_wavelength_from_table(spec1d):
    centers = [0, 10, 20, 30]
    w = [5000, 5100, 5198, 5305]*u.AA
    table = QTable([w], names=["wavelength"])
    WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=table)


def test_linear_from_table(spec1d):
    centers = [0, 10, 20, 30]
    w = [5000, 5100, 5198, 5305]*u.AA
    table = QTable([centers, w], names=["pixel_center", "wavelength"])
    test = WavelengthCalibration1D(spec1d, matched_line_list=table)
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_poly_from_table(spec1d):
    # This test is mostly to prove that you can use other models
    centers = [0, 10, 20, 30, 40]
    w = [5005, 5110, 5214, 5330, 5438]*u.AA
    table = QTable([centers, w], names=["pixel_center", "wavelength"])

    test = WavelengthCalibration1D(spec1d, matched_line_list=table,
                                   input_model=Polynomial1D(2), fitter=LinearLSQFitter())
    test.apply_to_spectrum(spec1d)

    assert_allclose(test.fitted_model.parameters, [5.00477143e+03, 1.03457143e+01, 1.28571429e-02])


def test_replace_spectrum(spec1d, spec1d_with_emission_line):
    centers = [0, 10, 20, 30]*u.pix
    w = [5000, 5100, 5198, 5305]*u.AA
    test = WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w)
    # Accessing this property causes fits the model and caches the resulting WCS
    test.wcs
    assert "wcs" in test.__dict__

    # Replace the input spectrum, which should clear the cached properties
    test.input_spectrum = spec1d_with_emission_line
    assert "wcs" not in test.__dict__


def test_expected_errors(spec1d):
    centers = [0, 10, 20, 30, 40]
    w = [5005, 5110, 5214, 5330, 5438]*u.AA
    table = QTable([centers, w], names=["pixel_center", "wavelength"])

    with pytest.raises(ValueError, match="Cannot specify line_wavelengths separately"):
        WavelengthCalibration1D(spec1d, matched_line_list=table, line_wavelengths=w)

    with pytest.raises(ValueError, match="must have the same length"):
        w2 = [5005, 5110, 5214, 5330, 5438, 5500]*u.AA
        WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w2)

    with pytest.raises(ValueError, match="astropy.units.Quantity array or"
                                         " as an astropy.table.QTable"):
        w2 = [5005, 5110, 5214, 5330, 5438]
        WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w2)

    with pytest.raises(ValueError, match="specify at least one"):
        WavelengthCalibration1D(spec1d, line_pixels=centers)


def test_fit_residuals(spec1d):
    # test that fit residuals are all 0 when input is perfectly linear and model
    # is a linear model

    centers = np.array([0, 10, 20, 30])
    w = (0.5 * centers + 2) * u.AA
    test = WavelengthCalibration1D(spec1d, line_pixels=centers,
                                   line_wavelengths=w)

    test.apply_to_spectrum(spec1d)  # have to apply for residuals to be computed

    assert_quantity_allclose(test.residuals, 0.*u.AA, atol=1e-07*u.AA)


def test_fit_residuals_access(spec1d):
    # make sure that accessing residuals can be called before wcs/apply_to_spectrum

    centers = np.array([0, 10, 20, 30])
    w = (0.5 * centers + 2) * u.AA
    test = WavelengthCalibration1D(spec1d, line_pixels=centers,
                                   line_wavelengths=w)
    test.residuals
    test.wcs


def test_unsorted_pixels_wavelengths(spec1d):
    # make sure an error is raised if input matched pixels/wavelengths are
    # not strictly increasing or decreasing.

    centers = np.array([0, 10, 5, 30])
    w = (0.5 * centers + 2) * u.AA

    with pytest.raises(ValueError, match='Pixels must be strictly increasing or decreasing.'):
        WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w)

    # now test that it fails when wavelengths are unsorted
    centers = np.array([0, 10, 20, 30])
    w = np.array([2, 5, 6, 1]) * u.AA
    with pytest.raises(ValueError, match='Wavelengths must be strictly increasing or decreasing.'):
        WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=w)

    # and same if those wavelengths are provided in a table
    table = QTable([w], names=["wavelength"])
    with pytest.raises(ValueError, match='Wavelengths must be strictly increasing or decreasing.'):
        WavelengthCalibration1D(spec1d, line_pixels=centers, line_wavelengths=table)
