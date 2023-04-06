from numpy.testing import assert_allclose

from astropy.table import QTable
import astropy.units as u
from astropy.modeling.models import Polynomial1D
from astropy.tests.helper import assert_quantity_allclose

from specreduce import WavelengthCalibration1D


def test_linear_from_list(spec1d):
    centers = [0, 10, 20, 30]
    w = [5000, 5100, 5198, 5305]*u.AA
    test = WavelengthCalibration1D(spec1d, centers, line_wavelengths=w)
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_linear_from_table(spec1d):
    centers = [0, 10, 20, 30]
    w = [5000, 5100, 5198, 5305]*u.AA
    table = QTable([centers, w], names=["pixel_center", "wavelength"])
    test = WavelengthCalibration1D(spec1d, table)
    spec2 = test.apply_to_spectrum(spec1d)

    assert_quantity_allclose(spec2.spectral_axis[0], 4998.8*u.AA)
    assert_quantity_allclose(spec2.spectral_axis[-1], 5495.169999*u.AA)


def test_poly_from_table(spec1d):
    # This test is mostly to prove that you can use other models
    centers = [0, 10, 20, 30, 40]
    w = [5005, 5110, 5214, 5330, 5438]*u.AA
    table = QTable([centers, w], names=["pixel_center", "wavelength"])

    test = WavelengthCalibration1D(spec1d, table, model=Polynomial1D(2))
    test.apply_to_spectrum(spec1d)

    assert_allclose(test.model.parameters, [5.00477143e+03, 1.03457143e+01, 1.28571429e-02])


def test_replace_spectrum(spec1d, spec1d_with_emission_line):
    centers = [0, 10, 20, 30]*u.pix
    w = [5000, 5100, 5198, 5305]*u.AA
    test = WavelengthCalibration1D(spec1d, centers, line_wavelengths=w)
    # Accessing this property causes fits the model and caches the resulting WCS
    test.wcs
    assert "wcs" in test.__dict__

    # Replace the input spectrum, which should clear the cached properties
    test.input_spectrum = spec1d_with_emission_line
    assert "wcs" not in test.__dict__
