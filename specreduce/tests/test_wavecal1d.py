# specreduce/tests/test_wavecal1d.py
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.polynomial import Polynomial1D
from numpy import array

from specreduce.wavecal1d import WavelengthSolution1D, _diff_poly1d
from specutils import Spectrum1D

ref_pixel = 100

def test_diff_poly1d():
    p = _diff_poly1d(Polynomial1D(3, c0=1.0, c1=2.0, c2=3.0, c3=4.0))
    np.testing.assert_array_equal(p.parameters, [ 2.,  6., 12.])


def test_init_default_values():
    ref_pixel = 100
    wavelength_solution = WavelengthSolution1D(ref_pixel)

    assert wavelength_solution.ref_pixel == ref_pixel
    assert wavelength_solution.unit == u.angstrom
    assert wavelength_solution.degree == 3
    assert wavelength_solution.bounds_pix is None
    assert wavelength_solution.bounds_wav is None
    assert wavelength_solution._cat_lines is None
    assert wavelength_solution._obs_lines is None
    assert wavelength_solution._trees is None
    assert wavelength_solution._fit is None
    assert wavelength_solution._wcs is None
    assert wavelength_solution._p2w is None
    assert wavelength_solution._w2p is None
    assert wavelength_solution._p2w_dldx is None


def test_init_raises_error_for_multiple_sources():
    arc_spectra = Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom)
    obs_lines = [np.array([500.0])]
    with pytest.raises(ValueError, match="Only one of arc_spectra or obs_lines can be provided."):
        WavelengthSolution1D(ref_pixel, arc_spectra=arc_spectra, obs_lines=obs_lines)


def test_init_arc_spectra_validation():
    arc_spectra = Spectrum1D(flux=np.array([[1, 2]]) * u.DN, spectral_axis=np.array([1, 2]) * u.angstrom)
    with pytest.raises(ValueError, match="The arc spectra must be 1 dimensional."):
        WavelengthSolution1D(ref_pixel, arc_spectra=arc_spectra)


def test_init_obs_lines_requires_pixel_bounds():
    obs_lines = [np.array([500.0])]
    with pytest.raises(ValueError, match="Must give pixel bounds when"):
        WavelengthSolution1D(ref_pixel, obs_lines=obs_lines)


def test_init_line_list():
    """Test the catalog line list initialization with various configurations of `arc_spectra` and `line_lists`.
    """
    arc = Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom)
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=['ArI'])
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists='ArI')
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=[array([0.1])])
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=array([0.1]))
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[['ArI'], ['ArI']])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=['ArI', ['ArI']])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=['ArI', 'ArI'])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[array([0.1]), array([0.1])])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[array([0.1, 0.3]), ['ArI']])
    with pytest.raises(ValueError, match="The number of line lists"):
        WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[['ArI']])



# def test_fit_lines_with_valid_input():
#    ref_pixel = 100
#    pix_bounds = (0, 10)
#    pixels = [2, 4, 6, 8]
#    wavelengths = [500, 600, 700, 800]

#    wavelength_solution = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)

#    wavelength_solution.fit_lines(pixels=pixels, wavelengths=wavelengths)

#    assert wavelength_solution._p2w is not None
#    assert wavelength_solution._p2w[1].degree == wavelength_solution.degree




def test_fit_lines_raises_error_for_mismatched_sizes():
    ref_pixel = 100
    pix_bounds = (0, 10)
    pixels = [2, 4, 6]
    wavelengths = [500, 600, 700, 800]

    wavelength_solution = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)

    with pytest.raises(ValueError, match="The sizes of pixel and wavelength arrays must match."):
        wavelength_solution.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_insufficient_lines():
    ref_pixel = 100
    pix_bounds = (0, 10)
    pixels = [5]
    wavelengths = [500]

    wavelength_solution = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)

    with pytest.raises(ValueError, match="Need at least two lines for a fit"):
        wavelength_solution.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_missing_pixel_bounds():
    ref_pixel = 100
    pixels = [2, 4, 6, 8]
    wavelengths = [500, 600, 700, 800]

    wavelength_solution = WavelengthSolution1D(ref_pixel)

    with pytest.raises(ValueError, match="Cannot fit without pixel bounds set."):
        wavelength_solution.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_find_lines_with_valid_input(mocker):
    ref_pixel = 100
    arc_spectra = [Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom)]
    wavelength_solution = WavelengthSolution1D(ref_pixel, arc_spectra=arc_spectra)

    mock_find_arc_lines = mocker.patch("specreduce.wavecal1d.find_arc_lines")
    mock_find_arc_lines.return_value = {"centroid": np.array([5.0]) * u.angstrom}

    wavelength_solution.find_lines(fwhm=2.0, noise_factor=1.5)

    assert wavelength_solution._obs_lines is not None
    assert len(wavelength_solution._obs_lines) == len(arc_spectra)
    assert mock_find_arc_lines.called_once_with(arc_spectra[0], 2.0, noise_factor=1.5)


def test_find_lines_with_missing_arc_spectra():
    ref_pixel = 100
    wavelength_solution = WavelengthSolution1D(ref_pixel)

    with pytest.raises(ValueError, match="Must provide arc spectra to find lines."):
        wavelength_solution.find_lines(fwhm=2.0, noise_factor=1.5)

