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
    ws = WavelengthSolution1D(ref_pixel)
    assert ws.ref_pixel == ref_pixel
    assert ws.unit == u.angstrom
    assert ws.degree == 3
    assert ws.bounds_pix is None
    assert ws.bounds_wav is None
    assert ws._cat_lines is None
    assert ws._obs_lines is None
    assert ws._trees is None
    assert ws._fit is None
    assert ws._wcs is None
    assert ws._p2w is None
    assert ws._w2p is None
    assert ws._p2w_dldx is None


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


def test_find_lines_with_valid_input(mocker):
    arc_spectra = [Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom)]
    ws = WavelengthSolution1D(ref_pixel, arc_spectra=arc_spectra)
    mock_find_arc_lines = mocker.patch("specreduce.wavecal1d.find_arc_lines")
    mock_find_arc_lines.return_value = {"centroid": np.array([5.0]) * u.angstrom}
    ws.find_lines(fwhm=2.0, noise_factor=1.5)
    assert ws._obs_lines is not None
    assert len(ws._obs_lines) == len(arc_spectra)
    assert mock_find_arc_lines.called_once_with(arc_spectra[0], 2.0, noise_factor=1.5)


def test_find_lines_with_missing_arc_spectra():
    ws = WavelengthSolution1D(ref_pixel)
    with pytest.raises(ValueError, match="Must provide arc spectra to find lines."):
        ws.find_lines(fwhm=2.0, noise_factor=1.5)


def test_fit_lines_with_valid_input():
    pix_bounds = (0, 10)
    pixels = array([2, 4, 6, 8])
    wavelengths = array([500, 600, 700, 800])
    ws = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)
    ws.fit_lines(pixels=pixels, wavelengths=wavelengths)
    assert ws._p2w is not None
    assert ws._p2w[1].degree == ws.degree
    ws = WavelengthSolution1D(ref_pixel, obs_lines=pixels, line_lists=wavelengths, pix_bounds=pix_bounds)
    ws.fit_lines(pixels=pixels, wavelengths=wavelengths, match_cat=True, match_obs=True)
    assert ws._p2w is not None
    assert ws._p2w[1].degree == ws.degree
    ws = WavelengthSolution1D(ref_pixel, degree=5, pix_bounds=pix_bounds)
    ws.fit_lines(pixels=pixels[:3], wavelengths=wavelengths[:3])


def test_fit_lines_raises_error_for_missing_input():
    pix_bounds = (0, 10)
    pixels = array([2, 4, 6, 8])
    wavelengths = array([500, 600, 700, 800])
    ws = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)
    with pytest.raises(ValueError, match="Cannot fit without catalog"):
        ws.fit_lines(pixels=pixels, wavelengths=wavelengths, match_cat=True, match_obs=True)
    with pytest.raises(ValueError, match="Cannot fit without observed"):
        ws.fit_lines(pixels=pixels, wavelengths=wavelengths, match_cat=False, match_obs=True)

#def test_fit_lines_raises_error_for_nonexisting_lists():
#    pix_bounds = (0, 10)
#    pixels = array([2, 4, 6, 8])
#    wavelengths = array([500, 600, 700, 800])
#    ws = WavelengthSolution1D(ref_pixel, line_lists=wavelengths, pix_bounds=pix_bounds)
#    #with pytest.raises(ValueError, match="The sizes of pixel and wavelength arrays must match."):
#    ws.fit_lines(pixels=pixels, wavelengths=wavelengths)

def test_fit_lines_raises_error_for_mismatched_sizes():
    pix_bounds = (0, 10)
    pixels = array([2, 4, 6])
    wavelengths = array([500, 600, 700, 800])
    ws = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)
    with pytest.raises(ValueError, match="The sizes of pixel and wavelength arrays must match."):
        ws.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_insufficient_lines():
    pix_bounds = (0, 10)
    pixels = [5]
    wavelengths = [500]
    ws = WavelengthSolution1D(ref_pixel, pix_bounds=pix_bounds)
    with pytest.raises(ValueError, match="Need at least two lines for a fit"):
        ws.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_missing_pixel_bounds():
    pixels = [2, 4, 6, 8]
    wavelengths = [500, 600, 700, 800]
    ws = WavelengthSolution1D(ref_pixel)
    with pytest.raises(ValueError, match="Cannot fit without pixel bounds set."):
        ws.fit_lines(pixels=pixels, wavelengths=wavelengths)
