# specreduce/tests/test_wavecal1d.py
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from gwcs import wcs
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import array

from specreduce.wavecal1d import WavelengthSolution1D, _diff_poly1d
from specutils import Spectrum1D

ref_pixel = 5
p2w =  models.Shift(ref_pixel) | models.Polynomial1D(degree=3, c0=1, c1=2, c2=3)
obs_lines = array([1, 2, 5, 8, 10])
cat_lines = p2w(array([1, 3, 5, 7, 8, 10]))

def test_diff_poly1d():
    p = _diff_poly1d(Polynomial1D(3, c0=1.0, c1=2.0, c2=3.0, c3=4.0))
    np.testing.assert_array_equal(p.parameters, [2.0, 6.0, 12.0])


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
    arc_spectra = Spectrum1D(
        flux=np.array([[1, 2]]) * u.DN, spectral_axis=np.array([1, 2]) * u.angstrom
    )
    with pytest.raises(ValueError, match="The arc spectra must be 1 dimensional."):
        WavelengthSolution1D(ref_pixel, arc_spectra=arc_spectra)


def test_init_obs_lines_requires_pixel_bounds():
    obs_lines = [np.array([500.0])]
    with pytest.raises(ValueError, match="Must give pixel bounds when"):
        WavelengthSolution1D(ref_pixel, obs_lines=obs_lines)


def test_init_line_list():
    """Test the catalog line list initialization with various configurations of `arc_spectra` and `line_lists`."""
    arc = Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom)
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=["ArI"])
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists="ArI")
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=[array([0.1])])
    WavelengthSolution1D(ref_pixel, arc_spectra=arc, line_lists=array([0.1]))
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[["ArI"], ["ArI"]])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=["ArI", ["ArI"]])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=["ArI", "ArI"])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[array([0.1]), array([0.1])])
    WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[array([0.1, 0.3]), ["ArI"]])
    with pytest.raises(ValueError, match="The number of line lists"):
        WavelengthSolution1D(ref_pixel, arc_spectra=[arc, arc], line_lists=[["ArI"]])


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
    ws = WavelengthSolution1D(
        ref_pixel, obs_lines=pixels, line_lists=wavelengths, pix_bounds=pix_bounds
    )
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


def test_observed_lines_returns_none_when_not_set():
    ws = WavelengthSolution1D(ref_pixel)
    assert ws.observed_lines is None


def test_observed_lines_returns_expected_lines():
    mock_obs_lines = [np.ma.masked_array([100, 200], mask=[False, True])]
    ws = WavelengthSolution1D(ref_pixel)
    ws._obs_lines = mock_obs_lines
    assert ws.observed_lines == mock_obs_lines


# def test_fit_lines_raises_error_for_nonexisting_lists():
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


def test_fit_global_runs_successfully_with_valid_input():
    pix_bounds = (0, 10)
    pixels = array([2, 4, 5, 6, 8])
    wavelengths = array([500, 550, 600, 650, 700, 750, 800])
    wavelength_bounds = (649, 651)
    dispersion_bounds = (49, 51)
    ws = WavelengthSolution1D(5, pix_bounds=pix_bounds, obs_lines=pixels, line_lists=wavelengths)
    ws.fit_global(wavelength_bounds, dispersion_bounds, popsize=10)
    np.testing.assert_allclose(ws._fit.x, [650.0, 50.0, 0.0, 0.0], atol=1e-4)
    assert ws._fit is not None
    assert ws._fit.success
    assert ws._p2w is not None

    ws = WavelengthSolution1D(5, pix_bounds=pix_bounds, obs_lines=pixels, line_lists=wavelengths)
    ws.fit_global(wavelength_bounds, dispersion_bounds, popsize=10, refine_fit=False)


def test_resample_with_valid_input():
    arc_spectrum = Spectrum1D(flux=np.ones(20) * u.DN, spectral_axis=np.arange(20) * u.angstrom, uncertainty=StdDevUncertainty(np.ones(20)))
    ws = WavelengthSolution1D(ref_pixel)
    ws._p2w = lambda x: x * 2  # Mock pixel-to-wavelength conversion
    ws._p2w_dldx = lambda x: np.ones_like(x) * 2  # Mock derivative
    ws._w2p = lambda x: x / 2  # Mock wavelength-to-pixel conversion
    resampled = ws.resample(arc_spectrum, nbins=10)
    assert resampled is not None
    assert len(resampled.flux) == 10
    assert resampled.flux.unit == u.DN
    np.testing.assert_almost_equal(arc_spectrum.flux.value.sum(), resampled.flux.value.sum(), decimal=10)


def test_resample_raises_error_for_missing_transforms():
    arc_spectrum = Spectrum1D(flux=np.ones(10) * u.DN, spectral_axis=np.arange(10) * u.angstrom, uncertainty=StdDevUncertainty(np.ones(10)))
    ws = WavelengthSolution1D(ref_pixel)
    with pytest.raises(ValueError, match="Wavelength solution not yet"):
        ws.resample(arc_spectrum)


def test_resample_raises_error_for_invalid_bins():
    arc_spectrum = Spectrum1D(flux=np.ones(20) * u.DN, spectral_axis=np.arange(20) * u.angstrom, uncertainty=StdDevUncertainty(np.ones(20)))
    ws = WavelengthSolution1D(ref_pixel)
    ws._p2w = lambda x: x * 2  # Mock pixel-to-wavelength conversion
    ws._p2w_dldx = lambda x: np.ones_like(x) * 2  # Mock derivative
    ws._w2p = lambda x: x / 2  # Mock wavelength-to-pixel conversion
    with pytest.raises(ValueError, match="Number of bins must be positive"):
        ws.resample(arc_spectrum, nbins=-5)


def test_pix_to_wav_with_array():
    pix_values = np.array([1, 2, 3, 4, 5])
    ws = WavelengthSolution1D(ref_pixel)
    ws._p2w = lambda x: x * 10  # Mock pixel-to-wavelength conversion
    wavelengths = ws.pix_to_wav(pix_values)
    np.testing.assert_array_equal(wavelengths, np.array([10, 20, 30, 40, 50]))


def test_pix_to_wav_with_masked_array():
    pix_values = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
    ws = WavelengthSolution1D(ref_pixel)
    ws._p2w = lambda x: x * 10  # Mock pixel-to-wavelength conversion
    wavelengths = ws.pix_to_wav(pix_values)
    np.testing.assert_array_equal(wavelengths.data, np.array([10, 20, 30]))
    np.testing.assert_array_equal(wavelengths.mask, np.array([0, 1, 0]))


def test_wav_to_pix_with_array():
    wav_values = np.array([500, 1000, 1500])
    ws = WavelengthSolution1D(ref_pixel)
    ws._w2p = lambda x: x / 10  # Mock wavelength-to-pixel conversion
    pixel_values = ws.wav_to_pix(wav_values)
    np.testing.assert_array_equal(pixel_values, np.array([50, 100, 150]))


def test_wav_to_pix_with_masked_array():
    wav_values = np.ma.masked_array([500, 1000, 1500], mask=[0, 1, 0])
    ws = WavelengthSolution1D(ref_pixel)
    ws._w2p = lambda x: x / 10  # Mock wavelength-to-pixel conversion
    pixel_values = ws.wav_to_pix(wav_values)
    np.testing.assert_array_equal(pixel_values.data, np.array([50, 100, 150]))
    np.testing.assert_array_equal(pixel_values.mask, np.array([0, 1, 0]))


def test_wcs_creates_valid_gwcs_object():
    ws = WavelengthSolution1D(ref_pixel, unit=u.angstrom, pix_bounds=(0, 100))
    ws._p2w = models.Shift(5) | models.Polynomial1D(degree=3, c0=1, c1=2, c2=3)
    ws._calculate_p2w_inverse()
    ws._calculate_p2w_derivative()
    wcs_obj = ws.wcs
    assert wcs_obj is not None
    assert isinstance(wcs_obj, wcs.WCS)
    assert wcs_obj.output_frame.unit[0] == u.angstrom


def test_rms():
    ws = WavelengthSolution1D(ref_pixel, obs_lines=obs_lines, line_lists=cat_lines, pix_bounds=(0, 10))
    ws._p2w = p2w
    ws._calculate_p2w_inverse()
    assert np.isclose(ws.rms(space="wavelength"), 0)  # Perfect match, so RMS should be zero
    assert np.isclose(ws.rms(space="pixel"), 0)  # Perfect match, so RMS should be zero
    with pytest.raises(ValueError, match="Space must be either 'pixel' or 'wavelength'"):
        ws.rms(space="wavelenght")


def test_remove_unmatched_lines():
    ws = WavelengthSolution1D(ref_pixel, obs_lines=obs_lines, line_lists=cat_lines, pix_bounds=(0, 10))
    ws._p2w = p2w
    ws._calculate_p2w_inverse()
    ws.match_lines()
    ws.remove_ummatched_lines()


def test_plot_lines_with_valid_input():
    ws = WavelengthSolution1D(ref_pixel)
    ws._obs_lines = [np.ma.masked_array([100, 200, 300], mask=[False, True, False])]
    ws._cat_lines = ws._obs_lines
    fig = ws._plot_lines(kind="observed", frames=0, figsize=(8, 4), plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = ws._plot_lines(kind="catalog", frames=0, figsize=(8, 4), plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    fig = ws._plot_lines(kind="catalog", frames=0, axs=ax, plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, axs = plt.subplots(1, 2)
    fig = ws._plot_lines(kind="catalog", frames=0, axs=axs, plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = ws._plot_lines(kind="observed", frames=0, axs=axs, plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()


def test_plot_lines_raises_for_missing_transform():
    ws = WavelengthSolution1D(
        ref_pixel, obs_lines=array([2, 5, 8]), line_lists=array([500, 600, 700]), pix_bounds=(0, 10)
    )
    with pytest.raises(ValueError, match="Cannot map between pixels and"):
        ws._plot_lines(kind="observed", map_x=True)


def test_plot_lines_calls_transform_correctly(mocker):
    ws = WavelengthSolution1D(
        ref_pixel, obs_lines=array([2, 5, 8]), line_lists=array([500, 600, 700]), pix_bounds=(0, 10)
    )
    ws._p2w = p2w
    ws._calculate_p2w_inverse()
    ws._plot_lines(kind="observed", map_x=True)
    ws._plot_lines(kind="catalog", map_x=True)


def test_plot_catalog_lines_with_valid_input():
    ws = WavelengthSolution1D(ref_pixel)
    ws._cat_lines = [np.ma.masked_array([400, 500, 600], mask=[False, True, False])]
    fig = ws.plot_catalog_lines(frames=0, figsize=(10, 6), plot_values=True, map_to_pix=False)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    fig = ws.plot_catalog_lines(frames=0, axes=ax, plot_values=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, axs = plt.subplots(1, 2)
    fig = ws.plot_catalog_lines(frames=[0], axes=axs, plot_values=False)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()


def test_plot_observed_lines_with_valid_input():
    ws = WavelengthSolution1D(ref_pixel)
    ws._obs_lines = [np.ma.masked_array([100, 200, 300], mask=[False, True, False])]
    arc_spectrum = Spectrum1D(flux=np.ones(20) * u.DN, spectral_axis=np.arange(20) * u.angstrom)
    ws.arc_spectra = [arc_spectrum]
    ws._p2w = p2w
    fig = ws.plot_observed_lines(frames=0, figsize=(10, 5), plot_values=True, plot_spectra=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()
    assert len(fig.axes) == 1


def test_plot_fit_with_valid_input():
    arc_spectrum = Spectrum1D(flux=np.ones(20) * u.DN, spectral_axis=np.arange(20) * u.angstrom)
    ws = WavelengthSolution1D(
        ref_pixel, obs_lines=array([2, 5, 8]), line_lists=array([500, 600, 700]), pix_bounds=(0, 10)
    )
    ws.arc_spectra = [arc_spectrum]
    ws._p2w = p2w
    fig = ws.plot_fit(frames=0, figsize=(12, 6), plot_values=True)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert fig.axes[0].has_data()
    assert fig.axes[1].has_data()


def test_plot_residuals():
    obs_lines = array([2, 5, 8])
    cat_lines = p2w(obs_lines)
    ws = WavelengthSolution1D(ref_pixel, pix_bounds=(0, 10), obs_lines=obs_lines, line_lists=cat_lines)
    ws._p2w = p2w
    ws._calculate_p2w_inverse()

    fig = ws.plot_residuals(space="pixel", figsize=(8, 4))
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = ws.plot_residuals(space="wavelength", figsize=(8, 4))
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    ws.plot_residuals(ax=ax, space="wavelength", figsize=(8, 4))

    with pytest.raises(ValueError, match="Invalid space specified"):
        fig = ws.plot_residuals(space="wavelenght", figsize=(8, 4))
