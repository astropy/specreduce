# specreduce/tests/test_wavecal1d.py
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import array

from specreduce.compat import Spectrum
from specreduce.wavecal1d import WavelengthCalibration1D

ref_pixel = 250
pix_bounds = (0, 500)
p2w = models.Shift(-ref_pixel) | models.Polynomial1D(degree=3, c0=1, c1=0.2, c2=0.001)


@pytest.fixture
def mk_lines():
    obs_lines = ref_pixel + array([1, 2, 5, 8, 10])
    cat_lines = p2w(ref_pixel + array([1, 3, 5, 7, 8, 10]))
    return obs_lines, cat_lines


@pytest.fixture
def mk_matched_lines():
    obs_lines = ref_pixel + array([1, 2, 5, 8, 10])
    cat_lines = p2w(obs_lines)
    return obs_lines, cat_lines


@pytest.fixture
def mk_wc(mk_lines):
    obs_lines, cat_lines = mk_lines
    return WavelengthCalibration1D(
        obs_lines=obs_lines,
        line_lists=cat_lines,
        pix_bounds=(0, 10),
        ref_pixel=ref_pixel,
    )


@pytest.fixture
def mk_good_wc_with_transform(mk_lines):
    obs_lines, cat_lines = mk_lines
    wc = WavelengthCalibration1D(
        obs_lines=obs_lines,
        line_lists=cat_lines,
        pix_bounds=pix_bounds,
        ref_pixel=ref_pixel,
    )
    wc.solution.p2w = p2w
    return wc


@pytest.fixture
def mk_arc():
    return Spectrum(
        flux=np.ones(pix_bounds[1]) * u.DN,
        spectral_axis=np.arange(pix_bounds[1]) * u.pix,
        uncertainty=StdDevUncertainty(np.ones(pix_bounds[1])),
    )


def test_init(mk_arc, mk_lines):
    arc = mk_arc
    obs_lines, cat_lines = mk_lines
    WavelengthCalibration1D(arc_spectra=arc, line_lists=cat_lines, ref_pixel=ref_pixel)
    WavelengthCalibration1D(
        obs_lines=obs_lines,
        line_lists=cat_lines,
        pix_bounds=(0, 10),
        ref_pixel=ref_pixel,
    )

    with pytest.raises(ValueError, match="Only one of arc_spectra or obs_lines can be provided."):
        WavelengthCalibration1D(arc_spectra=arc, obs_lines=obs_lines, ref_pixel=ref_pixel)

    arc = Spectrum(
        flux=np.array([[1, 2, 3, 4, 5]]) * u.DN,
        spectral_axis=np.array([1, 2, 3, 4, 5]) * u.angstrom,
    )
    with pytest.raises(ValueError, match="The arc spectrum must be one dimensional."):
        WavelengthCalibration1D(arc_spectra=arc, ref_pixel=ref_pixel)

    with pytest.raises(ValueError, match="Must give pixel bounds when"):
        WavelengthCalibration1D(obs_lines=obs_lines, ref_pixel=ref_pixel)


@pytest.mark.remote_data
def test_init_line_list(mk_arc):
    arc = mk_arc
    WavelengthCalibration1D(arc_spectra=arc, line_lists=["ArI"])
    WavelengthCalibration1D(arc_spectra=arc, line_lists="ArI")
    WavelengthCalibration1D(arc_spectra=arc, line_lists=[array([0.1])])
    WavelengthCalibration1D(arc_spectra=arc, line_lists=array([0.1]))
    WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=[["ArI"], ["ArI"]])
    WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=["ArI", ["ArI"]])
    WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=["ArI", "ArI"])
    WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=[array([0.1]), array([0.1])])
    WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=[array([0.1, 0.3]), ["ArI"]])
    with pytest.raises(ValueError, match="The number of line lists"):
        WavelengthCalibration1D(arc_spectra=[arc, arc], line_lists=[["ArI"]])


def test_find_lines(mocker, mk_arc):
    wc = WavelengthCalibration1D(arc_spectra=mk_arc)
    mock_find_arc_lines = mocker.patch("specreduce.wavecal1d.find_arc_lines")
    mock_find_arc_lines.return_value = {"centroid": np.array([5.0]) * u.angstrom}
    wc.find_lines(fwhm=2.0, noise_factor=1.5)
    assert wc._obs_lines is not None
    assert len(wc._obs_lines) == 1

    wc = WavelengthCalibration1D()
    with pytest.raises(ValueError, match="Must provide arc spectra to find lines."):
        wc.find_lines(fwhm=2.0, noise_factor=1.5)


def test_fit_lines(mk_matched_lines):
    lo, lc = mk_matched_lines
    wc = WavelengthCalibration1D(pix_bounds=pix_bounds, ref_pixel=ref_pixel)
    wc.fit_lines(pixels=lo, wavelengths=lc)
    assert wc.solution.p2w is not None
    assert wc.solution.p2w[1].degree == wc.degree

    wc = WavelengthCalibration1D(
        obs_lines=lo, line_lists=lc, pix_bounds=pix_bounds, ref_pixel=ref_pixel
    )
    wc.fit_lines(pixels=lo, wavelengths=lc, match_cat=True, match_obs=True)
    assert wc.solution.p2w is not None
    assert wc.solution.p2w[1].degree == wc.degree

    wc = WavelengthCalibration1D(pix_bounds=pix_bounds, ref_pixel=ref_pixel)
    with pytest.warns(UserWarning, match="The degree of the polynomial"):
        wc.fit_lines(degree=5, pixels=lo[:3], wavelengths=lc[:3])
    with pytest.raises(ValueError, match="Cannot fit without catalog"):
        wc.fit_lines(pixels=lo, wavelengths=lc, match_cat=True, match_obs=True)
    with pytest.raises(ValueError, match="Cannot fit without observed"):
        wc.fit_lines(pixels=lo, wavelengths=lc, match_cat=False, match_obs=True)


def test_observed_lines(mk_lines):
    wc = WavelengthCalibration1D()
    assert wc.observed_lines is None
    obs_lines, cat_lines = mk_lines
    wc = WavelengthCalibration1D(obs_lines=obs_lines, pix_bounds=pix_bounds, ref_pixel=ref_pixel)
    assert len(wc.observed_lines) == 1
    assert wc.observed_lines[0].shape == (len(obs_lines), 2)
    assert wc.observed_lines[0].mask.shape == (len(obs_lines), 2)
    np.testing.assert_allclose(wc.observed_lines[0].data[:, 0], obs_lines)
    assert np.all(wc.observed_lines[0].mask[:, 0] == 0)

    wc.observed_lines = obs_lines
    assert len(wc.observed_lines) == 1
    assert wc.observed_lines[0].mask.shape == (len(obs_lines), 2)
    np.testing.assert_allclose(wc.observed_lines[0].data[:, 0], obs_lines)
    assert np.all(wc.observed_lines[0].mask[:, 0] == 0)

    wc.observed_lines = wc.observed_lines
    assert len(wc.observed_lines) == 1
    assert wc.observed_lines[0].mask.shape == (len(obs_lines), 2)
    np.testing.assert_allclose(wc.observed_lines[0].data[:, 0], obs_lines)
    assert np.all(wc.observed_lines[0].mask[:, 0] == 0)

    # Line locations and amplitudes
    assert wc.observed_line_locations[0].shape == (len(obs_lines),)
    np.testing.assert_allclose(wc.observed_line_locations[0], obs_lines)
    assert wc.observed_line_amplitudes[0].shape == (len(obs_lines),)
    np.testing.assert_allclose(wc.observed_line_amplitudes[0], 0.0)


def test_catalog_lines(mk_lines):
    wc = WavelengthCalibration1D(ref_pixel=ref_pixel)
    assert wc.catalog_lines is None
    obs_lines, cat_lines = mk_lines
    wc = WavelengthCalibration1D(
        obs_lines=obs_lines, line_lists=cat_lines, pix_bounds=pix_bounds, ref_pixel=ref_pixel
    )
    assert len(wc.catalog_lines) == 1
    assert wc.catalog_lines[0].shape == (len(cat_lines), 2)
    np.testing.assert_allclose(wc.catalog_lines[0].data[:, 0], cat_lines)
    assert np.all(wc.catalog_lines[0].data[:, 1] == 0)

    wc.catalog_lines = cat_lines
    assert len(wc.catalog_lines) == 1
    np.testing.assert_allclose(wc.catalog_lines[0].data[:, 0], cat_lines)
    assert np.all(wc.catalog_lines[0].data[:, 1] == 0)

    wc.catalog_lines = wc.catalog_lines
    assert len(wc.catalog_lines) == 1
    np.testing.assert_allclose(wc.catalog_lines[0].data[:, 0], cat_lines)
    assert np.all(wc.catalog_lines[0].data[:, 1] == 0)


def test_fit_lines_raises_error_for_mismatched_sizes():
    pixels = array([2, 4, 6])
    wavelengths = p2w(array([2, 4, 5, 6]))
    wc = WavelengthCalibration1D(pix_bounds=pix_bounds, ref_pixel=ref_pixel)
    with pytest.raises(ValueError, match="The sizes of pixel and wavelength arrays must match."):
        wc.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_insufficient_lines():
    pixels = [5]
    wavelengths = p2w(pixels)
    wc = WavelengthCalibration1D(pix_bounds=pix_bounds, ref_pixel=ref_pixel)
    with pytest.raises(ValueError, match="Need at least two lines for a fit"):
        wc.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_lines_raises_error_for_missing_pixel_bounds():
    pixels = [2, 4, 6, 8]
    wavelengths = p2w(pixels)
    wc = WavelengthCalibration1D(ref_pixel=ref_pixel)
    with pytest.raises(ValueError, match="Cannot fit without pixel bounds set."):
        wc.fit_lines(pixels=pixels, wavelengths=wavelengths)


def test_fit_global():
    p2w = models.Shift(-ref_pixel) | models.Polynomial1D(degree=3, c0=650, c1=50.0, c2=-0.001)

    lines_obs = ref_pixel + array([2, 4, 4.5, 5, 6, 6.3, 8])
    lines_cat = p2w(ref_pixel + array([1, 2, 2.3, 4, 5, 6, 6.3, 7, 8, 9, 10]))
    wavelength_bounds = (649, 651)
    dispersion_bounds = (49, 51)
    wc = WavelengthCalibration1D(
        obs_lines=lines_obs, line_lists=lines_cat, pix_bounds=pix_bounds, ref_pixel=ref_pixel
    )
    ws = wc.fit_dispersion(wavelength_bounds, dispersion_bounds, popsize=10, refine_fit=True)
    np.testing.assert_allclose(ws.p2w[1].parameters, [650.0, 50.0, -0.001, 0.0], atol=1e-4)
    assert wc._fit is not None
    assert wc._fit.success
    assert wc.solution.p2w is not None

    wc = WavelengthCalibration1D(
        obs_lines=lines_obs, line_lists=lines_cat, pix_bounds=pix_bounds, ref_pixel=ref_pixel
    )
    wc.fit_dispersion(wavelength_bounds, dispersion_bounds, popsize=10, refine_fit=False)


def test_rms(mk_good_wc_with_transform):
    wc = mk_good_wc_with_transform
    assert np.isclose(wc.rms(space="wavelength"), 0)  # Perfect match, so RMS should be zero
    assert np.isclose(wc.rms(space="pixel"), 0)  # Perfect match, so RMS should be zero
    with pytest.raises(ValueError, match="Space must be either 'pixel' or 'wavelength'"):
        wc.rms(space="wavelenght")


def test_remove_unmatched_lines(mk_good_wc_with_transform):
    wc = mk_good_wc_with_transform
    wc.match_lines()
    wc.remove_unmatched_lines()
    assert wc.catalog_lines[0].size == wc.observed_lines[0].size


def test_plot_lines_with_valid_input():
    wc = WavelengthCalibration1D(ref_pixel=ref_pixel)
    wc.observed_lines = [np.ma.masked_array([100, 200, 300], mask=[False, True, False])]
    wc._cat_lines = wc.observed_lines
    fig = wc._plot_lines(kind="observed", frames=0, figsize=(8, 4), plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = wc._plot_lines(kind="catalog", frames=0, figsize=(8, 4), plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    fig = wc._plot_lines(kind="catalog", frames=0, axes=ax, plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, axs = plt.subplots(1, 2)
    fig = wc._plot_lines(kind="catalog", frames=0, axes=axs, plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = wc._plot_lines(kind="observed", frames=0, axes=axs, plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()


def test_plot_lines_raises_for_missing_transform(mk_wc):
    wc = mk_wc
    with pytest.raises(ValueError, match="Cannot map between pixels and"):
        wc._plot_lines(kind="observed", map_x=True)


def test_plot_lines_calls_transform_correctly(mk_good_wc_with_transform):
    wc = mk_good_wc_with_transform
    wc._plot_lines(kind="observed", map_x=True)
    wc._plot_lines(kind="catalog", map_x=True)


def test_plot_catalog_lines(mk_wc):
    wc = mk_wc
    wc.catalog_lines = [np.ma.masked_array([400, 500, 600], mask=[False, True, False])]
    fig = wc.plot_catalog_lines(frames=0, figsize=(10, 6), plot_labels=True, map_to_pix=False)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    fig = wc.plot_catalog_lines(frames=0, axes=ax, plot_labels=True)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, axs = plt.subplots(1, 2)
    fig = wc.plot_catalog_lines(frames=[0], axes=axs, plot_labels=False)
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()


def test_plot_observed_lines(mk_good_wc_with_transform, mk_arc):
    wc = mk_good_wc_with_transform
    wc.observed_lines = [np.ma.masked_array([100, 200, 300], mask=[False, True, False])]
    wc.arc_spectra = [mk_arc]
    for frames in [None, 0]:
        fig = wc.plot_observed_lines(frames=frames, figsize=(10, 5), plot_labels=True)
        assert isinstance(fig, Figure)
        assert fig.axes[0].has_data()
        assert len(fig.axes) == 1


def test_plot_fit(mk_arc, mk_good_wc_with_transform):
    wc = mk_good_wc_with_transform
    wc.arc_spectra = [mk_arc]
    for frames in [None, 0]:
        fig = wc.plot_fit(frames=frames, figsize=(12, 6), plot_labels=True)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig.axes[0].has_data()
        assert fig.axes[1].has_data()
    wc.plot_fit(frames=frames, figsize=(12, 6), plot_labels=True, obs_to_wav=True)


def test_plot_residuals(mk_good_wc_with_transform):
    wc = mk_good_wc_with_transform

    fig = wc.plot_residuals(space="pixel", figsize=(8, 4))
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig = wc.plot_residuals(space="wavelength", figsize=(8, 4))
    assert isinstance(fig, Figure)
    assert fig.axes[0].has_data()

    fig, ax = plt.subplots(1, 1)
    wc.plot_residuals(ax=ax, space="wavelength", figsize=(8, 4))

    with pytest.raises(ValueError, match="Invalid space specified"):
        wc.plot_residuals(space="wavelenght", figsize=(8, 4))
