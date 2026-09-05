import asdf
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models, CompoundModel
from astropy.nddata import NDData, StdDevUncertainty, VarianceUncertainty
from gwcs import coordinate_frames
from specutils import Spectrum

from specreduce.tilt_solution import TiltSolution
from specreduce.wavesol1d import WavelengthSolution1D
from specreduce.wavesol2d import WavelengthSolution2D

NY, NX = 40, 200
REF_X, REF_Y = 100.0, 20.0
BOUNDS_PIX = (0, NX)

# Detector x = rectified x at the reference row, with a linear tilt and a mild
# shear that makes the tilt grow along the dispersion axis.
TILT_COEFFS = dict(c0_0=REF_X, c1_0=1.0, c0_1=0.05, c1_1=0.0005)
WAVE_COEFFS = dict(c0=5000.0, c1=2.0, c2=1e-4)


def make_tilt(image_shape=(NY, NX), disp_axis=1, **coeffs):
    coeffs = coeffs or TILT_COEFFS
    c2d = (models.Shift(-REF_X) & models.Shift(-REF_Y)) | models.Polynomial2D(2, **coeffs)
    return TiltSolution(c2d, disp_axis=disp_axis, image_shape=image_shape)


def make_wave(wave_air=False):
    p2w = models.Shift(-REF_X) | models.Polynomial1D(2, **WAVE_COEFFS)
    return WavelengthSolution1D(p2w, BOUNDS_PIX, u.angstrom, wave_air=wave_air)


@pytest.fixture
def tilt():
    return make_tilt()


@pytest.fixture
def wave():
    return make_wave()


@pytest.fixture
def ws2d(tilt, wave):
    return WavelengthSolution2D(tilt, wave)


@pytest.fixture
def grid():
    return np.meshgrid(np.linspace(5, NX - 5, 7), np.linspace(2, NY - 2, 5))


# --- Construction ---------------------------------------------------------------------------


def test_init_from_solutions(tilt, wave):
    ws = WavelengthSolution2D(tilt, wave)
    assert ws.tilt is tilt
    assert ws.wavelength is wave
    assert ws.unit == u.angstrom
    assert ws.wave_air is False
    assert ws.disp_axis == 1
    assert ws.image_shape == (NY, NX)
    assert ws.bounds_pix == BOUNDS_PIX


def test_init_wave_air_follows_wavelength_solution(tilt):
    ws = WavelengthSolution2D(tilt, make_wave(wave_air=True))
    assert ws.wave_air is True


def test_p2w_is_a_2d_to_1d_model(ws2d):
    assert isinstance(ws2d.p2w, CompoundModel)
    assert ws2d.p2w.n_inputs == 2
    assert ws2d.p2w.n_outputs == 1


def test_p2w_chains_tilt_inverse_and_wavelength(ws2d, tilt, wave, grid):
    x, y = grid
    expected = wave.p2w(tilt.d2c(x, y))
    np.testing.assert_allclose(ws2d.p2w(x, y), expected)


def test_pix_to_wav_matches_1d_solution_at_reference_row(ws2d, wave):
    x = np.linspace(5, NX - 5, 20)
    np.testing.assert_allclose(ws2d.pix_to_wav(x, np.full_like(x, REF_Y)), wave.p2w(x), rtol=1e-6)


def test_pix_to_wav_off_reference_row_is_tilted(ws2d, wave):
    # At y = REF_Y + 10 the tilt model gives x_det = x_rec + 0.5 + 0.005 (x_rec - REF_X),
    # whose exact inverse is x_rec = REF_X + (x_det - REF_X - 0.5) / 1.005.
    x = np.array([50.0, 150.0])
    y = np.full_like(x, REF_Y + 10)
    x_rec = REF_X + (x - REF_X - 0.5) / 1.005
    np.testing.assert_allclose(ws2d.pix_to_wav(x, y), wave.p2w(x_rec), rtol=1e-6)


def test_pix_to_wav_masked_array(ws2d):
    x = np.ma.masked_array([10.0, 20.0, 30.0], mask=[False, True, False])
    y = np.ma.masked_array([5.0, 5.0, 5.0], mask=[False, False, False])
    wav = ws2d.pix_to_wav(x, y)
    assert isinstance(wav, np.ma.MaskedArray)
    np.testing.assert_array_equal(wav.mask, [False, True, False])
    np.testing.assert_allclose(wav.data, ws2d.p2w(x.data, y.data))


def test_wav_to_pix_round_trip(ws2d, grid):
    x, y = grid
    wav = ws2d.pix_to_wav(x, y)
    np.testing.assert_allclose(ws2d.wav_to_pix(wav, y), x, atol=1e-3)


def test_wav_to_pix_masked_array(ws2d):
    wav = np.ma.masked_array([5100.0, 5200.0], mask=[True, False])
    y = np.array([10.0, 10.0])
    pix = ws2d.wav_to_pix(wav, y)
    assert isinstance(pix, np.ma.MaskedArray)
    np.testing.assert_array_equal(pix.mask, [True, False])


def test_wav_to_pix_outside_solution_is_nan(ws2d):
    assert np.isnan(ws2d.wav_to_pix(1000.0, 10.0))


def test_init_requires_image_shape(wave):
    tilt = make_tilt(image_shape=None)
    with pytest.raises(ValueError, match="image shape"):
        WavelengthSolution2D(tilt, wave)


def test_init_image_shape_argument_completes_tilt_solution(wave):
    tilt = make_tilt(image_shape=None)
    ws = WavelengthSolution2D(tilt, wave, image_shape=(NY, NX))
    assert ws.image_shape == (NY, NX)
    assert np.isfinite(ws.p2w(50.0, 10.0))


def test_init_from_gwcs_objects(tilt, wave, grid):
    ws_ref = WavelengthSolution2D(tilt, wave)
    ws = WavelengthSolution2D(tilt.gwcs, wave.gwcs, image_shape=(NY, NX))
    assert ws.unit == u.angstrom
    assert ws.image_shape == (NY, NX)
    assert ws.bounds_pix == BOUNDS_PIX
    x, y = grid
    np.testing.assert_allclose(ws.p2w(x, y), ws_ref.p2w(x, y))


def test_init_from_gwcs_requires_image_shape(tilt, wave):
    with pytest.raises(ValueError, match="image shape"):
        WavelengthSolution2D(tilt.gwcs, wave.gwcs)


def test_setting_components_invalidates_caches(ws2d, tilt, wave):
    cached = {"p2w", "_w2p_table", "w2p", "gwcs"}
    _ = ws2d.p2w, ws2d.w2p, ws2d.gwcs
    assert cached <= set(ws2d.__dict__)
    ws2d.tilt = tilt
    assert not cached & set(ws2d.__dict__)

    _ = ws2d.p2w, ws2d.w2p, ws2d.gwcs
    ws2d.wavelength = wave
    assert not cached & set(ws2d.__dict__)


# --- GWCS -----------------------------------------------------------------------------------


def test_gwcs_maps_detector_pixels_to_wavelength(ws2d, grid):
    w = ws2d.gwcs
    assert w.input_frame.naxes == 2
    assert w.input_frame.name == "detector"
    assert isinstance(w.output_frame, coordinate_frames.SpectralFrame)
    assert w.output_frame.unit == (u.angstrom,)
    x, y = grid
    np.testing.assert_allclose(w(x, y), ws2d.p2w(x, y))


def test_gwcs_is_unbounded(ws2d):
    assert np.isfinite(ws2d.gwcs(-50.0, -10.0))


# --- ASDF -----------------------------------------------------------------------------------


def test_asdf_round_trip_is_lossless(tmp_path, tilt, grid):
    ws = WavelengthSolution2D(tilt, make_wave(wave_air=True))
    path = tmp_path / "ws2d.asdf"
    ws.to_asdf(path)
    ws2 = WavelengthSolution2D.from_asdf(path)

    assert ws2.unit == ws.unit
    assert ws2.wave_air is True
    assert ws2.disp_axis == ws.disp_axis
    assert ws2.image_shape == ws.image_shape
    assert ws2.bounds_pix == ws.bounds_pix
    x, y = grid
    np.testing.assert_allclose(ws2.p2w(x, y), ws.p2w(x, y), rtol=1e-12)
    np.testing.assert_allclose(
        ws2.tilt.corr_to_det(x, y)[0], ws.tilt.corr_to_det(x, y)[0], rtol=1e-12
    )
    np.testing.assert_allclose(ws2.wavelength.p2w(x[0]), ws.wavelength.p2w(x[0]), rtol=1e-12)


def test_asdf_file_carries_chained_gwcs(tmp_path, ws2d, grid):
    path = tmp_path / "ws2d.asdf"
    ws2d.to_asdf(path)
    x, y = grid
    with asdf.open(path) as af:
        w = af["wavelength_solution_2d"]["gwcs"]
        np.testing.assert_allclose(w(x, y), ws2d.p2w(x, y))
        # Outside the image the exported GWCS is bounded ...
        assert np.isnan(w(-50.0, 10.0))
    # ... but neither the live solution nor the restored one is.
    assert np.isfinite(ws2d.p2w(-50.0, 10.0))
    ws2 = WavelengthSolution2D.from_asdf(path)
    assert np.isfinite(ws2.p2w(-50.0, 10.0))
    assert np.isfinite(ws2.tilt.c2d(-50.0, 10.0))


def test_from_asdf_raises_on_foreign_file(tmp_path):
    path = tmp_path / "foreign.asdf"
    asdf.AsdfFile({"something": 1}).write_to(path)
    with pytest.raises(ValueError, match="WavelengthSolution2D.to_asdf"):
        WavelengthSolution2D.from_asdf(path)


# --- Resampling -----------------------------------------------------------------------------


def make_image(tilt=None, line_x=None, uncertainty=True):
    """A smooth 2D image, optionally with a tilted emission line at rectified column line_x."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:NY, 0:NX]
    flux = 100.0 + 50.0 * np.sin(x / 30.0) + rng.normal(0, 1, (NY, NX))
    if line_x is not None:
        xc = tilt.c2d(np.full(NY, float(line_x)), np.arange(NY, dtype=float))
        flux += 1000.0 * np.exp(-0.5 * ((x - xc[:, None]) / 1.5) ** 2)
    ucty = StdDevUncertainty(np.full((NY, NX), 3.0)) if uncertainty else None
    return NDData(flux * u.ct, uncertainty=ucty)


def pixel_wavelength_widths(ws):
    ny, nx = ws.image_shape
    xe, ye = np.meshgrid(np.arange(nx + 1) - 0.5, np.arange(ny))
    return np.diff(ws.p2w(xe, ye), axis=1)


def test_resample_output_structure(ws2d):
    image = make_image()
    result = ws2d.resample(image, nbins=150)
    assert isinstance(result, Spectrum)
    assert result.flux.shape == (NY, 150)
    assert result.flux.unit == u.ct / u.angstrom
    assert result.spectral_axis.unit == u.angstrom
    assert result.spectral_axis.shape == (150,)
    assert result.spectral_axis_index == 1
    assert isinstance(result.uncertainty, StdDevUncertainty)
    assert result.uncertainty.array.shape == (NY, 150)
    assert result.mask.shape == (NY, 150)
    assert not result.mask.any()
    assert np.all(np.isfinite(result.flux.value))


def test_resample_default_bounds_are_reference_row_coverage(ws2d, wave):
    result = ws2d.resample(make_image(), nbins=NX)
    l1, l2 = wave.p2w(np.array([-0.5, NX - 0.5]))
    edges = np.linspace(l1, l2, NX + 1)
    np.testing.assert_allclose(result.spectral_axis.value, 0.5 * (edges[:-1] + edges[1:]))


def test_resample_conserves_flux(ws2d):
    image = make_image()
    ye = np.arange(NY)
    # A grid that covers every row with a margin, so no flux falls off its ends.
    lo = ws2d.p2w(np.full(NY, -0.5), ye).min() - 1.0
    hi = ws2d.p2w(np.full(NY, NX - 0.5), ye).max() + 1.0
    result = ws2d.resample(image, nbins=120, wlbounds=(lo, hi))
    dl = np.diff(result.spectral_axis.value)[0]
    f_in = image.data.sum(1)
    f_out = (result.flux.value * dl).sum(1)
    np.testing.assert_allclose(f_out, f_in, rtol=1e-8)


def test_resample_straightens_tilted_line(ws2d, tilt, wave):
    line_x = 130.0
    image = make_image(tilt, line_x=line_x)
    # Sanity check: on the detector the line peak drifts across rows.
    assert np.ptp(np.argmax(image.data, axis=1)) > 1
    result = ws2d.resample(image, nbins=NX)
    peaks = np.argmax(result.flux.value, axis=1)
    assert np.ptp(peaks) == 0
    expected = np.argmin(np.abs(result.spectral_axis.value - wave.p2w(line_x)))
    assert peaks[0] == expected


def rebin_row(flux, x_edges):
    """Reference histogram rebinning of one row onto bins with pixel-space edges x_edges."""
    out = np.zeros(x_edges.size - 1)
    for i in range(out.size):
        lo, hi = sorted(x_edges[i : i + 2])
        lo, hi = np.clip([lo, hi], 0, flux.size)
        px = np.arange(int(np.floor(lo)), int(np.ceil(hi)))
        overlap = np.minimum(hi, px + 1) - np.maximum(lo, px)
        out[i] = (overlap * flux[px]).sum()
    return out


def test_resample_matches_direct_rebinning_without_tilt(wave):
    tilt = make_tilt(c0_0=REF_X, c1_0=1.0)
    ws = WavelengthSolution2D(tilt, wave)
    image = make_image()
    result = ws.resample(image, nbins=90)
    edges = result.spectral_axis.bin_edges.value
    x_edges = wave.wav_to_pix(edges) + 0.5  # pixel j spans [j, j + 1) in this coordinate
    for row in (0, NY // 2, NY - 1):
        expected = rebin_row(image.data[row], x_edges) / np.diff(edges)
        np.testing.assert_allclose(result.flux.value[row], expected, rtol=1e-6)


def test_resample_propagates_variance(ws2d):
    image = make_image()
    # Bins wide enough to contain many pixels: the variance of a weighted sum of
    # independent pixels is the weighted sum of squares of sigma / dlambda_bin^2.
    result = ws2d.resample(image, nbins=10)
    var = result.uncertainty.represent_as(VarianceUncertainty).array
    dl = np.diff(result.spectral_axis.value)[0]
    # Interior bin, reference row: gather the contributing detector pixels exactly.
    row, ibin = int(REF_Y), 5
    edges = result.spectral_axis.value[ibin] + np.array([-0.5, 0.5]) * dl
    x_edges = ws2d.wav_to_pix(edges, np.full(2, float(row))) + 0.5
    i1, i2 = np.floor(x_edges).astype(int)
    w = np.zeros(NX)
    w[i1 + 1 : i2] = 1.0
    w[i1] = 1.0 - (x_edges[0] - i1)
    w[i2] = x_edges[1] - i2
    expected = (w**2 * 9.0).sum() / dl**2
    np.testing.assert_allclose(var[row, ibin], expected, rtol=1e-6)


def test_resample_without_uncertainty(ws2d):
    result = ws2d.resample(make_image(uncertainty=False), nbins=50)
    assert result.uncertainty is None
    assert result.flux.shape == (NY, 50)
    result = ws2d.resample(np.ones((NY, NX)), nbins=50)
    assert result.uncertainty is None


def test_resample_propagates_mask(ws2d):
    image = make_image()
    mask = np.zeros((NY, NX), dtype=bool)
    mask[7, 60] = True
    image = NDData(image.data * u.ct, mask=mask)
    result = ws2d.resample(image, nbins=50)
    assert result.mask[7].sum() in (1, 2)
    assert not result.mask[[6, 8]].any()
    wav = ws2d.pix_to_wav(60.0, 7.0)
    ibin = np.argmin(np.abs(result.spectral_axis.value - wav))
    assert result.mask[7, ibin]


def test_resample_nan_pixels_are_masked_not_spread(ws2d):
    image = make_image()
    data = image.data.copy()
    data[3, 100] = np.nan
    result = ws2d.resample(NDData(data * u.ct), nbins=50, mask_treatment="zero_fill")
    assert np.all(np.isfinite(result.flux.value))
    assert not result.mask.any()


def test_resample_explicit_bin_edges(ws2d):
    edges = 5050.0 + 100.0 * np.linspace(0, 1, 21) ** 2  # non-uniform widths
    result = ws2d.resample(make_image(), bin_edges=edges)
    assert result.flux.shape == (NY, 20)
    np.testing.assert_allclose(result.spectral_axis.value, 0.5 * (edges[:-1] + edges[1:]))
    np.testing.assert_allclose(result.spectral_axis.bin_edges.value, edges, rtol=1e-12)


def test_resample_disp_axis_0(wave):
    n = 64
    tilt = TiltSolution(
        (models.Shift(-32) & models.Shift(-32)) | models.Polynomial2D(1, c0_0=32, c1_0=1, c0_1=0.1),
        disp_axis=0,
        image_shape=(n, n),
    )
    w1d = WavelengthSolution1D(
        models.Shift(-32) | models.Polynomial1D(1, c0=5000.0, c1=2.0), (0, n), u.angstrom
    )
    ws = WavelengthSolution2D(tilt, w1d)
    image = NDData(np.ones((n, n)) * u.ct, uncertainty=StdDevUncertainty(np.ones((n, n))))
    result = ws.resample(image, nbins=30)
    assert result.flux.shape == (30, n)
    assert result.spectral_axis_index == 0
    assert result.uncertainty.array.shape == (30, n)
    assert result.mask.shape == (30, n)


def test_resample_rejects_negative_nbins(ws2d):
    with pytest.raises(ValueError, match="Number of bins"):
        ws2d.resample(make_image(), nbins=-3)
    with pytest.raises(ValueError, match="Number of bins"):
        ws2d.resample(make_image(), nbins=0)


def test_resample_returns_flux_density_per_wavelength(wave):
    """
    With no tilt and bins at the pixel edges, a flat image in counts per pixel becomes
    counts per wavelength unit divided by the local dispersion, row by row.
    """
    ws = WavelengthSolution2D(make_tilt(c0_0=REF_X, c1_0=1.0), wave)
    image = NDData(
        np.full((NY, NX), 100.0) * u.ct, uncertainty=StdDevUncertainty(np.full((NY, NX), 4.0))
    )
    edges = wave.p2w(np.arange(NX + 1) - 0.5)
    dldx = np.diff(edges)

    result = ws.resample(image, bin_edges=edges)
    assert result.flux.unit == u.ct / u.angstrom
    expected_flux = np.broadcast_to(100.0 / dldx, (NY, NX))
    expected_sigma = np.broadcast_to(4.0 / dldx, (NY, NX))
    # the outermost bins see the interpolated inverse at the detector edge
    np.testing.assert_allclose(result.flux.value[:, 1:-1], expected_flux[:, 1:-1], rtol=1e-8)
    np.testing.assert_allclose(result.flux.value, expected_flux, rtol=1e-4)
    np.testing.assert_allclose(result.uncertainty.array, expected_sigma, rtol=1e-4)


def test_resample_handles_decreasing_dispersion():
    """Wavelength decreasing with pixel number gives an ascending, flux-conserving result."""
    dispersion = 2.0
    wave = WavelengthSolution1D(
        models.Shift(-REF_X) | models.Polynomial1D(1, c0=5000.0, c1=-dispersion),
        BOUNDS_PIX,
        u.angstrom,
    )
    ws = WavelengthSolution2D(make_tilt(c0_0=REF_X, c1_0=1.0), wave)
    flux = 100.0 + np.arange(NX)[None, :] * np.ones((NY, 1))
    image = NDData(flux * u.ct, uncertainty=StdDevUncertainty(np.full((NY, NX), 4.0)))

    result = ws.resample(image)
    assert np.all(np.diff(result.spectral_axis.value) > 0)
    np.testing.assert_allclose(result.flux.value, flux[:, ::-1] / dispersion, rtol=1e-8)
    np.testing.assert_allclose(result.uncertainty.array, 4.0 / dispersion, rtol=1e-8)

    result = ws.resample(image, nbins=NX // 4)
    dl = np.diff(result.spectral_axis.value)[0]
    np.testing.assert_allclose((result.flux.value * dl).sum(1), flux.sum(1), rtol=1e-8)


def test_resample_copies_meta(ws2d):
    meta = {"OBJECT": "target", "HISTORY": ["reduced"]}
    image = NDData(make_image().data * u.ct, meta=meta)
    result = ws2d.resample(image, nbins=50)
    assert result.meta == meta
    assert result.meta is not meta
    result.meta["HISTORY"].append("resampled")
    assert meta["HISTORY"] == ["reduced"]
