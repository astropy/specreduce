import asdf
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models, fitting
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from gwcs import coordinate_frames, wcs
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS as astropy_WCS
from specutils import Spectrum

from specreduce.wavesol1d import _diff_poly1d, WavelengthSolution1D

ref_pixel = 250.0
p2w = models.Shift(ref_pixel) | models.Polynomial1D(degree=3, c0=1, c1=0.2, c2=0.001)
pix_bounds = (0, 500)
wav_bounds = p2w(pix_bounds)


@pytest.fixture
def mk_ws_without_transform():
    return WavelengthSolution1D(None, pix_bounds, u.angstrom)


@pytest.fixture
def mk_ws_with_transform():
    return WavelengthSolution1D(p2w, pix_bounds, u.angstrom)


def _make_gra_solution(npix: int, wave_air: bool = False) -> WavelengthSolution1D:
    """Create a solution whose polynomial closely follows a true grating dispersion curve."""
    wt = astropy_WCS(naxis=1)
    wt.wcs.ctype = ["AWAV-GRA"]
    wt.wcs.cunit = ["Angstrom"]
    wt.wcs.crpix = [npix // 2]
    wt.wcs.crval = [5410.0]
    wt.wcs.cdelt = [2.4]
    wt.wcs.set_pv([(1, 0, 5.0e5), (1, 1, 1.0), (1, 2, 8.05)])
    x = np.arange(npix)
    ref_pixel = npix // 2 - 1
    lam = wt.wcs_pix2world(x[:, None] + 1.0, 1)[:, 0] * 1e10
    poly = fitting.LinearLSQFitter()(Polynomial1D(3), x - ref_pixel, lam)
    m = models.Shift(-ref_pixel) | poly
    return WavelengthSolution1D(m, (0, npix), u.angstrom, wave_air=wave_air)


@pytest.fixture(scope="module")
def gra_solution():
    return _make_gra_solution(512, wave_air=True)


@pytest.fixture(scope="module")
def gra_fitted_wcs(gra_solution):
    return gra_solution.wcs()


@pytest.fixture
def mk_spectrum():
    return Spectrum(
        flux=np.ones(pix_bounds[1]) * u.DN,
        spectral_axis=np.arange(pix_bounds[1]) * u.pix,
        uncertainty=StdDevUncertainty(np.ones(pix_bounds[1])),
    )


def test_diff_poly1d():
    p = _diff_poly1d(Polynomial1D(3, c0=1.0, c1=2.0, c2=3.0, c3=4.0))
    np.testing.assert_array_equal(p.parameters, [2.0, 6.0, 12.0])


def test_init():
    ws = WavelengthSolution1D(p2w, pix_bounds, u.angstrom)
    assert ws._p2w is p2w
    assert ws.bounds_pix == pix_bounds
    assert ws.unit == u.angstrom
    assert "w2p" not in ws.__dict__
    assert "p2d_dldx" not in ws.__dict__
    assert "gwcs" not in ws.__dict__

    # Test that the cached properties are created correctly
    ws.w2p(0.5)
    assert "w2p" in ws.__dict__
    ws.p2w_dldx(pix_bounds[0])
    assert "p2w_dldx" in ws.__dict__
    wcs = ws.gwcs  # noqa: F841
    assert "gwcs" in ws.__dict__

    # Test that the cached properties are deleted correctly
    ws.p2w = p2w
    assert "w2p" not in ws.__dict__
    assert "p2d_dldx" not in ws.__dict__
    assert "gwcs" not in ws.__dict__

    ws = WavelengthSolution1D(p2w, pix_bounds, u.micron)
    assert ws.unit == u.micron

    ws = WavelengthSolution1D(None, pix_bounds, u.angstrom)
    assert ws._p2w is None

    assert ws.wave_air is False
    ws = WavelengthSolution1D(p2w, pix_bounds, u.angstrom, wave_air=True)
    assert ws.wave_air is True


def test_resample(mk_spectrum, mk_ws_with_transform, mk_ws_without_transform):
    ws = mk_ws_with_transform
    spectrum = mk_spectrum

    # Resample a spectrum with uncertainty
    resampled = ws.resample(spectrum, nbins=50)
    assert resampled is not None
    assert len(resampled.flux) == 50
    assert resampled.flux.unit == u.DN / u.angstrom

    pix_edges = np.arange(spectrum.spectral_axis.size + 1) - 0.5
    f0 = (spectrum.flux.value * np.diff(ws._p2w(pix_edges))).sum()
    f1 = (resampled.flux.value * np.diff(resampled.spectral_axis.value)[0]).sum()
    np.testing.assert_approx_equal(f0, f1, 5)

    resampled = ws.resample(spectrum, wlbounds=wav_bounds)
    resampled = ws.resample(spectrum, bin_edges=np.linspace(*wav_bounds, num=50))

    # Resample a spectrum without uncertainty
    spectrum.uncertainty = None
    resampled = ws.resample(spectrum, nbins=50)
    assert resampled.uncertainty is not None

    ws = mk_ws_without_transform
    with pytest.raises(ValueError, match="Wavelength solution not set."):
        ws.resample(mk_spectrum)

    ws = mk_ws_with_transform
    with pytest.raises(ValueError, match="Number of bins must be non-zero and positive"):
        ws.resample(mk_spectrum, nbins=-5)


def test_pix_to_wav(mk_ws_with_transform):
    ws = mk_ws_with_transform
    pix = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(ws.pix_to_wav(pix), p2w(pix))

    pix = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
    wav = ws.pix_to_wav(pix)
    np.testing.assert_array_equal(wav.data, p2w(pix.data))
    np.testing.assert_array_equal(wav.mask, np.array([0, 1, 0]))


def test_wav_to_pix(mk_ws_with_transform):
    ws = mk_ws_with_transform
    pix_values_orig = np.array([1, 2, 3, 4, 5])
    pix_values_tran = ws.wav_to_pix(ws.pix_to_wav(pix_values_orig))
    np.testing.assert_array_almost_equal(pix_values_orig, pix_values_tran)

    pix_values_orig = np.ma.masked_array([1, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])
    pix_values_tran = ws.wav_to_pix(ws.pix_to_wav(pix_values_orig))
    np.testing.assert_array_almost_equal(pix_values_orig.data, pix_values_tran.data)
    np.testing.assert_array_almost_equal(pix_values_orig.mask, pix_values_tran.mask)


def test_wcs_creates_valid_gwcs_object(mk_ws_with_transform):
    wc = mk_ws_with_transform
    wcs_obj = wc.gwcs
    assert wcs_obj is not None
    assert isinstance(wcs_obj, wcs.WCS)
    assert wcs_obj.output_frame.unit[0] == u.angstrom


def test_wcs_gra_round_trip(gra_solution, gra_fitted_wcs):
    ws, w = gra_solution, gra_fitted_wcs
    assert isinstance(w, astropy_WCS)
    assert w.wcs.ctype[0] == "AWAV-GRA"
    x = np.arange(*ws.bounds_pix)
    res_pix = (w.wcs_pix2world(x[:, None] + 1.0, 1)[:, 0] * 1e10 - ws.p2w(x)) / ws.p2w_dldx(x)
    assert np.max(np.abs(res_pix)) < 0.1


def test_wcs_pinned_keywords(gra_solution, gra_fitted_wcs):
    ws, w = gra_solution, gra_fitted_wcs
    # wcslib normalizes crval/cdelt/cunit to metres in place when WCS.set() first runs,
    # so the comparison must go through the current cunit.
    scale = (1.0 * u.Unit(w.wcs.cunit[0])).to_value(u.angstrom)
    assert w.wcs.crpix[0] == -ws._p2w[0].offset.value + 1.0
    assert w.wcs.crval[0] * scale == pytest.approx(ws._p2w[1].c0.value, rel=1e-12)
    assert w.wcs.cdelt[0] * scale == pytest.approx(ws._p2w[1].c1.value, rel=1e-12)
    pv = dict((i, v) for _, i, v in w.wcs.get_pv())
    assert pv[1] == 1.0  # diffraction order
    assert pv[3] == 1.0  # refractive index, absorbed by the fitted grating density
    assert pv[5] == 0.0  # grating rotation


def test_wcs_ctype_air_vacuum():
    ws_air = _make_gra_solution(128, wave_air=True)
    ws_vac = _make_gra_solution(128, wave_air=False)
    assert ws_air.wcs().wcs.ctype[0] == "AWAV-GRA"
    assert ws_vac.wcs().wcs.ctype[0] == "WAVE-GRI"
    assert ws_air.wcs(wave_air=False).wcs.ctype[0] == "WAVE-GRI"
    assert ws_vac.wcs(wave_air=True).wcs.ctype[0] == "AWAV-GRA"


def test_wcs_warning_on_poor_fit():
    ws = _make_gra_solution(128)
    with pytest.warns(AstropyUserWarning, match="lossless"):
        w = ws.wcs(max_residual=0.0)
    assert isinstance(w, astropy_WCS)


def test_wcs_header_round_trip(gra_solution, gra_fitted_wcs):
    w = gra_fitted_wcs
    w2 = astropy_WCS(w.to_header())
    x = np.arange(*gra_solution.bounds_pix)[:, None] + 1.0
    np.testing.assert_allclose(w2.wcs_pix2world(x, 1), w.wcs_pix2world(x, 1), rtol=1e-12)


def test_wcs_raises_without_transform(mk_ws_without_transform):
    with pytest.raises(ValueError, match="Wavelength solution not set."):
        mk_ws_without_transform.wcs()


def test_wcs_raises_for_non_length_unit():
    ws = WavelengthSolution1D(p2w, pix_bounds, u.Hz)
    with pytest.raises(ValueError, match="length unit"):
        ws.wcs()


def _assert_wcs_matches_solution(ws, scale, npix=512, limit=0.5):
    w = ws.wcs()
    x = np.arange(0, npix)
    lam_fit = w.wcs_pix2world(x[:, None] + 1.0, 1)[:, 0] * scale
    assert np.all(np.isfinite(lam_fit))
    res_pix = (lam_fit - ws.p2w(x)) / ws.p2w_dldx(x)
    assert np.max(np.abs(res_pix)) < limit


def test_wcs_red_ir_solution():
    # A K-band solution: the grating fit must converge far from the optical regime.
    poly = Polynomial1D(3, c0=24000.0, c1=2.4, c2=5e-5, c3=-2e-8)
    ws = WavelengthSolution1D(models.Shift(-255) | poly, (0, 512), u.angstrom)
    _assert_wcs_matches_solution(ws, 1e10)


def test_wcs_micron_unit():
    # 'micron' has no FITS unit string of its own; export must use the FITS format ('um').
    poly = Polynomial1D(3, c0=2.4, c1=2.4e-4, c2=5e-9, c3=-2e-12)
    ws = WavelengthSolution1D(models.Shift(-255) | poly, (0, 512), u.micron)
    _assert_wcs_matches_solution(ws, 1e6)


def test_wcs_uv_vacuum_solution():
    # A vacuum axis exported as 'WAVE-GRA' would pass through wcslib's air-refraction
    # model, which is invalid below ~2000 A; vacuum solutions must use 'WAVE-GRI'.
    poly = Polynomial1D(3, c0=1600.0, c1=0.6, c2=1e-5, c3=-5e-9)
    ws = WavelengthSolution1D(models.Shift(-255) | poly, (0, 512), u.angstrom)
    assert ws.wcs().wcs.ctype[0] == "WAVE-GRI"
    _assert_wcs_matches_solution(ws, 1e10, limit=0.1)


def test_wcs_raises_on_nonfinite_solution():
    poly = Polynomial1D(3, c0=5410.0, c1=2.4, c2=np.nan, c3=0.0)
    ws = WavelengthSolution1D(models.Shift(-255) | poly, (0, 512), u.angstrom)
    with pytest.raises(ValueError, match="non-finite"):
        ws.wcs()


def test_wcs_raises_for_non_power_series_model():
    cheb = models.Chebyshev1D(3, c0=5410.0, c1=600.0, c2=5.0, c3=-1.0)
    ws = WavelengthSolution1D(models.Shift(-255) | cheb, (0, 512), u.angstrom)
    with pytest.raises(TypeError, match="Polynomial1D"):
        ws.wcs()


def test_wcs_raises_when_fit_degenerate():
    # Metre-scale "wavelengths": no grating within the parameter bounds can reproduce
    # this, so every residual evaluation is unphysical and the fit must fail loudly.
    poly = Polynomial1D(1, c0=1e10, c1=2.4)
    ws = WavelengthSolution1D(models.Shift(-255) | poly, (0, 512), u.angstrom)
    with pytest.raises(RuntimeError, match="grating dispersion"):
        ws.wcs()


def test_asdf_round_trip_is_lossless(tmp_path):
    ws = WavelengthSolution1D(p2w, pix_bounds, u.angstrom, wave_air=True)
    path = tmp_path / "solution.asdf"
    ws.to_asdf(path)
    rs = WavelengthSolution1D.from_asdf(path)

    assert rs.unit == ws.unit
    assert rs.wave_air == ws.wave_air
    assert rs.bounds_pix == ws.bounds_pix
    assert rs.ref_pixel == ws.ref_pixel
    assert isinstance(rs.p2w[1], Polynomial1D)
    np.testing.assert_array_equal(rs.p2w.parameters, ws.p2w.parameters)

    # The transform must be reproduced exactly, extrapolation outside the bounds included.
    x = np.linspace(pix_bounds[0] - 50, pix_bounds[1] + 50, 137)
    np.testing.assert_array_equal(rs.pix_to_wav(x), ws.pix_to_wav(x))
    np.testing.assert_array_equal(rs.p2w_dldx(x), ws.p2w_dldx(x))


def test_asdf_export_leaves_solution_unbounded(tmp_path, mk_ws_with_transform):
    ws = mk_ws_with_transform
    ws.to_asdf(tmp_path / "solution.asdf")

    # The pixel bounds travel as the GWCS bounding box, which GWCS stores on the forward
    # transform. Exporting must not bound the live solution, whose GWCS extrapolates.
    assert ws.gwcs.bounding_box is None
    with pytest.raises(NotImplementedError):
        ws.p2w.bounding_box

    rs = WavelengthSolution1D.from_asdf(tmp_path / "solution.asdf")
    assert rs.gwcs.bounding_box is None
    with pytest.raises(NotImplementedError):
        rs.p2w.bounding_box


def test_to_asdf_raises_without_transform(mk_ws_without_transform, tmp_path):
    with pytest.raises(ValueError, match="not set"):
        mk_ws_without_transform.to_asdf(tmp_path / "solution.asdf")


def test_from_asdf_raises_on_foreign_file(tmp_path):
    path = tmp_path / "foreign.asdf"
    asdf.AsdfFile({"something_else": 1}).write_to(path)
    with pytest.raises(ValueError, match="no wavelength solution"):
        WavelengthSolution1D.from_asdf(path)


def test_from_asdf_raises_without_bounding_box(tmp_path, mk_ws_with_transform):
    ws = mk_ws_with_transform
    path = tmp_path / "unbounded.asdf"
    tree = {"wavelength_solution": {"gwcs": ws.gwcs, "wave_air": False}}
    asdf.AsdfFile(tree).write_to(path)
    with pytest.raises(ValueError, match="no bounding box"):
        WavelengthSolution1D.from_asdf(path)


def test_from_asdf_raises_on_unsupported_transform(tmp_path):
    pixel_frame = coordinate_frames.CoordinateFrame(
        1, "SPECTRAL", (0,), axes_names=["x"], unit=[u.pix]
    )
    spectral_frame = coordinate_frames.SpectralFrame(axes_names=("wavelength",), unit=[u.angstrom])
    w = wcs.WCS([(pixel_frame, Polynomial1D(2, c0=1, c1=2)), (spectral_frame, None)])
    w.bounding_box = (pix_bounds,)

    path = tmp_path / "unsupported.asdf"
    asdf.AsdfFile({"wavelength_solution": {"gwcs": w, "wave_air": False}}).write_to(path)
    with pytest.raises(ValueError, match="shift followed by a polynomial"):
        WavelengthSolution1D.from_asdf(path)
