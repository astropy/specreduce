import asdf
import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from gwcs import coordinate_frames, wcs
from specreduce.wavesol1d import _diff_poly1d, WavelengthSolution1D
from specutils import Spectrum


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
