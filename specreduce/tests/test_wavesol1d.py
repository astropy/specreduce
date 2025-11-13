import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models
from astropy.modeling.polynomial import Polynomial1D
from astropy.nddata import StdDevUncertainty
from gwcs import wcs
from specreduce.wavesol1d import _diff_poly1d, WavelengthSolution1D
from specreduce.compat import Spectrum


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
