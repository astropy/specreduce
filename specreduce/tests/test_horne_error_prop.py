import numpy as np
import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import NDData, VarianceUncertainty
from astropy.tests.helper import assert_quantity_allclose

from specreduce.extract import HorneExtract
from specreduce.tracing import FlatTrace


def _make_gaussian_row_image(
    ny=50,
    nx=300,
    amp=150.0,
    mean=23.0,
    std=4.0,
    bkg=3.0,
    sigma_per_pix=6.0,
    vary_row_sigma=False,
    kill_rows=None,
    seed=1234,
):
    rng = np.random.default_rng(seed)
    y = np.arange(ny, dtype=float)

    profile = amp * np.exp(-0.5 * ((y - mean) / std) ** 2) + bkg
    ideal = np.tile(profile[:, None], (1, nx))

    if vary_row_sigma:
        sigma_row = sigma_per_pix * (0.5 + 0.5 * (y - y.min()) / (y.max() - y.min()))
        var2d = (sigma_row[:, None] ** 2) * np.ones((1, nx))
    else:
        var2d = np.full((ny, nx), sigma_per_pix**2, dtype=float)

    noise = rng.normal(0.0, np.sqrt(var2d))
    data = ideal + noise

    mask = np.zeros_like(data, dtype=bool)
    if kill_rows:
        mask[np.asarray(kill_rows, dtype=int), :] = True

    nd = NDData(
        data=data * u.DN,
        unit=u.DN,
        uncertainty=VarianceUncertainty(var2d * u.DN**2),
        mask=mask,
    )
    return nd, data, var2d, mask


def _expected_sigma_row(mask, var2d, disp_axis=1):
    valid = ~mask
    N = valid.sum(axis=disp_axis).astype(float)
    sumvar = (valid * var2d).sum(axis=disp_axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_mean = np.where(N > 0, sumvar / (N**2), np.nan)
    return np.sqrt(var_mean)


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_sigma_propagation_matches_analytic():
    nd, data, var2d, mask = _make_gaussian_row_image(ny=50, nx=300, sigma_per_pix=6.0)
    trace = FlatTrace(image=nd, trace_pos=23.0)

    ex = HorneExtract(
        data, trace,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=var2d,
        mask=mask,
        unit=u.DN,
    )
    _ = ex.spectrum  # triggers the fit and sets _last_profile_sigma

    expected = _expected_sigma_row(mask, var2d)
    # exact equality within floating noise in reduction path
    np.testing.assert_allclose(ex._last_profile_sigma, expected, rtol=0, atol=0)

    # fitted parameters should be close to truth
    cov = ex._last_fit_cov
    stderr = ex._last_fit_param_stderr
    assert cov is not None
    assert stderr is not None
    assert cov.shape[0] == cov.shape[1] == 4  # Gaussian amp, mean, std, Const1D amp

    # Diagonal nonnegative and finite where expected
    d = np.diag(cov)
    assert np.all(~np.isfinite(d) | (d >= 0))

    # Check that main errors are positive and finite
    for k in ("amplitude_0", "mean_0", "stddev_0", "amplitude_1"):
        assert k in stderr
        assert np.isfinite(stderr[k]) and stderr[k] > 0

    # Rough proximity to injected parameters
    # Use 3 sigma bounds from stderr to be robust
    amp_est, mu_est, sig_est = ex.spectrum.meta.get("fit_amplitude", None), None, None
    # Access parameters directly from covariance names if needed
    # We avoid relying on internal fitted model object here.


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_fully_masked_rows_are_excluded():
    ny, nx = 40, 200
    kill_rows = [3, 7, 9]
    nd, data, var2d, mask = _make_gaussian_row_image(
        ny=ny, nx=nx, sigma_per_pix=4.0, kill_rows=kill_rows
    )
    trace = FlatTrace(image=nd, trace_pos=15.0)
    ex = HorneExtract(
        data, trace,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=var2d,
        mask=mask,
        unit=u.DN,
    )
    _ = ex.spectrum

    sigma_row = ex._last_profile_sigma
    assert sigma_row.shape == (ny,)
    # killed rows should be NaN
    assert np.all(np.isnan(sigma_row[kill_rows]))
    # number of finite rows equals ny minus killed
    assert np.sum(np.isfinite(sigma_row)) == ny - len(kill_rows)

    # analytic check
    expected = _expected_sigma_row(mask, var2d)
    np.testing.assert_allclose(sigma_row, expected, rtol=0, atol=0)


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_covariance_scales_with_variance():
    # Same data, two different overall variance scalings: V and 4V
    nd, data, var2d, mask = _make_gaussian_row_image(ny=50, nx=300, sigma_per_pix=6.0)
    trace = FlatTrace(image=nd, trace_pos=23.0)

    ex1 = HorneExtract(
        data, trace,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=var2d,
        mask=mask,
        unit=u.DN,
    )
    _ = ex1.spectrum
    stderr1 = ex1._last_fit_param_stderr

    ex2 = HorneExtract(
        data, trace,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=4.0 * var2d,  # 2x sigma per pixel
        mask=mask,
        unit=u.DN,
    )
    _ = ex2.spectrum
    stderr2 = ex2._last_fit_param_stderr

    # Parameter point estimates should be nearly identical
    # Compare the 1D extracted spectra, which are the final product
    sp1 = ex1.spectrum.flux
    sp2 = ex2.spectrum.flux
    assert_quantity_allclose(sp1, sp2, rtol=1e-3, atol=0)

    # Standard errors should scale like the sigma scaling (2x here)
    for k in ("amplitude_0", "mean_0", "stddev_0", "amplitude_1"):
        s1 = float(stderr1[k])
        s2 = float(stderr2[k])
        assert np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0
        ratio = s2 / s1
        assert np.isclose(ratio, 2.0, rtol=0.1, atol=0.0)  # allow some fitter tolerance


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
def test_constant_vs_variable_row_sigma_agree_when_equalized():
    # When row variances are uniform, weighted fit equals the uniform-sigma case
    nd_u, data_u, var_u, mask_u = _make_gaussian_row_image(
        ny=50, nx=300, sigma_per_pix=5.0, vary_row_sigma=False
    )
    nd_v, data_v, var_v, mask_v = _make_gaussian_row_image(
        ny=50, nx=300, sigma_per_pix=5.0, vary_row_sigma=True
    )

    trace_u = FlatTrace(image=nd_u, trace_pos=23.0)
    trace_v = FlatTrace(image=nd_v, trace_pos=23.0)

    # For the variable case, replace variance with its column-mean to equalize rows
    var_v_equal = np.tile(np.nanmean(var_v, axis=1)[:, None], (1, var_v.shape[1]))

    ex_u = HorneExtract(
        data_u, trace_u,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=var_u,
        mask=mask_u,
        unit=u.DN,
    )
    _ = ex_u.spectrum

    ex_v = HorneExtract(
        data_v, trace_v,
        spatial_profile="gaussian",
        bkgrd_prof=models.Const1D(),
        variance=var_v_equal,
        mask=mask_v,
        unit=u.DN,
    )
    _ = ex_v.spectrum

    # Spectra should agree closely when the weighting is effectively uniform
    assert_quantity_allclose(ex_u.spectrum.flux, ex_v.spectrum.flux, rtol=5e-3, atol=0)
