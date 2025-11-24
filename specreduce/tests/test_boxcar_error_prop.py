import numpy as np
import pytest
import astropy.units as u
from astropy.nddata import NDData, CCDData, VarianceUncertainty, StdDevUncertainty

from specreduce.extract import BoxcarExtract
from specreduce.tracing import FlatTrace, ArrayTrace


ny, nx = 20, 60
true_flux = 100.0
sigma_per_pixel = 5.0
width = 5
rng = np.random.default_rng(42)
data = true_flux + rng.normal(0, sigma_per_pixel, size=(ny, nx))


def make_image(uncertainty, data_override=None, mask=None):
    """Synthetic image parameters."""
    use_data = data if data_override is None else data_override
    img = NDData(data=use_data * u.DN, uncertainty=uncertainty, mask=mask)
    img.spectral_axis = np.arange(nx) * u.pix
    return img


def n_in_window(trace_pos, w, nrows):
    """Expected-sigma helpers."""
    half = w // 2
    y0 = int(trace_pos)
    y1 = max(0, y0 - half)
    y2 = min(nrows, y0 + half + (1 if (w % 2) else 0))
    return max(0, y2 - y1)


def n_unmasked_in_window(mask, trace_pos, w):
    """Expected-sigma helpers."""
    half = w // 2
    y0 = int(trace_pos)
    y1 = max(0, y0 - half)
    y2 = min(mask.shape[0], y0 + half + (1 if (w % 2) else 0))
    return (~mask[y1:y2, :]).sum(axis=0)


def expected_std_per_column(mode, trace_pos, w, mask=None, sigma=sigma_per_pixel):
    """Expected-sigma helpers."""
    if mode == "apply":
        if mask is None:
            n_eff = n_in_window(trace_pos, w, ny)
            return np.full(nx, sigma * w / np.sqrt(max(n_eff, 1)))
        n_um = n_unmasked_in_window(mask, trace_pos, w).astype(float)
        n_um[n_um <= 0] = np.nan
        return sigma * w / np.sqrt(n_um)
    else:
        n_eff = n_in_window(trace_pos, w, ny)
        return np.full(nx, sigma * np.sqrt(max(n_eff, 1)))


def run_extract(image, trace_pos=None, mask_treatment="ignore"):
    """Case builders."""
    tpos = ny // 2 if trace_pos is None else trace_pos
    trace = FlatTrace(image, trace_pos=tpos)
    extractor = BoxcarExtract(
        image=image,
        trace_object=trace,
        width=width,
        disp_axis=1,
        crossdisp_axis=0,
        mask_treatment=mask_treatment,
    )
    spec = extractor()
    assert getattr(spec, "uncertainty", None) is not None, "Extractor returned no uncertainty"
    return np.asarray(spec.uncertainty.array)


var_unc        = VarianceUncertainty(np.full((ny, nx), sigma_per_pixel**2) * (u.DN**2))
std_unc        = StdDevUncertainty(np.full((ny, nx), sigma_per_pixel) * u.DN)
std_unc_nounit = StdDevUncertainty(np.full((ny, nx), sigma_per_pixel))
neg_var        = VarianceUncertainty(np.full((ny, nx), -1.0) * (u.DN**2))
zero_var_all   = VarianceUncertainty(np.zeros((ny, nx)) * (u.DN**2))

_some_zero_var = np.full((ny, nx), sigma_per_pixel**2) * (u.DN**2)
_some_zero_var[0, :] = 0.0 * (u.DN**2)
some_zero_unc = VarianceUncertainty(_some_zero_var)

half_mask = np.zeros((ny, nx), dtype=bool)
half_mask[::2, :] = True

bad_data = data.copy()
bad_data[ny // 2, 5] = np.nan
bad_data[ny // 2, 8] = np.inf


def test_boxcar_variance_uncertainty_matches_expectation():
    """Basic uncertainty wiring."""
    img = make_image(var_unc)
    arr = run_extract(img, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_boxcar_stddev_uncertainty_matches_variance_case():
    """Basic uncertainty wiring."""
    img = make_image(std_unc)
    arr = run_extract(img, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_boxcar_unitless_stddev_matches():
    """Basic uncertainty wiring."""
    img = make_image(std_unc_nounit)
    arr = run_extract(img, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_boxcar_missing_uncertainty_errors():
    """Basic uncertainty wiring."""
    img = NDData(data=data * u.DN)
    with pytest.raises(Exception):
        run_extract(img, mask_treatment="ignore")


@pytest.mark.parametrize(
    "mode, use_halfmask, expect_vector, expect_nans",
    [
        ("apply", True, True, False),
        ("ignore", False, True, False),
        ("zero_fill", False, True, False),
        ("apply_mask_only", False, True, False),
        ("apply_nan_only", False, True, False),
        ("nan_fill", False, False, True),
    ],
)
def test_boxcar_mask_treatments(mode, use_halfmask, expect_vector, expect_nans):
    """Mask_treatment modes."""
    img = make_image(var_unc, data_override=bad_data, mask=half_mask if use_halfmask else None)
    if mode == "apply":
        mask = half_mask
    else:
        mask = None

    if mode == "nan_fill":
        arr = run_extract(img, mask_treatment=mode)
        assert np.isnan(arr).any()
        return

    if mode in ("ignore", "zero_fill", "apply_mask_only", "apply_nan_only"):
        arr = run_extract(img, mask_treatment=mode)
        exp = expected_std_per_column("ignore", ny // 2, width)
        assert np.all(np.isfinite(arr))
        assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)
        return

    if mode == "apply":
        arr = run_extract(img, mask_treatment=mode)
        exp = expected_std_per_column("apply", ny // 2, width, mask=mask)
        finite = np.isfinite(exp)
        assert np.allclose(arr[finite], exp[finite], rtol=5e-3, atol=5e-3)
        return


def test_boxcar_propagate_is_finite():
    """Mask_treatment modes."""
    img = make_image(var_unc, data_override=bad_data)
    arr = run_extract(img, mask_treatment="propagate")
    assert arr.shape == (nx,)
    assert np.all(np.isfinite(arr))


def test_boxcar_edge_top_clipped_window():
    """Aperture partly off the detector."""
    img = make_image(var_unc)
    arr = run_extract(img, trace_pos=1, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", 1, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)
    assert exp[0] < sigma_per_pixel * np.sqrt(width)


def test_boxcar_edge_bottom_clipped_window():
    """Aperture partly off the detector."""
    img = make_image(var_unc)
    arr = run_extract(img, trace_pos=ny - 2, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny - 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_boxcar_fully_masked_input_raises():
    """Fully masked input."""
    masked_all = np.ones((ny, nx), dtype=bool)
    img = make_image(var_unc, mask=masked_all)
    with pytest.raises(Exception):
        run_extract(img, mask_treatment="ignore")


def test_boxcar_arraytrace_equivalence():
    """ArrayTrace equivalence."""
    img = make_image(var_unc)
    trace_array = np.full(nx, ny // 2)
    trace = ArrayTrace(img, trace_array)
    extractor = BoxcarExtract(image=img, trace_object=trace, width=width, disp_axis=1, crossdisp_axis=0)
    spec = extractor()
    arr = np.asarray(spec.uncertainty.array)
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_boxcar_apply_half_mask_normalization():
    """Apply normalization with half masked rows."""
    img = make_image(var_unc, mask=half_mask)
    arr = run_extract(img, mask_treatment="apply")
    exp = expected_std_per_column("apply", ny // 2, width, mask=half_mask)
    finite = np.isfinite(exp)
    assert np.allclose(arr[finite], exp[finite], rtol=5e-3, atol=5e-3)


def test_boxcar_negative_variance_rejected():
    """Additional variance edge cases."""
    img = make_image(neg_var)
    with pytest.raises(Exception):
        run_extract(img, mask_treatment="ignore")


def test_boxcar_zero_variance_allows_finite_output():
    """Additional variance edge cases."""
    img = make_image(zero_var_all)
    arr = run_extract(img, mask_treatment="ignore")
    assert np.all(np.isfinite(arr))


def test_boxcar_some_zero_variances_row():
    """Additional variance edge cases."""
    img = make_image(some_zero_unc)
    arr = run_extract(img, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


# Extra coverage for policy and units

def test_ccddata_without_uncertainty_runs_unweighted():
    img = CCDData(data=data.copy(), unit=u.DN)  # no uncertainty on purpose
    img.spectral_axis = np.arange(nx) * u.pix
    arr_u = run_extract(img, mask_treatment="ignore")

    # estimate sigma from the image the same way as the code path
    arr = img.data.astype(float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    baseline = np.nanmedian(arr, axis=0, keepdims=True)
    resid = arr - baseline
    sigma_hat = float(np.nanstd(resid, ddof=0))

    exp = expected_std_per_column("ignore", ny // 2, width, sigma=sigma_hat)
    assert np.allclose(arr_u, exp, rtol=5e-3, atol=5e-3)

def test_missing_uncertainty_policy_freezes_across_calls():
    img = CCDData(data=data.copy(), unit=u.DN)  # no uncertainty
    img.spectral_axis = np.arange(nx) * u.pix
    trace = FlatTrace(img, ny // 2)
    ext = BoxcarExtract(image=img, trace_object=trace, width=width, disp_axis=1, crossdisp_axis=0)
    spec1 = ext.spectrum  # first call establishes policy
    trace.set_position(ny // 2 - 0.5)
    spec2 = ext()         # second call should not flip to error
    uq1 = getattr(spec1.uncertainty, "quantity", None)
    uq2 = getattr(spec2.uncertainty, "quantity", None)
    assert uq1 is not None and uq1.unit == u.DN
    assert uq2 is not None and uq2.unit == u.DN


def test_variance_unitless_is_interpreted_as_image_units_squared():
    var_unitless = VarianceUncertainty(np.full((ny, nx), sigma_per_pixel**2))  # no units
    img = make_image(var_unitless)
    arr = run_extract(img, mask_treatment="ignore")
    exp = expected_std_per_column("ignore", ny // 2, width)
    assert np.allclose(arr, exp, rtol=5e-3, atol=5e-3)


def test_uncertainty_units_are_stddev_units():
    img = make_image(StdDevUncertainty(np.full((ny, nx), sigma_per_pixel) * u.DN))
    trace = FlatTrace(img, ny // 2)
    spec = BoxcarExtract(image=img, trace_object=trace, width=width, disp_axis=1, crossdisp_axis=0)()
    uq = getattr(spec.uncertainty, "quantity", None)
    assert uq is not None and uq.unit == u.DN
