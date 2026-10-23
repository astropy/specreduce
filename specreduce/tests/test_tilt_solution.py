import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling.models import Shift, Polynomial2D
from astropy.nddata import (
    NDData,
    CCDData,
    StdDevUncertainty,
    VarianceUncertainty,
    InverseVariance,
)

from specreduce.tilt_solution import TiltSolution, diff_poly2d_x


def _linear_ts(ny, nx, shift=0.0, disp_axis=1):
    """A fit-free tilt solution: detector x = rectified x + shift, for every row."""
    solution = Shift(0) & Shift(0) | Polynomial2D(1, c0_0=shift, c1_0=1.0)
    return TiltSolution(solution, disp_axis=disp_axis, image_shape=(ny, nx))


def test_diff_poly2d_x_valid_derivative():
    model = models.Polynomial2D(degree=2, c0_0=1, c1_0=2, c2_0=3, c0_1=4, c1_1=5, c0_2=6)
    derivative = diff_poly2d_x(model)
    assert derivative.degree == 1
    assert derivative.c0_0 == 2
    assert derivative.c1_0 == 6
    assert derivative.c0_1 == 5


@pytest.mark.remote_data
def test_tilt_solution_gwcs(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution
    wcs = ts.gwcs

    # Verify frame names
    assert wcs.input_frame.name == "rectified"
    assert wcs.output_frame.name == "detector"

    # Verify forward transform matches rec_to_det
    disp_arr = np.array([100.0, 200.0, 300.0])
    cdisp_arr = np.array([30.0, 60.0, 90.0])
    det_x_expected, det_y_expected = ts.corr_to_det(disp_arr, cdisp_arr)
    det_x_gwcs, det_y_gwcs = wcs(disp_arr, cdisp_arr)
    np.testing.assert_allclose(det_x_gwcs, det_x_expected)
    np.testing.assert_allclose(det_y_gwcs, det_y_expected)

    # Verify cdisp passes through unchanged
    np.testing.assert_allclose(det_y_gwcs, cdisp_arr)


@pytest.mark.remote_data
def test_tilt_solution_gwcs_cache_invalidation(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Access gwcs to populate the cache
    _ = ts.gwcs
    assert "gwcs" in ts.__dict__

    # Setting cor2det should invalidate the gwcs cache
    ts.c2d = ts.c2d
    assert "gwcs" not in ts.__dict__


@pytest.mark.remote_data
def test_det_to_corr_round_trip(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Test round-trip: corr_to_det then det_to_corr should recover original coordinates
    disp_cor = np.array([100.0, 200.0, 300.0, 400.0])
    cdisp = np.array([20.0, 40.0, 80.0, 100.0])

    disp_det, cdisp_out = ts.corr_to_det(disp_cor, cdisp)
    disp_cor_recovered, cdisp_out2 = ts.det_to_corr(disp_det, cdisp_out)

    np.testing.assert_allclose(disp_cor_recovered, disp_cor, atol=0.01)
    np.testing.assert_allclose(cdisp_out2, cdisp)


@pytest.mark.remote_data
def test_d2c_cache_invalidation(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Access d2c to populate the cache
    _ = ts.d2c
    assert "d2c" in ts.__dict__

    # Setting c2d should invalidate the d2c cache
    ts.c2d = ts.c2d
    assert "d2c" not in ts.__dict__


@pytest.mark.remote_data
def test_gwcs_inverse(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    disp_cor = np.array([100.0, 200.0, 300.0])
    cdisp = np.array([30.0, 60.0, 90.0])

    # Forward: rectified -> detector via GWCS
    disp_det, cdisp_det = ts.gwcs(disp_cor, cdisp)

    # Inverse: detector -> rectified via GWCS
    disp_cor_inv, cdisp_cor_inv = ts.gwcs.invert(disp_det, cdisp_det)

    # Should match det_to_corr
    disp_rec_direct, cdisp_direct = ts.det_to_corr(disp_det, cdisp_det)
    np.testing.assert_allclose(disp_cor_inv, disp_rec_direct)
    np.testing.assert_allclose(cdisp_cor_inv, cdisp_direct)


@pytest.mark.remote_data
def test_resample(mk_default_tc, mk_arc_frames):
    arcs = mk_arc_frames
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    result = tc.solution.resample(arcs[0])
    assert isinstance(result.uncertainty, StdDevUncertainty)
    assert result.uncertainty.array.shape == result.data.shape


@pytest.mark.parametrize(
    "uncertainty_cls", [StdDevUncertainty, VarianceUncertainty, InverseVariance]
)
def test_resample_preserves_uncertainty_type(uncertainty_cls):
    ny, nx = 4, 12
    data = np.full((ny, nx), 10.0)
    image = NDData(data * u.ct, uncertainty=uncertainty_cls(np.full((ny, nx), 4.0)))
    result = _linear_ts(ny, nx).resample(image)
    assert isinstance(result.uncertainty, uncertainty_cls)
    assert result.uncertainty.array.shape == result.data.shape


def test_resample_uncertainty_identity():
    ny, nx = 4, 12
    data = np.full((ny, nx), 10.0)
    variance = np.arange(1.0, ny * nx + 1).reshape(ny, nx)
    image = NDData(data * u.ct, uncertainty=VarianceUncertainty(variance))
    result = _linear_ts(ny, nx).resample(image)
    np.testing.assert_allclose(result.data, data)
    np.testing.assert_allclose(result.uncertainty.array, variance)
    assert result.uncertainty.unit == u.ct**2


def test_resample_uncertainty_half_pixel_shift():
    """
    Each rectified bin takes half of two neighboring detector pixels, so the
    variance is 0.25 + 0.25 of the input variance, not 0.5 + 0.5. The last bin
    covers half of the last pixel only.
    """
    ny, nx = 3, 10
    data = np.full((ny, nx), 10.0)
    image = NDData(data * u.ct, uncertainty=StdDevUncertainty(np.full((ny, nx), 3.0)))
    result = _linear_ts(ny, nx, shift=0.5).resample(image)
    np.testing.assert_allclose(result.data[:, :-1], 10.0)
    np.testing.assert_allclose(result.data[:, -1], 5.0)
    np.testing.assert_allclose(result.uncertainty.array[:, :-1], np.sqrt(4.5), rtol=1e-9)
    np.testing.assert_allclose(result.uncertainty.array[:, -1], 1.5, rtol=1e-9)


@pytest.mark.parametrize(
    "image",
    [
        NDData(np.full((4, 12), 10.0) * u.ct),
        np.full((4, 12), 10.0),
        np.full((4, 12), 10.0) * u.ct,
    ],
)
def test_resample_without_uncertainty_returns_none(image):
    result = _linear_ts(4, 12).resample(image)
    assert result.uncertainty is None


def test_resample_copies_meta():
    ny, nx = 4, 12
    data = np.full((ny, nx), 10.0)
    ts = _linear_ts(ny, nx)

    meta = {"OBJECT": "target", "HISTORY": ["step 1"]}
    result = ts.resample(NDData(data * u.ct, meta=meta))
    assert result.meta == meta
    assert result.meta is not meta
    result.meta["OBJECT"] = "changed"
    result.meta["HISTORY"].append("step 2")
    assert meta == {"OBJECT": "target", "HISTORY": ["step 1"]}

    header = fits.Header([("OBJECT", "target"), ("EXPTIME", 30.0)])
    result = ts.resample(CCDData(data, unit="ct", meta=header))
    assert isinstance(result.meta, fits.Header)
    assert result.meta["OBJECT"] == "target" and result.meta["EXPTIME"] == 30.0
    result.meta["OBJECT"] = "changed"
    assert header["OBJECT"] == "target"

    assert len(ts.resample(NDData(data * u.ct)).meta) == 0


def test_resample_propagates_mask():
    ny, nx = 4, 12
    data = np.full((ny, nx), 10.0)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[2, 3] = True

    result = _linear_ts(ny, nx).resample(NDData(data * u.ct, mask=mask), mask_treatment="apply")
    assert result.mask.dtype == bool
    np.testing.assert_array_equal(result.mask, mask)

    # a half-pixel shift spreads the masked pixel over the two bins that overlap it
    result = _linear_ts(ny, nx, shift=0.5).resample(
        NDData(data * u.ct, mask=mask), mask_treatment="apply"
    )
    expected = np.zeros((ny, nx), dtype=bool)
    expected[2, 2:4] = True
    np.testing.assert_array_equal(result.mask, expected)

    # fill treatments drop the mask before resampling
    result = _linear_ts(ny, nx).resample(NDData(data * u.ct, mask=mask), mask_treatment="zero_fill")
    assert not result.mask.any()


def test_resample_disp_axis_0_propagates_arrays():
    n = 8
    data = np.arange(1.0, n * n + 1).reshape(n, n)
    variance = 2.0 * data
    mask = np.zeros((n, n), dtype=bool)
    mask[1, 5] = True
    image = NDData(data * u.ct, uncertainty=VarianceUncertainty(variance), mask=mask)

    result = _linear_ts(n, n, disp_axis=0).resample(image)
    np.testing.assert_allclose(result.data, data.T)
    np.testing.assert_allclose(result.uncertainty.array, variance.T)
    np.testing.assert_array_equal(result.mask, mask.T)

    result = _linear_ts(n, n, disp_axis=0).resample(image, nbins=2 * n)
    assert result.data.shape == result.uncertainty.array.shape == result.mask.shape == (2 * n, n)


@pytest.mark.remote_data
def test_c2d_derivative_cache_invalidation(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Access c2d_derivative to populate the cache
    _ = ts.c2d_derivative
    assert "c2d_derivative" in ts.__dict__

    # Setting c2d should invalidate the c2d_derivative cache
    ts.c2d = ts.c2d
    assert "c2d_derivative" not in ts.__dict__


@pytest.mark.remote_data
def test_inverse_without_image_shape(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Create a TiltSolution without image_shape
    ts_no_shape = TiltSolution(ts.c2d, image_shape=None)
    with pytest.raises(TypeError, match="image_shape must be provided"):
        _ = ts_no_shape.d2c


@pytest.mark.remote_data
def test_from_gwcs(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    ts = tc.solution

    # Round-trip through GWCS
    ts2 = TiltSolution.from_gwcs(ts.gwcs, image_shape=(128, 512))

    disp_arr = np.array([100.0, 200.0, 300.0])
    cdisp_arr = np.array([30.0, 60.0, 90.0])
    np.testing.assert_allclose(
        ts2.corr_to_det(disp_arr, cdisp_arr)[0],
        ts.corr_to_det(disp_arr, cdisp_arr)[0],
    )


def test_from_gwcs_invalid():
    import gwcs
    from gwcs import coordinate_frames
    import astropy.units as u
    from astropy.modeling.models import Shift, Polynomial1D

    frame_in = coordinate_frames.CoordinateFrame(
        2, ("PIXEL", "PIXEL"), (0, 1),
        axes_names=("x", "y"), unit=[u.pix, u.pix], name="in",
    )
    frame_out = coordinate_frames.CoordinateFrame(
        2, ("PIXEL", "PIXEL"), (0, 1),
        axes_names=("x", "y"), unit=[u.pix, u.pix], name="out",
    )
    # A GWCS with Shifts but a Polynomial1D instead of Polynomial2D
    transform = Shift(0) | Shift(0) | Shift(0) | Polynomial1D(1)
    wcs = gwcs.wcs.WCS([(frame_in, transform), (frame_out, None)])
    with pytest.raises(ValueError, match="2D polynomial transformation"):
        TiltSolution.from_gwcs(wcs)


@pytest.mark.remote_data
def test_resample_disp_axis_0(mk_default_tc, mk_arc_frames):
    arcs = mk_arc_frames
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)

    # Use a square crop so _parse_image works with disp_axis=0
    ny = arcs[0].data.shape[0]
    square = NDData(
        arcs[0].data[:, :ny] * u.ct, uncertainty=StdDevUncertainty(np.full((ny, ny), 5.0))
    )

    ts = tc.solution
    ts.disp_axis = 0
    result = ts.resample(square, nbins=ny)
    assert result.uncertainty.array.shape == result.data.shape
    # With disp_axis=0, output should be transposed
    assert result.data.shape[0] == ny  # nbins along axis 0
    assert result.data.shape[1] == ny  # cdisp along axis 1
