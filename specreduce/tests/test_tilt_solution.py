import numpy as np
import pytest
from astropy.modeling import models

from specreduce.tilt_solution import TiltSolution, diff_poly2d_x


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
    tc.solution.resample(arcs[0])


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
    from astropy.nddata import NDData
    import astropy.units as u
    ny = arcs[0].data.shape[0]
    square = NDData(arcs[0].data[:, :ny] * u.ct)

    ts = tc.solution
    ts.disp_axis = 0
    result = ts.resample(square, nbins=ny)
    # With disp_axis=0, output should be transposed
    assert result.data.shape[0] == ny  # nbins along axis 0
    assert result.data.shape[1] == ny  # cdisp along axis 1


def test_image_shape_property():
    c2d = (models.Shift(-10) & models.Shift(-5)) | models.Polynomial2D(1, c0_0=10, c1_0=1)
    assert TiltSolution(c2d, image_shape=(20, 40)).image_shape == (20, 40)
    assert TiltSolution(c2d).image_shape is None


def _make_analytic_tilt(image_shape=(20, 40), disp_axis=1):
    c2d = (models.Shift(-20) & models.Shift(-10)) | models.Polynomial2D(
        2, c0_0=20, c1_0=1.0, c0_1=0.05, c1_1=0.001
    )
    return TiltSolution(c2d, disp_axis=disp_axis, image_shape=image_shape)


def test_from_gwcs_reads_image_shape_from_bounding_box():
    from copy import deepcopy

    ts = _make_analytic_tilt()
    w = deepcopy(ts.gwcs)
    w.bounding_box = ((0, 40), (0, 20))
    ts2 = TiltSolution.from_gwcs(w)
    assert ts2.image_shape == (20, 40)
    # The restored transformation extrapolates beyond the bounding box.
    assert np.isfinite(ts2.c2d(-5.0, -5.0))
    assert np.isfinite(ts2.d2c(-5.0, -5.0))


def test_asdf_round_trip_is_lossless(tmp_path):
    ts = _make_analytic_tilt(disp_axis=0)
    path = tmp_path / "tilt.asdf"
    ts.to_asdf(path)
    ts2 = TiltSolution.from_asdf(path)

    assert ts2.image_shape == (20, 40)
    assert ts2.disp_axis == 0
    x, y = np.meshgrid(np.linspace(-5, 45, 6), np.linspace(-5, 25, 5))
    np.testing.assert_allclose(ts2.corr_to_det(x, y)[0], ts.corr_to_det(x, y)[0], rtol=1e-12)
    np.testing.assert_allclose(ts2.det_to_corr(x, y)[0], ts.det_to_corr(x, y)[0], rtol=1e-12)


def test_asdf_export_is_bounded_but_solution_is_not(tmp_path):
    import asdf

    ts = _make_analytic_tilt()
    path = tmp_path / "tilt.asdf"
    ts.to_asdf(path)
    with asdf.open(path) as af:
        w = af["tilt_solution"]["gwcs"]
        assert w.bounding_box.bounding_box() == ((0, 40), (0, 20))
        assert np.isnan(w(-5.0, 5.0)[0])
        np.testing.assert_allclose(w(10.0, 5.0)[0], ts.c2d(10.0, 5.0))
    assert np.isfinite(ts.gwcs(-5.0, 5.0)[0])
    assert np.isfinite(ts.c2d(-5.0, 5.0))


def test_asdf_round_trip_without_image_shape(tmp_path):
    ts = _make_analytic_tilt(image_shape=None)
    path = tmp_path / "tilt.asdf"
    ts.to_asdf(path)
    ts2 = TiltSolution.from_asdf(path)
    assert ts2.image_shape is None
    np.testing.assert_allclose(ts2.c2d(10.0, 5.0), ts.c2d(10.0, 5.0))


def test_from_asdf_raises_on_foreign_file(tmp_path):
    import asdf

    path = tmp_path / "foreign.asdf"
    asdf.AsdfFile({"something": 1}).write_to(path)
    with pytest.raises(ValueError, match="TiltSolution.to_asdf"):
        TiltSolution.from_asdf(path)
