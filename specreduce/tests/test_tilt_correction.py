from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from specreduce.tilt_correction import TiltCorrection
from specreduce.tracing import FlatTrace


@pytest.mark.remote_data
def test_init_trace(mk_arc_frames):
    arcs = mk_arc_frames
    trace = FlatTrace(arcs[0], arcs[0].shape[0] // 2)
    tc = TiltCorrection(arc_frames=arcs, trace=trace)
    assert tc.ref_pixel == (arcs[0].shape[0] // 2, arcs[0].shape[1] // 2)


@pytest.mark.remote_data
def test_init_default_params(mk_arc_frames):
    arcs = mk_arc_frames
    tc = TiltCorrection(arcs, cdisp_ref_pixel=64, disp_ref_pixel=256)
    assert tc.ref_pixel == (64, 256)
    assert tc.disp_axis == 1
    assert tc.mask_treatment == "apply"
    assert len(tc.arc_frames) == 2

    tc = TiltCorrection(arcs, cdisp_ref_pixel=64)
    assert tc.ref_pixel == (64, arcs[0].shape[1] // 2)

    tc = TiltCorrection(arcs[0], cdisp_ref_pixel=64)
    assert tc.ref_pixel == (64, arcs[0].shape[1] // 2)

    with pytest.raises(ValueError, match="cdisp_ref_position must be provided"):
        TiltCorrection(arcs)

    tc = TiltCorrection(arcs[0], cdisp_ref_pixel=64, cdisp_samples=[10, 20, 30, 40, 50])
    np.testing.assert_array_equal(tc.cd_samples, np.array([10, 20, 30, 40, 50]))


@pytest.mark.remote_data
def test_find_lines(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    np.testing.assert_array_equal(tc.cd_samples, np.array([14, 28, 43, 57, 71, 85, 100, 114]))


@pytest.mark.remote_data
def test_fit(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)


@pytest.mark.remote_data
def test_plot_fit_quality(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.plot_fit_quality()


@pytest.mark.remote_data
def test_plot_wavelength_contours(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.plot_wavelength_contours()


@pytest.mark.remote_data
def test_refine_fit_before_fit(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    with pytest.raises(ValueError, match="solution must be calculated"):
        tc.refine_fit()


@pytest.mark.remote_data
def test_match_lines_before_fit(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    with pytest.raises(ValueError, match="solution must be calculated"):
        tc.match_lines()


@pytest.mark.remote_data
def test_plot_wavelength_contours_options(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)

    # Test with line_args and a pre-created ax
    fig, ax = plt.subplots()
    tc.plot_wavelength_contours(ax=ax, line_args={"c": "red"})

    # Test with explicit disp_values
    disp_values = np.array([50.0, 100.0, 200.0, 300.0, 400.0])
    tc.plot_wavelength_contours(disp_values=disp_values)
    plt.close("all")


@pytest.mark.remote_data
def test_plot_fit_quality_with_rlim(mk_default_tc):
    tc = mk_default_tc
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.plot_fit_quality(rlim=(-1, 1))
    plt.close("all")


@pytest.mark.remote_data
def test_find_lines_bright_background(mk_arc_frames):
    """A pedestal above the detection threshold hides the lines unless it is subtracted."""
    arcs = mk_arc_frames
    reference = TiltCorrection(arc_frames=arcs, cdisp_ref_pixel=64, n_cdisp_samples=8)
    reference.find_arc_lines(3.0, 5.0)

    # Add a pedestal of 40 sigma (uncertainty is 5) on top of the arc lines.
    bright = [deepcopy(arc) for arc in arcs]
    for arc in bright:
        arc.data = arc.data + 200.0
    tc = TiltCorrection(arc_frames=bright, cdisp_ref_pixel=64, n_cdisp_samples=8)

    # Without baseline subtraction the whole row is above threshold: one "line" at most.
    tc.find_arc_lines(3.0, 5.0, subtract_baseline=False)
    assert all(lines.size <= 1 for lines in tc._lines_ref)

    # With the default baseline subtraction the result is the same as for the faint frames.
    tc.find_arc_lines(3.0, 5.0)
    for ref_lines, lines, det_x in zip(reference._lines_ref, tc._lines_ref, tc._samples_det_x):
        assert lines.size > 1
        np.testing.assert_allclose(lines, ref_lines, atol=0.05)
        assert det_x.size >= 0.8 * lines.size * tc.cd_samples.size

    # A running baseline estimate works too.
    tc.find_arc_lines(3.0, 5.0, baseline_window=64)
    for ref_lines, lines in zip(reference._lines_ref, tc._lines_ref):
        np.testing.assert_allclose(lines, ref_lines, atol=0.2)
