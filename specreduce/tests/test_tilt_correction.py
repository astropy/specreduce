import numpy as np
import pytest
from astropy.modeling import models
from astropy.nddata import NDData, StdDevUncertainty
from astropy.wcs import WCS

from specreduce.tilt_correction import TiltCorrection, diff_poly2d_x
from specreduce.utils.synth_data import make_2d_arc_image

# Arc frame creation code taken from Tim Pickering's example notebook
@pytest.fixture
def mk_arc_frames():

    blue_channel_header = {
        "CTYPE1": "AWAV-GRA",  # Grating dispersion function with air wavelengths
        "CUNIT1": "Angstrom",  # Dispersion units
        "CRPIX1": 1344 // 2,  # Reference pixel [pix]
        "CRVAL1": 5410,  # Reference value [Angstrom]
        "CDELT1": 1.19 * 2,  # Linear dispersion [Angstrom/pix]
        "PV1_0": 5.0e5,  # Grating density [1/m]
        "PV1_1": 1,  # Diffraction order
        "PV1_2": 8.05,  # Incident angle [deg]
        "CTYPE2": "PIXEL",  # Spatial detector coordinates
        "CUNIT2": "pix",  # Spatial units
        "CRPIX2": 1,  # Reference pixel
        "CRVAL2": 0,  # Reference value
        "CDELT2": 1,  # Spatial units per pixel
    }
    blue_channel_wcs = WCS(header=blue_channel_header)

    tilt_mod = models.Legendre1D(degree=2, c0=25, c1=0, c2=50)

    arcs = []
    for ll in (["HeI", "NeI", "XeI"], ["ArI"]):
        arc = make_2d_arc_image(
            nx=512,
            ny=128,
            linelists=ll,
            wcs=blue_channel_wcs,
            line_fwhm=3,
            tilt_func=tilt_mod,
            amplitude_scale=1e-2,
        )
        arc.wcs = None
        arc.uncertainty = StdDevUncertainty(np.full_like(arc.data, 5))
        arcs.append(arc)
    return arcs


def test_diff_poly2d_x_valid_derivative():
    model = models.Polynomial2D(degree=2, c0_0=1, c1_0=2, c2_0=3, c0_1=4, c1_1=5, c0_2=6)
    derivative = diff_poly2d_x(model)
    assert derivative.degree == 1
    assert derivative.c0_0 == 2
    assert derivative.c1_0 == 6
    assert derivative.c0_1 == 5


def test_diff_poly2d_x_zero_x_coefficients():
    model = models.Polynomial2D(degree=2, c0_0=1, c0_1=2, c0_2=3)
    derivative = diff_poly2d_x(model)
    assert derivative.degree == 1
    assert derivative.c0_0 == 0  # All x coefficients are zero, so derivative is zero
    assert derivative.c0_1 == 0


def test_init_default_params(mk_arc_frames):
    tilt_correction = TiltCorrection(ref_pixel=(128, 64), arc_frames=mk_arc_frames)
    assert tilt_correction.ref_pixel == (128, 64)
    assert tilt_correction.disp_axis == 1
    assert tilt_correction.mask_treatment == "apply"
    assert len(tilt_correction.arc_frames) == 2


def test_find_lines(mk_arc_frames):
    tc = TiltCorrection(
        ref_pixel=(256, 64), arc_frames=mk_arc_frames, cd_sample_lims=(0, 128), n_cd_samples=8
    )
    tc.find_arc_lines(3.0, 5.0)
    np.testing.assert_array_equal(tc.cd_samples, np.array([14, 28, 43, 57, 71, 85, 100, 114]))


def test_fit(mk_arc_frames):
    tc = TiltCorrection(
        ref_pixel=(256, 64), arc_frames=mk_arc_frames, cd_sample_lims=(0, 128), n_cd_samples=8
    )
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)


def test_plot_fit_quality(mk_arc_frames):
    tc = TiltCorrection(
        ref_pixel=(256, 64), arc_frames=mk_arc_frames, cd_sample_lims=(0, 128), n_cd_samples=8
    )
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.plot_fit_quality()


def test_plot_wavelength_contours(mk_arc_frames):
    tc = TiltCorrection(
        ref_pixel=(256, 64), arc_frames=mk_arc_frames, cd_sample_lims=(0, 128), n_cd_samples=8
    )
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.plot_wavelength_contours()


def test_rectify(mk_arc_frames):
    arcs = mk_arc_frames
    tc = TiltCorrection(
        ref_pixel=(256, 64), arc_frames=arcs, cd_sample_lims=(0, 128), n_cd_samples=8
    )
    tc.find_arc_lines(3.0, 5.0)
    tc.fit(4)
    tc.rectify(arcs[0])
