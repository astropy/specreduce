import numpy as np
import pytest
import astropy.units as u

from astropy.wcs import WCS
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning
from specutils.fitting import fit_generic_continuum

from specreduce.calibration_data import load_pypeit_calibration_lines
from specutils import Spectrum
from specreduce.extract import BoxcarExtract
from specreduce.line_matching import match_lines_wcs, find_arc_lines, _estimate_baseline
from specreduce.tracing import FlatTrace
from specreduce.utils.synth_data import SynthImage


@pytest.fixture
def mk_test_data():
    """
    Create test data for the line matching routines.
    """
    non_linear_header = {
        'CTYPE1': 'AWAV-GRA',  # Grating dispersion function with air wavelengths
        'CUNIT1': 'Angstrom',  # Dispersion units
        'CRPIX1': 519.8,       # Reference pixel [pix]
        'CRVAL1': 7245.2,      # Reference value [Angstrom]
        'CDELT1': 2.956,       # Linear dispersion [Angstrom/pix]
        'PV1_0': 4.5e5,        # Grating density [1/m]
        'PV1_1': 1,            # Diffraction order
        'PV1_2': 27.0,         # Incident angle [deg]
        'PV1_3': 1.765,        # Reference refraction
        'PV1_4': -1.077e6,     # Refraction derivative [1/m]
        'CTYPE2': 'PIXEL',     # Spatial detector coordinates
        'CUNIT2': 'pix',       # Spatial units
        'CRPIX2': 1,           # Reference pixel
        'CRVAL2': 0,           # Reference value
        'CDELT2': 1            # Spatial units per pixel
    }
    linear_header = {
        'CTYPE1': 'AWAV',  # Grating dispersion function with air wavelengths
        'CUNIT1': 'Angstrom',  # Dispersion units
        'CRPIX1': 519.8,       # Reference pixel [pix]
        'CRVAL1': 7245.2,      # Reference value [Angstrom]
        'CDELT1': 2.956,       # Linear dispersion [Angstrom/pix]
        'CTYPE2': 'PIXEL',     # Spatial detector coordinates
        'CUNIT2': 'pix',       # Spatial units
        'CRPIX2': 1,           # Reference pixel
        'CRVAL2': 0,           # Reference value
        'CDELT2': 1            # Spatial units per pixel
    }
    non_linear_wcs = WCS(header=non_linear_header)
    linear_wcs = WCS(header=linear_header)

    tilt_mod = models.Legendre1D(degree=2, c0=50, c1=0, c2=100)
    match_im = (
        SynthImage(nx=1400, ny=1024, wcs=linear_wcs)
        .add_background(5)
        .add_arcs(
            linelists=['HeI', 'NeI'],
            line_fwhm=5,
            tilt_func=tilt_mod,
            amplitude_scale=5e-4,
        )
        .add_poisson_noise()
        .to_ccddata()
    )

    arclist = load_pypeit_calibration_lines(['HeI', 'NeI'])['wavelength']

    trace = FlatTrace(match_im, 512)
    arc_sp = BoxcarExtract(match_im, trace, width=5).spectrum
    arc_sp.uncertainty = StdDevUncertainty(np.sqrt(arc_sp.flux).value)
    continuum = fit_generic_continuum(arc_sp, median_window=51)
    arc_sub = Spectrum(
        spectral_axis=arc_sp.spectral_axis,
        flux=arc_sp.flux - continuum(arc_sp.spectral_axis)
    )
    arc_sub.uncertainty = arc_sp.uncertainty

    return linear_wcs, non_linear_wcs, arclist, arc_sub


@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
@pytest.mark.filterwarnings("ignore:Model is linear in parameters")
def test_find_arc_lines(mk_test_data):
    """
    Test the find_arc_lines routine.
    """
    _, _, _, arc_sub = mk_test_data
    lines = find_arc_lines(arc_sub, fwhm=5, window=5, noise_factor=5)
    assert len(lines) > 1

    with pytest.raises(ValueError, match="fwhm must have"):
        find_arc_lines(arc_sub, fwhm=5*u.angstrom, window=5, noise_factor=5)

    arc_sub.uncertainty = None
    lines = find_arc_lines(arc_sub, fwhm=5, window=5, noise_factor=5)
    assert len(lines) > 1


@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
@pytest.mark.filterwarnings("ignore:Model is linear in parameters")
def test_match_lines_wcs(mk_test_data):
    """
    Test the match_lines_wcs routine.
    """
    linear_wcs, _, arclist, arc_sub = mk_test_data
    lines = find_arc_lines(arc_sub, fwhm=5, window=5, noise_factor=5)
    matched_lines = match_lines_wcs(
        pixel_positions=lines['centroid'],
        catalog_wavelengths=arclist,
        spectral_wcs=linear_wcs.spectral,
        tolerance=5
    )
    assert len(matched_lines) > 1


def _mk_pedestal_spectrum(
    pedestal, line_centers, amplitude=200.0, fwhm=4.0, npix=1000, seed=1
):
    """Synthetic 1D arc spectrum: Gaussian lines on top of a (possibly varying) pedestal."""
    rng = np.random.default_rng(seed)
    x = np.arange(npix)
    sigma = fwhm * 0.4247
    flux = np.broadcast_to(np.asarray(pedestal, dtype=float), (npix,)).copy()
    for c in line_centers:
        flux += amplitude * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    flux += rng.normal(0.0, 1.0, npix)
    return Spectrum(
        flux * u.DN,
        spectral_axis=x * u.pix,
        uncertainty=StdDevUncertainty(np.ones(npix)),
    )


def test_estimate_baseline_global():
    """A global sigma-clipped median should recover a flat pedestal under bright lines."""
    spec = _mk_pedestal_spectrum(50.0, [100, 300, 500, 700, 900])
    baseline = _estimate_baseline(spec.flux.value)
    assert baseline.shape == spec.flux.shape
    np.testing.assert_allclose(baseline, 50.0, atol=1.0)


def test_estimate_baseline_windowed():
    """A windowed estimate should follow a slowly varying pedestal."""
    npix = 1000
    ramp = np.linspace(20.0, 120.0, npix)
    spec = _mk_pedestal_spectrum(ramp, [100, 300, 500, 700, 900], npix=npix)
    baseline = _estimate_baseline(spec.flux.value, window=100)
    # Away from the edges (where the interpolation is constant) the estimate tracks the ramp.
    np.testing.assert_allclose(baseline[100:-100], ramp[100:-100], atol=3.0)
    # The global estimate cannot follow the ramp.
    assert np.abs(_estimate_baseline(spec.flux.value) - ramp).max() > 30.0


@pytest.mark.parametrize("window", [0, -5, 2.5])
def test_estimate_baseline_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        _estimate_baseline(np.ones(100), window=window)


def test_find_arc_lines_subtract_baseline():
    """A pedestal above the detection threshold hides every line unless it is subtracted."""
    centers = [100, 250, 400, 550, 700, 850]
    spec = _mk_pedestal_spectrum(50.0, centers)

    # Threshold is noise_factor * sigma = 5 < 50, so the whole spectrum is one "line"
    # and specutils warns that the spectrum is not below the threshold.
    with pytest.warns(AstropyUserWarning, match="not below the threshold"):
        lines = find_arc_lines(spec, fwhm=4, noise_factor=5, subtract_baseline=False)
    assert len(lines) <= 1

    lines = find_arc_lines(spec, fwhm=4, noise_factor=5, subtract_baseline=True)
    assert len(lines) == len(centers)
    np.testing.assert_allclose(np.sort(lines["centroid"].value), centers, atol=0.1)
    # Amplitudes are measured above the baseline, not from zero.
    np.testing.assert_allclose(lines["amplitude"].value, 200.0, rtol=0.05)
    # The input spectrum is left untouched.
    assert np.median(spec.flux.value) > 40.0


def test_find_arc_lines_subtract_baseline_windowed():
    """A ramped pedestal needs the windowed baseline to recover every line."""
    centers = [100, 250, 400, 550, 700, 850]
    ramp = np.linspace(20.0, 120.0, 1000)
    spec = _mk_pedestal_spectrum(ramp, centers)

    lines = find_arc_lines(
        spec, fwhm=4, noise_factor=5, subtract_baseline=True, baseline_window=100
    )
    assert len(lines) == len(centers)
    np.testing.assert_allclose(np.sort(lines["centroid"].value), centers, atol=0.1)
