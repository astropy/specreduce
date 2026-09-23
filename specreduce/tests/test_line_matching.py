import numpy as np
import pytest
import astropy.units as u

from astropy.wcs import WCS
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty, VarianceUncertainty, InverseVariance
from specutils.fitting import fit_generic_continuum

from specreduce.calibration_data import load_pypeit_calibration_lines
from specutils import Spectrum
from specreduce.extract import BoxcarExtract
from specreduce.line_matching import match_lines_wcs, find_arc_lines
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


@pytest.fixture
def synthetic_arc():
    """
    A small synthetic arc spectrum with Gaussian emission lines at known pixel positions
    and a constant standard deviation of 3 DN per pixel.
    """
    rng = np.random.default_rng(42)
    x = np.arange(500.0)
    centers = [50.0, 120.0, 210.0, 330.0, 410.0]
    flux = np.zeros_like(x)
    for c, a in zip(centers, [200.0, 80.0, 500.0, 60.0, 150.0]):
        flux += a * np.exp(-0.5 * ((x - c) / 2.1) ** 2)
    sigma = 3.0
    flux += rng.normal(0.0, sigma, x.size)
    return x * u.pix, flux * u.DN, np.full(x.size, sigma), np.array(centers)


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
@pytest.mark.filterwarnings("ignore:Spectrum is not below the threshold")
@pytest.mark.parametrize(
    "uncertainty_cls, transform",
    [
        (StdDevUncertainty, lambda s: s),
        (VarianceUncertainty, lambda s: s**2),
        (InverseVariance, lambda s: 1.0 / s**2),
    ],
)
def test_find_arc_lines_uncertainty_types(synthetic_arc, uncertainty_cls, transform):
    """
    find_arc_lines must accept any of the three astropy uncertainty types and produce
    the same lines as it does for an equivalent StdDevUncertainty.
    """
    spectral_axis, flux, sigma, _ = synthetic_arc
    reference = Spectrum(
        flux=flux, spectral_axis=spectral_axis, uncertainty=StdDevUncertainty(sigma)
    )
    spectrum = Spectrum(
        flux=flux, spectral_axis=spectral_axis, uncertainty=uncertainty_cls(transform(sigma))
    )

    expected = find_arc_lines(reference, fwhm=5, window=3, noise_factor=5)
    lines = find_arc_lines(spectrum, fwhm=5, window=3, noise_factor=5)

    assert len(expected) >= 5
    assert len(lines) == len(expected)
    np.testing.assert_allclose(lines["centroid"].value, expected["centroid"].value)
    np.testing.assert_allclose(lines["fwhm"].value, expected["fwhm"].value)
    np.testing.assert_allclose(lines["amplitude"].value, expected["amplitude"].value)
    # The input spectrum must not be modified in place.
    assert isinstance(spectrum.uncertainty, uncertainty_cls)


@pytest.mark.filterwarnings("ignore:The fit may be unsuccessful")
@pytest.mark.filterwarnings("ignore:Spectrum is not below the threshold")
@pytest.mark.parametrize("scale", [1.0 / 30.0, 10.0 / 3.0], ids=["faint", "noisy"])
def test_find_arc_lines_estimates_noise_without_uncertainty(synthetic_arc, scale):
    """
    Without an uncertainty, find_arc_lines must estimate the noise from the data so that
    the detection threshold follows the actual noise level, independent of the flux scale.
    """
    spectral_axis, flux, _, centers = synthetic_arc
    spectrum = Spectrum(flux=flux * scale, spectral_axis=spectral_axis)

    lines = find_arc_lines(spectrum, fwhm=5, window=3, noise_factor=5)

    assert len(lines) == len(centers)
    np.testing.assert_allclose(np.sort(lines["centroid"].value), centers, atol=0.5)


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
