import numpy as np
import pytest
import astropy.units as u

from astropy.wcs import WCS
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from specutils.fitting import fit_generic_continuum

from specreduce.calibration_data import load_pypeit_calibration_lines
from specreduce.compat import Spectrum
from specreduce.extract import BoxcarExtract
from specreduce.line_matching import match_lines_wcs, find_arc_lines
from specreduce.tracing import FlatTrace
from specreduce.utils.synth_data import make_2d_arc_image


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
    match_im = make_2d_arc_image(
        nx=1400,
        ny=1024,
        linelists=['HeI', 'NeI'],
        wcs=linear_wcs,
        line_fwhm=5,
        tilt_func=tilt_mod,
        amplitude_scale=5e-4
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
