# This file is used to configure the behavior of pytest

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, NDData, VarianceUncertainty
from astropy.utils.data import get_pkg_data_filename
from specutils import Spectrum1D, SpectralAxis

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


# Test image is comprised of 30 rows with 10 columns each. Row content
# is row index itself. This makes it easy to predict what should be the
# value extracted from a region centered at any arbitrary Y position.
def _mk_test_data(imgtype, nrows=30, ncols=10):
    image_ones = np.ones(shape=(nrows, ncols))
    image = image_ones.copy()
    for j in range(nrows):
        image[j, ::] *= j
    if imgtype == "raw":
        pass  # no extra processing
    elif imgtype == "ccddata":
        image = CCDData(image, unit=u.Jy)
    else:  # spectrum
        flux = image * u.DN
        uncert = VarianceUncertainty(image_ones)
        if imgtype == "spec_no_axis":
            image = Spectrum1D(flux, uncertainty=uncert)
        else:  # "spec"
            image = Spectrum1D(flux, spectral_axis=np.arange(ncols) * u.um, uncertainty=uncert)
    return image


@pytest.fixture
def mk_test_img_raw():
    return _mk_test_data("raw")


@pytest.fixture
def mk_test_img():
    return _mk_test_data("ccddata")


@pytest.fixture
def mk_test_spec_no_spectral_axis():
    return _mk_test_data("spec_no_axis")


@pytest.fixture
def mk_test_spec_with_spectral_axis():
    return _mk_test_data("spec")


# Test data file already transposed like this:
# fn = download_file('https://stsci.box.com/shared/static/exnkul627fcuhy5akf2gswytud5tazmw.fits', cache=True)  # noqa: E501
# img = fits.getdata(fn).T
@pytest.fixture
def all_images():
    np.random.seed(7)

    filename = get_pkg_data_filename(
        "data/transposed_det_image_seq5_MIRIMAGE_P750Lexp1_s2d.fits", package="specreduce.tests")
    img = fits.getdata(filename)
    flux = img * (u.MJy / u.sr)
    sax = SpectralAxis(np.linspace(14.377, 3.677, flux.shape[-1]) * u.um)
    unc = VarianceUncertainty(np.random.rand(*flux.shape))

    all_images = {}
    all_images['arr'] = img
    all_images['s1d'] = Spectrum1D(flux, spectral_axis=sax, uncertainty=unc)
    all_images['s1d_pix'] = Spectrum1D(flux, uncertainty=unc)
    all_images['ccd'] = CCDData(img, uncertainty=unc, unit=flux.unit)
    all_images['ndd'] = NDData(img, uncertainty=unc, unit=flux.unit)
    all_images['qnt'] = img * flux.unit
    return all_images


@pytest.fixture
def spec1d():
    np.random.seed(7)
    flux = np.random.random(50)*u.Jy
    sa = np.arange(0, 50)*u.pix
    spec = Spectrum1D(flux, spectral_axis=sa)
    return spec


@pytest.fixture
def spec1d_with_emission_line():
    np.random.seed(7)
    sa = np.arange(0, 200)*u.pix
    flux = (np.random.randn(200) +
            10*np.exp(-0.01*((sa.value-130)**2)) +
            sa.value/100) * u.Jy
    spec = Spectrum1D(flux, spectral_axis=sa)
    return spec


@pytest.fixture
def spec1d_with_absorption_line():
    np.random.seed(7)
    sa = np.arange(0, 200)*u.pix
    flux = (np.random.randn(200) -
            10*np.exp(-0.01*((sa.value-130)**2)) +
            sa.value/100) * u.Jy
    spec = Spectrum1D(flux, spectral_axis=sa)
    return spec


def pytest_configure(config):

    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES.pop('h5py', None)
        PYTEST_HEADER_MODULES['astropy'] = 'astropy'
        PYTEST_HEADER_MODULES['specutils'] = 'specutils'
        PYTEST_HEADER_MODULES['photutils'] = 'photutils'
        PYTEST_HEADER_MODULES['synphot'] = 'synphot'

        from specreduce import __version__
        TESTED_VERSIONS["specreduce"] = __version__
