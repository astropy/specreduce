# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import astropy.units as u
import numpy as np
import pytest
from specutils import Spectrum1D


try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


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
        # packages for which version numbers are displayed when running the tests.  # noqa: E501
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'

        from . import __version__
        TESTED_VERSIONS["specreduce"] = __version__
