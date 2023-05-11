# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

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
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'

        from . import __version__
        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__

# Uncomment the last two lines in this block to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
#     modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
#     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# from astropy.tests.helper import enable_deprecations_as_exceptions  # noqa
# enable_deprecations_as_exceptions()
