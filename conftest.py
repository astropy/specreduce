"""Need to repeat the astropy header config here for tox."""

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


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
