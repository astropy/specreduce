from astropy.utils.data import download_file

from ..calibration_data import (
    load_MAST_calspec
)


def test_load_MAST_remote():
    sp = load_MAST_calspec("g191b2b_005.fits", remote=True, show_progress=False)
    assert(sp is not None)
    assert(len(sp.spectral_axis) > 0)


def test_load_MAST_local():
    sp_file = download_file(
        "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/g191b2b_005.fits",
        show_progress=False,
        pkgname='specreduce'
    )
    sp = load_MAST_calspec(sp_file, remote=False)
    assert(sp is not None)
    assert(len(sp.spectral_axis) > 0)
