import pytest
from astropy.utils.exceptions import AstropyUserWarning

from specreduce.calibration_data import load_MAST_calspec, load_onedstds


@pytest.mark.remote_data
def test_load_MAST():
    sp = load_MAST_calspec("g191b2b_005.fits", show_progress=False)
    assert sp is not None
    assert len(sp.spectral_axis) > 0


@pytest.mark.remote_data
def test_load_MAST_bad_filename():
    with pytest.warns(AstropyUserWarning, match="Downloading of"):
        load_MAST_calspec("j191b2b_005.fits", show_progress=False)


@pytest.mark.remote_data
def test_load_onedstds():
    sp = load_onedstds()
    assert sp is not None
    assert len(sp.spectral_axis) > 0


@pytest.mark.remote_data
def test_load_onedstds_bad_dataset():
    with pytest.warns(AstropyUserWarning, match="Specfied dataset,"):
        sp = load_onedstds("snffactory")
    assert sp is None


@pytest.mark.remote_data
def test_load_onedstds_bad_specfile():
    with pytest.warns(AstropyUserWarning, match="Can't load"):
        sp = load_onedstds(specfile="FG131.dat")
    assert sp is None
