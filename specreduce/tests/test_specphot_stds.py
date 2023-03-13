from ..calibration_data import (
    load_MAST_calspec,
    load_onedstds
)


def test_load_MAST():
    sp = load_MAST_calspec("g191b2b_005.fits", show_progress=False)
    assert sp is not None
    assert len(sp.spectral_axis) > 0


def test_load_onedstds():
    sp = load_onedstds()
    assert sp is not None
    assert len(sp.spectral_axis) > 0
