import pytest

from specreduce.calibration_data import get_reference_file_path, get_pypeit_data_path


@pytest.mark.remote_data
def test_get_reference_file_path():
    """
    Test to make sure a calibration reference file provided by specreduce_data can be accessed.
    """
    test_path = "extinction/apoextinct.dat"
    p = get_reference_file_path(path=test_path)
    assert p is not None


@pytest.mark.remote_data
def test_get_pypeit_data_path():
    """
    Test to make sure pypeit reference data can be loaded
    """
    test_path = "arc_lines/lists/HeI_lines.dat"
    p = get_pypeit_data_path(path=test_path, show_progress=False)
    assert p is not None
