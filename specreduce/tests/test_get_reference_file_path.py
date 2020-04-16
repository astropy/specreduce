from ..calibration_data import get_reference_file_path


def test_get_reference_file_path():
    """
    Test to make sure a calibration reference file provided by specreduce_data can be accessed.
    """
    test_path = "extinction/apoextinct.dat"
    p = get_reference_file_path(path=test_path)
    assert(p is not None)
