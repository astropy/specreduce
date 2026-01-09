import warnings
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from specreduce.calibration_data import load_pypeit_calibration_lines


def test_pypeit_download_httperror():
    """
    Test that HTTPError is properly handled and closed when download fails.
    """

    # Create a mock HTTPError that tracks if close() was called
    mock_error = HTTPError(
        url="http://example.com",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None
    )
    close_called = []
    original_close = mock_error.close

    def tracking_close():
        close_called.append(True)
        original_close()

    mock_error.close = tracking_close

    with patch('specreduce.calibration_data.download_file', side_effect=mock_error):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = load_pypeit_calibration_lines(['HeI'], cache=False)

            # Check that we got the expected warnings
            warning_messages = [str(warning.message) for warning in w]
            assert any("Downloading of" in msg and "failed" in msg for msg in warning_messages)

    assert result is None
    assert len(close_called) == 1, "HTTPError.close() should have been called"


@pytest.mark.remote_data
def test_pypeit_single():
    """
    Test to load a single linelist from ``pypeit`` by passing a string.
    """
    line_tab = load_pypeit_calibration_lines('HeI', cache=True, show_progress=False)
    assert line_tab is not None
    assert "HeI" in line_tab['ion']
    assert sorted(list(line_tab.columns)) == [
        'Instr',
        'NIST',
        'Source',
        'amplitude',
        'ion',
        'wavelength'
    ]


@pytest.mark.remote_data
def test_pypeit_list():
    """
    Test to load and combine a set of linelists from ``pypeit`` by passing a list.
    """
    line_tab = load_pypeit_calibration_lines(['HeI', 'NeI'], cache=True, show_progress=False)
    assert line_tab is not None
    assert "HeI" in line_tab['ion']
    assert "NeI" in line_tab['ion']


@pytest.mark.remote_data
def test_pypeit_comma_list():
    """
    Test to load and combine a set of linelists from ``pypeit`` by passing a comma-separated list.
    """
    line_tab = load_pypeit_calibration_lines("HeI, NeI", cache=True, show_progress=False)
    assert line_tab is not None
    assert "HeI" in line_tab['ion']
    assert "NeI" in line_tab['ion']


@pytest.mark.remote_data
def test_pypeit_nonexisting_lamp():
    """
    Test to make sure a warning is raised if the lamp list includes a bad lamp name.
    """
    with pytest.warns(UserWarning, match='NeJ not in the list'):
        load_pypeit_calibration_lines(["HeI", "NeJ"], cache=True, show_progress=False)


@pytest.mark.remote_data
def test_pypeit_empty():
    """
    Test to make sure None is returned if an empty list is passed.
    """
    with pytest.warns(UserWarning, match='No calibration lines'):
        line_tab = load_pypeit_calibration_lines([], cache=True, show_progress=False)
    assert line_tab is None


@pytest.mark.remote_data
def test_pypeit_none():
    """
    Test to make sure None is returned if calibration lamp list is None.
    """
    line_tab = load_pypeit_calibration_lines(None, cache=True, show_progress=False)
    assert line_tab is None


@pytest.mark.remote_data
def test_pypeit_input_validation():
    """
    Check that bad inputs for ``pypeit`` linelists raise the appropriate warnings and exceptions
    """
    with pytest.raises(ValueError, match='.*Invalid calibration lamps specification.*'):
        _ = load_pypeit_calibration_lines({}, cache=True, show_progress=False)

    with pytest.warns() as record:
        _ = load_pypeit_calibration_lines(['HeI', 'ArIII'], cache=True, show_progress=False)
        if not record:
            pytest.fails("Expected warning about nonexistant linelist for ArIII.")
