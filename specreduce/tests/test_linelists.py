import pytest

from ..calibration_data import load_pypeit_calibration_lines


@pytest.mark.remote_data
def test_pypeit_single():
    """
    Test to load a single linelist from ``pypeit`` by passing a string.
    """
    line_tab = load_pypeit_calibration_lines('HeI', cache=True, show_progress=False)
    assert line_tab is not None
    if line_tab is not None:
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
    if line_tab is not None:
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
def test_pypeit_empty():
    """
    Test to make sure None is returned if an empty list is passed.
    """
    with pytest.warns() as record:
        line_tab = load_pypeit_calibration_lines([], cache=True, show_progress=False)
        assert line_tab is None
        assert 'No calibration lines' in record[0].message.args[0]


@pytest.mark.remote_data
def test_pypeit_input_validation():
    """
    Check that bad inputs for ``pypeit`` linelists raise the appropriate warnings and exceptions
    """
    with pytest.raises(ValueError, match=r'.*Invalid calibration lamps specification.*'):
        _ = load_pypeit_calibration_lines(42, cache=True, show_progress=False)

    with pytest.warns() as record:
        _ = load_pypeit_calibration_lines(['HeI', 'ArIII'], cache=True, show_progress=False)
        if not record:
            pytest.fails("Expected warning about nonexistant linelist for ArIII.")
