import pytest

from specreduce.calibration_data import load_pypeit_calibration_lines


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
        'wave'
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
def test_pypeit_empty():
    """
    Test to make sure None is returned if an empty list is passed.
    """
    with pytest.warns(UserWarning, match='No calibration lines'):
        line_tab = load_pypeit_calibration_lines([], cache=True, show_progress=False)
    assert line_tab is None


@pytest.mark.remote_data
def test_pypeit_input_validation():
    """
    Check that bad inputs for ``pypeit`` linelists raise the appropriate warnings and exceptions
    """
    with pytest.raises(ValueError, match='.*Invalid calibration lamps specification.*'):
        load_pypeit_calibration_lines({}, cache=True, show_progress=False)

    with pytest.warns(UserWarning, match="ArIII not in the list of supported calibration line lists"):  # noqa: E501
        load_pypeit_calibration_lines(['HeI', 'ArIII'], cache=True, show_progress=False)
