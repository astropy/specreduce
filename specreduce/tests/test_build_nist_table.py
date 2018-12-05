

import pytest

from ..build_nist_table import build_table

def test_columns():
    nist = build_table()
    assert nist.columns.keys() == ['Element', 'Wavelength', 'Intensity', 'Strength', 'On', 'Reference']
    
