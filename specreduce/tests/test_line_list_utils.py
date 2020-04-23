import ../line_list_utils as lu
#import line_list_utils as lu
import numpy as np

def test_query_nist_one_elem():
    table = lu.query_nist(['H I'])
    assert np.isclose(table['wavelength(A)'][0], 4102.86503481)

def test_query_nist_two_elem():
    table = lu.query_nist(['H I', 'He I'], sort_by_wavelength=True)
    assert np.isclose(table['wavelength(A)'][0], 4025.11)

def test_query_nist_sort_by_wvl():
    table = lu.query_nist(['H I', 'He I'], sort_by_wavelength=True)
    assert np.isclose(table['wavelength(A)'][-1], 6679.995)
