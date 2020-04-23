"""Utility functions to parse master NIST table.
"""

from astropy.table import Table, vstack
import numpy as np
from astroquery.nist import Nist #TODO: maybe make this optional only if NIST needs calling
import astropy.units as u


def query_nist(elem_list, wavelength_range_angstrom=[4000 * u.AA, 7000 * u.AA], sort_by_wavelength=False):
    """
    Queries the NIST database from astroquery and returns a simple table with wavelengths in A and element name


    Parameters
    ----------
    elem_list: list of string
        Element IDs to query from NIST
    wavelength_range_angstrom: list of quantities in u.AA
        Wavelength range to query. Default is [4000 * u.AA, 7000 * u.AA] (for dev purposes - remove later)
    sort_by_wavelength: bool, optional
        By default the returned table will be ordered by element first in the order of elem_list. Set to True to
        return a table ordered by wavelength. Default is False.

    Returns
    -------
    table: astropytable
        Table with columns "wavelength(A)" and "line"
    """

    tables = []
    for elem in elem_list:
        # TODO: need to check is temp_table empty. When that happens vstack will fail. Can happen if elem isn't/
        #  a recognised string

        temp_table = Nist.query(wavelength_range_angstrom[0],
                                wavelength_range_angstrom[1],
                                linename=elem)

        temp_wavelength_col = [wvl for wvl in temp_table['Observed'] if wvl != '--']
        temp_line_col = [elem]*len(temp_wavelength_col)
        tables.append(Table({'wavelength(A)': temp_wavelength_col, 'line': temp_line_col}))

    final_table = vstack(tables)

    # by default the table is sorted by element (in the order given in elem_list
    # but if sort_by_wavelength is true then we can reorder the table.
    if sort_by_wavelength:
        final_table.sort('wavelength(A)')

    return final_table


def sort_table_by_element(table, elem_list):
    """Build table based on list of elements

    Parameters
    ----------
    table: astropy table
        Table to sort
    elem_list: list
        list of strings to sort table by

    Returns
    -------
    element_filtered_table: astropytable
        Filtered table based on inputs
    """

    filtered_table_list = [table[np.where(table['Element'] == elem)] for elem in elem_list]
    element_filtered_table = vstack(filtered_table_list)

    return element_filtered_table


def sort_table_by_wavelength(table, min_wave, max_wave):
    """Build table off of wavelength ranges

    Parameters
    ----------
    min_wave: float
        Lower bound wavelength to filter on
    max_wave: float
        Upper bound wavelength to filter on

    Returns
    -------
    wave_filtered_table: astropytable
        Filtered table based on inputs
    """

    assert min_wave < max_wave, "Minimum wavelength greater than maximum wavelength."
    wave_filtered_table = table[
        np.where(
            (table['Wavelength'] >= min_wave) & (table['Wavelength'] <= max_wave)
        )
    ]

    return wave_filtered_table


def main():
    """A little example.
    """
    t = Table.read('data/line_lists/NIST/NIST_combined.csv', format='csv')
    elements = ['He I', 'Ne I', 'Ar I']
    sorted_by_elem = sort_table_by_element(t, elements)
    sorted_by_wave = sort_table_by_wavelength(t, 2000, 3000)

    print(sorted_by_wave)
    print(sorted_by_elem)


if __name__ == "__main__":
    main()
