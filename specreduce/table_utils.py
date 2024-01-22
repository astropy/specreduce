"""Utility functions to parse main NIST table."""

import numpy as np
from astropy.table import Table, vstack

__all__ = []


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
