"""Build combined NIST table from txt files included in package
"""

from astropy.table import Column, Table, vstack
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import re


def build_table():

    names = ['Intensity', 'Wavelength', 'Element', 'Reference']

    # Use packaging directory instead of relative path in the future.
    line_lists = glob.glob('data/line_lists/NIST/*.txt')

    master_table = vstack([Table.read(line_list, format='ascii', names=names) for line_list in line_lists])

    # Add on switch for users. Use line if True, don't if False
    # Set to True by default.
    on_off_column = Column([True] * len(master_table))
    master_table.add_column(on_off_column, name='On')
    
    # Strip the numeric characters off of the intensities and add the letters
    # that denote intensities to their own column
    intensity = master_table['Intensity']
    strength = [re.sub('[0-9]+', '', value).strip() for value in intensity]    
    master_table.add_column(Column(strength), name='Strength')

    # Find and strip all alphabetic + special characters
    intensity_wo_strength = [re.sub('[a-zA-Z!@#$%^&*]', '', value).strip() \
                             for value in intensity]
    
    # Delete old column
    master_table.remove_column('Intensity')

    # Add new Intensity column that only has intensity as an integer.
    master_table.add_column(Column(intensity_wo_strength, 
                                   dtype=int,
                                   name='Intensity'))
    
    # Reorder table columns
    neworder = ('Element','Wavelength','Intensity', 'Strength', 'On', 'Reference')
    master_table = master_table[neworder]

    return master_table

def save_table(location, table_name, table):
    """Save parsed table to location

    Parameters
    ----------
    location: str
        Location you wish to save table to
    table: astropytable

    Returns
    -------
    None
    """

    table.write(os.path.join(location, table_name), format='csv')


def main():
    t = build_table()
    save_table('data/line_lists/NIST/', 'NIST_combined.csv', t)

if __name__ == "__main__":
    main()