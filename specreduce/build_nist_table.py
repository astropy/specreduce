"""Build combined NIST table from txt files included in package
"""

import glob
import os
import numpy as np
import re

from astropy.table import Column, Table, vstack

def build_table(line_lists=None):
    """Build master table from NIST txt files

    Parameters
    ----------
    line_lists: list or None
        A list of line table to read in.  If set to 'None',
        it will read in tables in the NIST data directory

    Returns
    -------
    master_table: astropy table
        Table with all of the NIST line from the txt files
    """
    names = ['Intensity', 'Wavelength', 'Element', 'Reference']
    # Use packaging directory instead of relative path in the future.
    if line_lists is None:
       code_dir = os.path.dirname(os.path.realpath(__file__))
       line_lists = glob.glob(code_dir + '/datasets/line_lists/NIST/*.txt')

    tabs_to_stack = []
    for line_list in line_lists:
        try:
            t = Table.read(line_list, format='ascii', names=names)
            tabs_to_stack.append(t)
        except:
            # Use numpy to parse table that arent comma delimited.
            data = np.genfromtxt(line_list,
                                delimiter=(13, 14, 13, 16),
                                dtype=str)
            t = Table(data, names=names,
                    dtype=('S10', 'f8', 'S15' , 'S15'))
            tabs_to_stack.append(t)

    # Stack all of the tables.
    master_table = vstack(tabs_to_stack)

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

