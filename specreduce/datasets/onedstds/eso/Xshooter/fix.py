#!/usr/bin/env python

"""
the original data from the ESO ftp site does not include the 3rd column containing the bin width.
this script recreates that using np.diff() on the wavelength axis and outputs an IRAF format
file with columns for wavelength, AB mag, and bin width.
"""

import sys
import numpy as np

from astropy.table import Table


if __name__ == "__main__":
    infile = sys.argv[1]
    if infile[0] == 'm':
        outfile = infile[1:]  # trim the leading 'm'
    else:
        outfile = infile

    t = Table.read(infile, format='ascii')
    diff = np.diff(t['col1'].data)
    diff = np.append(diff, diff[-1])  # hack to add the extra binwidth at the end

    t['diff'] = diff.round(3)

    t.write(outfile, format='ascii.fixed_width_no_header', delimiter='', overwrite=True)
