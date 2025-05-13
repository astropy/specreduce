from pathlib import Path

from astropy import units as u
from astropy.io import fits as pf
from astropy.nddata import CCDData, VarianceUncertainty


def read_file(fname, bias):
    d = pf.getdata(fname).astype('d')
    return CCDData(d - bias, unit=u.dn, uncertainty=VarianceUncertainty(d))


def read_data():
    bias = pf.getdata('gtc_osiris_example/osiris_bias.fits.bz2').astype('d')
    obj = read_file('gtc_osiris_example/osiris_tres_3b.fits.bz2', bias)
    lamps = 'HgAr', 'Ne', 'Xe'
    arc_files = sorted(Path('gtc_osiris_example').glob('*arc*'))
    arcs = [read_file(f, bias) for f in arc_files]
    return arcs, lamps, obj
