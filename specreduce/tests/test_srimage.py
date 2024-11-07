import astropy.units as u
import numpy as np
import pytest

from astropy.nddata import NDData, VarianceUncertainty
from specreduce.image import SRImage

img = np.tile((np.arange(5, 15)), (7, 1)).astype('d')

def test_init_ndarray():
    image = SRImage(img)
    image = SRImage(img, disp_axis=1)
    image = SRImage(img, disp_axis=1, crossdisp_axis=0)
    image = SRImage(img, disp_axis=0, crossdisp_axis=1)

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=1, crossdisp_axis=1)

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=2)

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=-1)


def test_init_quantity():
    image = SRImage(img * u.DN, disp_axis=1)


def test_init_nddata():
    image = SRImage(NDData(img * u.DN), disp_axis=1)


def test_init_bad():
    with pytest.raises(ValueError, match='Unrecognized image type.'):
        image = SRImage([[0.0, 1.0],[2.0, 3.0]])