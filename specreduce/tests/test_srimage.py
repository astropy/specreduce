import astropy.units as u
import numpy as np
import pytest

from astropy.nddata import NDData, VarianceUncertainty
from specreduce.image import SRImage

def create_image(imtype: str, nrow: int = 7, ncol: int = 11):
    img = np.tile((np.arange(ncol)), (nrow, 1)).astype('d')
    if imtype == 'ndarray':
        return img
    elif imtype == 'quantity':
        return img * u.DN
    elif imtype == 'nddata':
        return NDData(img*u.DN)
    else:
        raise NotImplementedError('unkown image type')

img = np.tile((np.arange(5, 15)), (7, 1)).astype('d')


@pytest.mark.parametrize("imtype", ["ndarray", "quantity", 'nddata'])
def test_init(imtype):
    img = create_image(imtype)
    image = SRImage(img)
    assert image.shape == (7, 11)
    assert image.unit == u.DN

    image = SRImage(img, disp_axis=1, crossdisp_axis=0)
    assert image.shape == (7, 11)
    assert image.unit == u.DN

    image = SRImage(img, disp_axis=0, crossdisp_axis=1)
    assert image.shape == (11, 7)
    assert image.unit == u.DN

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=1, crossdisp_axis=1)

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=2)

    with pytest.raises(ValueError):
        image = SRImage(img, disp_axis=-1)


def test_init_bad():
    with pytest.raises(ValueError, match='Unrecognized image type.'):
        image = SRImage([[0.0, 1.0],[2.0, 3.0]])