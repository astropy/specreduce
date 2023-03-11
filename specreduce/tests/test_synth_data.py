import pytest

from specreduce.utils.synth_data import make_2dspec_image, make_2d_arc_image
from astropy.nddata import CCDData
from astropy.modeling import models
from astropy.wcs import WCS
import astropy.units as u

def test_make_2dspec_image():
    ccdim = make_2dspec_image(
        nx=3000,
        ny=1000,
        background=5,
        trace_center=None,
        trace_order=3,
        trace_coeffs={'c0': 0, 'c1': 50, 'c2': 100},
        source_amplitude=10,
        source_alpha=0.1
    )
    assert ccdim.data.shape == (1000, 3000)
    assert isinstance(ccdim, CCDData)


@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
def test_make_2d_arc_image_defaults():
    ccdim = make_2d_arc_image()
    assert isinstance(ccdim, CCDData)


@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
def test_make_2d_arc_pass_wcs():
    nx=3000
    ny=1000
    wave_unit = u.Angstrom
    extent = [3000, 6000]

    # test passing a valid WCS with dispersion along X
    wcs = WCS(naxis=2)
    wcs.wcs.ctype[0] = 'WAVE'
    wcs.wcs.ctype[1] = 'PIXEL'
    wcs.wcs.cunit[0] = wave_unit
    wcs.wcs.cunit[1] = u.pixel
    wcs.wcs.crval[0] = extent[0]
    wcs.wcs.cdelt[0] = (extent[1] - extent[0]) / nx
    wcs.wcs.crval[1] = 0
    wcs.wcs.cdelt[1] = 1

    ccdim = make_2d_arc_image(
        nx=nx,
        ny=ny,
        extent=None,
        wave_unit=None,
        wcs=wcs
    )
    assert ccdim.data.shape == (1000, 3000)
    assert isinstance(ccdim, CCDData)

    # make sure WCS without spectral axis gets rejected
    wcs.wcs.ctype[0] = 'PIXEL'
    assert wcs.spectral.naxis == 0
    with pytest.raises(ValueError, match='Provided WCS must have a spectral axis'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=None,
            wave_unit=None,
            wcs=wcs
        )

    # test passing valid WCS with dispersion along Y
    wcs = WCS(naxis=2)
    wcs.wcs.ctype[1] = 'WAVE'
    wcs.wcs.ctype[0] = 'PIXEL'
    wcs.wcs.cunit[1] = wave_unit
    wcs.wcs.cunit[0] = u.pixel
    wcs.wcs.crval[1] = extent[0]
    wcs.wcs.cdelt[1] = (extent[1] - extent[0]) / nx
    wcs.wcs.crval[0] = 0
    wcs.wcs.cdelt[0] = 1

    ccdim = make_2d_arc_image(
        nx=ny,
        ny=nx,
        extent=None,
        wave_unit=None,
        wcs=wcs
    )
    assert ccdim.data.shape == (3000, 1000)
    assert isinstance(ccdim, CCDData)

    # make sure a 1D WCS gets rejected
    wcs = WCS(naxis=1)
    wcs.wcs.ctype[0] = 'WAVE'
    wcs.wcs.cunit[0] = wave_unit
    wcs.wcs.crval[0] = extent[0]
    wcs.wcs.cdelt[0] = (extent[1] - extent[0]) / nx

    with pytest.raises(ValueError, match='WCS must have NAXIS=2 for a 2D image'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=None,
            wave_unit=None,
            wcs=wcs
        )

    # make sure a non-polynomial tilt_func gets rejected
    with pytest.raises(ValueError, match='The only tilt functions currently supported are 1D polynomials'):
        ccdim = make_2d_arc_image(
            tilt_func=models.Gaussian1D
        )