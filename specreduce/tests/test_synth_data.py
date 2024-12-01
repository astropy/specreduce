import pytest
from astropy import units as u
from astropy.modeling import models
from astropy.nddata import CCDData
from astropy.wcs import WCS

from specreduce.utils.synth_data import make_2d_trace_image, make_2d_arc_image, make_2d_spec_image


def test_make_2d_trace_image():
    ccdim = make_2d_trace_image(
        nx=3000,
        ny=1000,
        background=5,
        trace_center=None,
        trace_order=3,
        trace_coeffs={'c0': 0, 'c1': 50, 'c2': 100},
        profile=models.Gaussian1D(amplitude=100, stddev=10)
    )
    assert ccdim.data.shape == (1000, 3000)
    assert isinstance(ccdim, CCDData)


@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
def test_make_2d_arc_image_defaults():
    ccdim = make_2d_arc_image()
    assert isinstance(ccdim, CCDData)


@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
#@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_make_2d_arc_pass_wcs():
    nx = 3000
    ny = 1000
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

    # test passing a tilt model
    tilt_model = models.Chebyshev1D(degree=2, c0=50, c1=0, c2=100)
    ccdim = make_2d_arc_image(
        nx=nx,
        ny=ny,
        extent=None,
        wave_unit=None,
        wcs=wcs,
        tilt_func=tilt_model
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

    # test passing valid WCS with dispersion along Y while using air wavelengths
    wcs = WCS(naxis=2)
    wcs.wcs.ctype[1] = 'AWAV'
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
        wave_air=True,
        wcs=wcs,
        tilt_func=tilt_model
    )
    assert ccdim.data.shape == (3000, 1000)
    assert isinstance(ccdim, CCDData)

    # make sure no WCS and no extent gets rejected
    with pytest.raises(ValueError, match='Must specify either a wavelength extent or a WCS'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=None,
            wave_unit=None,
            wcs=None
        )

    # make sure if extent is provided, it has the right length
    with pytest.raises(ValueError, match='Wavelength extent must be of length 2'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=[1, 2, 3],
            wave_unit=None,
            wcs=None
        )

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

    # make sure a WCS with no spectral axis gets rejected
    wcs = WCS(naxis=2)
    wcs.wcs.ctype[1] = 'PIXEL'
    wcs.wcs.ctype[0] = 'PIXEL'
    wcs.wcs.cunit[1] = u.pixel
    wcs.wcs.cunit[0] = u.pixel
    wcs.wcs.crval[1] = extent[0]
    wcs.wcs.cdelt[1] = (extent[1] - extent[0]) / nx
    wcs.wcs.crval[0] = 0
    wcs.wcs.cdelt[0] = 1

    with pytest.raises(ValueError, match='Provided WCS must have a spectral axis'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=None,
            wave_unit=None,
            wcs=wcs
        )

    # make sure invalid wave_unit is caught
    with pytest.raises(ValueError, match='Wavelength unit must be a length unit'):
        ccdim = make_2d_arc_image(
            nx=nx,
            ny=ny,
            extent=[100, 300],
            wave_unit=u.pixel
        )

    # make sure a non-polynomial tilt_func gets rejected
    with pytest.raises(
        ValueError,
        match='The only tilt functions currently supported are 1D polynomials'
    ):
        ccdim = make_2d_arc_image(
            tilt_func=models.Gaussian1D
        )


@pytest.mark.remote_data
@pytest.mark.filterwarnings("ignore:No observer defined on WCS")
def test_make_2d_spec_image_defaults():
    ccdim = make_2d_spec_image()
    assert isinstance(ccdim, CCDData)
