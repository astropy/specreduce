from specreduce.utils.synth_data import make_2dspec_image
from astropy.nddata import CCDData


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
    assert(ccdim.data.shape == (1000, 3000))
    assert(isinstance(ccdim, CCDData))
