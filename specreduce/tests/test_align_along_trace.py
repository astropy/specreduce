import pytest
import numpy as np
import astropy.units as u

from specreduce.tracing import ArrayTrace
from specreduce.utils import align_spectrum_along_trace


def mk_test_image():
    height, width = 9, 2
    centers = np.array([5.5, 3.0])
    image = np.zeros((height, width))
    image[5, 0] = 1
    image[2:4, 1] = 0.5
    return image, ArrayTrace(image, centers)


def test_align_spectrum_along_trace_bad_input():
    image, trace = mk_test_image()
    with pytest.raises(ValueError, match='Unre'):
        im = align_spectrum_along_trace(image, None)   # noqa

    with pytest.raises(ValueError, match='method must be'):
        im = align_spectrum_along_trace(image, trace, method='int')   # noqa

    with pytest.raises(ValueError, match='Spectral axis length'):
        im = align_spectrum_along_trace(image.T, trace, method='interpolate', disp_axis=0)   # noqa

    with pytest.raises(ValueError, match='Displacement axis must be'):
        im = align_spectrum_along_trace(image, trace, disp_axis=2)  # noqa

    with pytest.raises(ValueError, match='The number of image dimensions must be'):
        im = align_spectrum_along_trace(np.zeros((3, 6, 9)), trace)  # noqa


@pytest.mark.parametrize("method, truth_data, truth_mask, truth_ucty",
                         [('interpolate',
                           np.array([[0, 0, 0, 0.00, 1.0, 0.00, 0, 0, 0],
                                     [0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0]]).T,
                           np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1],
                                     [1, 1, 0, 0, 0, 0, 0, 0, 0]]).astype(bool).T,
                           np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).T),
                          ('shift',
                           np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                     [0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.]]).T,
                           np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(bool).T,
                           np.ones((9, 2)))],
                         ids=('method=interpolate', 'method=shift'))
def test_align_spectrum_along_trace(method, truth_data, truth_mask, truth_ucty):
    image, trace = mk_test_image()
    im = align_spectrum_along_trace(image, trace, method=method)
    assert im.shape == image.shape
    assert im.unit == u.DN
    assert im.uncertainty.uncertainty_type == 'var'
    np.testing.assert_allclose(im.data, truth_data)
    np.testing.assert_allclose(im.uncertainty.array, truth_ucty)
    np.testing.assert_array_equal(im.mask, truth_mask)
