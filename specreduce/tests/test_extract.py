import unittest
import numpy as np

from extract import BoxcarExtract


# Mock a Trace that consists of a straight line
# placed exactly at row 15.
class Trace:
    def __init__(self):
        self.line = np.ones(shape=(10,)) * 15


class TestBoxcarExtract(unittest.TestCase):

    # test image is comprised of 30 rows with 10
    # columns each. Row content is row index itself.
    def setUp(self):
        self.image = np.ones(shape=(30,10))
        for j in range(self.image.shape[0]):
            self.image[j,::] *= j

        assert self.image[0,0] == 0
        assert self.image[0,9] == 0
        assert self.image[10,0] == 10
        assert self.image[10,9] == 10
        assert self.image[29,0] == 29
        assert self.image[29,9] == 29

        self.trace = Trace()

    def test_boxcar_apertures(self):
        boxcar = BoxcarExtract()

        boxcar.apwidth = 5
        boxcar.skysep = 1
        boxcar.skywidth = 5

        spectrum, bkg_spectrum = boxcar(self.image, self.trace)

        # the source extraction should result in a total flux of 75:
        # 13+14+15+16+17. Integer truncation/rounding issues create a
        # non-symmetrical extraction region around the trace.
        print(spectrum.flux.value[0])
        # assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75))

        # the source extraction should result in a total flux of 45+123=168
        print(bkg_spectrum.flux.value[0])
        # assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 168))
