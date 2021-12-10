import unittest
import numpy as np

from ..extract import BoxcarExtract


# Mock a Trace class that represents a line parallel to the image rows.
class Trace:
    def __init__(self, position):
        self.line = np.ones(shape=(10,)) * position


class TestBoxcarExtract(unittest.TestCase):

    # Test image is comprised of 30 rows with 10 columns each. Row content
    # is row index itself. This makes it easy to predict what should be the
    # value extracted from a region centered at any arbitrary Y position.
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

    def test_boxcar_extraction(self):
        #
        # Try combinations of extraction center, and even/odd
        # extraction aperture sizes.
        #
        boxcar = BoxcarExtract()

        boxcar.apwidth = 5

        trace = Trace(15.0)
        spectrum, bkg_spectrum = boxcar(self.image, trace)
        assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 75.))

        trace = Trace(14.5)
        spectrum, bkg_spectrum = boxcar(self.image, trace)
        assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 72.5))

        boxcar.apwidth = 6

        trace = Trace(15.0)
        spectrum, bkg_spectrum = boxcar(self.image, trace)
        assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 90.))

        boxcar.apwidth = 6
        trace = Trace(14.5)
        spectrum, bkg_spectrum = boxcar(self.image, trace)
        assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 87.))

        # TODO
        # Sky extraction should result in a total flux of 45+123=168
        # Here, both positioning and sizing seem to be wrong.
        print(bkg_spectrum.flux.value[0])
        # assert np.allclose(spectrum.flux.value, np.full_like(spectrum.flux.value, 168))
