#!/usr/bin/env python 
# -*- coding: utf8 -*-

from __future__ import division, print_function

import unittest
import numpy

from specreduce.utils import nist

__author__ = 'Bruno Quint'


class NistTest(unittest.TestCase):

    def test_get_data(self):

        ne_wav, ne_int = nist.get_nist_data('Ne', 6570., 6630.)

        test_ne_wav = numpy.array([6598.9528, 6602.9007])
        test_ne_int = numpy.array([10000., 1000.])

        numpy.testing.assert_almost_equal(test_ne_wav, test_ne_wav)
        numpy.testing.assert_almost_equal(ne_int, test_ne_int)


if __name__ == '__main__':
    unittest.main()