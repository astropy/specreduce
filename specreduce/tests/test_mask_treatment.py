from copy import deepcopy

import pytest
import numpy as np

from astropy.units import DN
from astropy.nddata import NDData
from specreduce.core import _ImageParser


def mk_image():
    image = np.ones((3, 5))
    mask = np.zeros(image.shape, dtype=bool)
    image[1, 3] = np.nan
    mask[[0, 1], [1, 0]] = 1
    return NDData(image * DN, mask=mask)


def test_bad_option():
    image = mk_image()
    with pytest.raises(ValueError, match="'mask_treatment' must be one of"):
        _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="bad")


def test_bad_mask_type():
    image = mk_image()
    image.mask = image.mask.astype(float)
    with pytest.raises(ValueError, match="'mask' must be a boolean or integer array."):
        _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply")


def test_apply():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(deepcopy(image), disp_axis=1, mask_treatment="apply")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[[0, 1, 1], [1, 0, 3]] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(deepcopy(image), disp_axis=1, mask_treatment="apply")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[1, 3] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_ignore():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="ignore")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="ignore")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_propagate():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="propagate")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[:, [0, 1, 3]] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="propagate")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[:, 3] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_zero_fill():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="zero_fill")
    image_true = np.ones(image.data.shape)
    image_true[[0, 1, 1], [1, 0, 3]] = 0
    np.testing.assert_array_equal(parsed_image.data, image_true)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="zero_fill")
    image_true = np.ones(image.data.shape)
    image_true[1, 3] = 0
    np.testing.assert_array_equal(parsed_image.data, image_true)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_nan_fill():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="nan_fill")
    image_true = np.ones(image.data.shape)
    image_true[[0, 1, 1], [1, 0, 3]] = np.nan
    np.testing.assert_array_equal(parsed_image.data, image_true)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="nan_fill")
    image_true = np.ones(image.data.shape)
    image_true[1, 3] = np.nan
    np.testing.assert_array_equal(parsed_image.data, image_true)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_apply_nan_only():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply_nan_only")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[1, 3] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply_nan_only")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[1, 3] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_apply_mask_only():
    image = mk_image()
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply_mask_only")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    mask_true[[0, 1], [1, 0]] = 1
    np.testing.assert_array_equal(parsed_image.mask, mask_true)

    image.mask = None
    parsed_image = _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply_mask_only")
    np.testing.assert_array_equal(parsed_image.data, image.data)
    mask_true = np.zeros(image.data.shape, dtype=bool)
    np.testing.assert_array_equal(parsed_image.mask, mask_true)


def test_fully_masked():
    image = mk_image()
    image.mask[:] = 1
    with pytest.raises(ValueError, match="Image is fully masked."):
        _ImageParser()._parse_image(image, disp_axis=1, mask_treatment="apply")
