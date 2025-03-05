.. _mask_treatment:

Treatment of masked and non-finite pixel values
===============================================

Specreduce provides several options for handling images that contain masked or non-finite pixels.
These options allow users to decide whether to preserve, modify, or remove invalid data.

The mask treatment option is selected using the ``mask_treatment`` argument in class initializers,
methods, and functions where applicable. Note that different classes and procedures within Specreduce
may support only a subset of these options.

Available Options
-----------------

- ``apply``
  The input image is left unmodified. Any pixel that is non-finite (e.g., NaN or infinity) is marked as
  masked, and if an existing mask is present, it is combined with the mask derived from non-finite values.

.. image:: fig_masking_apply.svg

- ``ignore``
  The input image remains unchanged, and any existing mask is discarded. Non-finite values are neither
  masked nor replaced.

.. image:: fig_masking_ignore.svg

- ``propagate``
  The image data remains unchanged. However, if any pixel is masked or non-finite, the entire
  cross-dispersion axis containing that pixel is marked as masked.

.. image:: fig_masking_propagate.svg

- ``zero_fill``
  All pixels that are masked or non-finite are replaced with ``0.0`` and the the mask
  is cleared from the image.

.. image:: fig_masking_zero_fill.svg

- ``nan_fill``
  Similar to zero_fill, but instead of replacing invalid pixels with ``0.0``, they are replaced
  with ``nan``. The mask is then removed from the image.

.. image:: fig_masking_nan_fill.svg

- ``apply_mask_only``
  The image and its mask are returned as provided. Only the existing mask is applied without
  incorporating non-finite value checks.

.. image:: fig_masking_apply_mask_only.svg

- ``apply_nan_only``
  The input image is returned unaltered, but any existing mask is ignored. A new mask is generated
  solely based on non-finite pixels.

.. image:: fig_masking_apply_nan_only.svg