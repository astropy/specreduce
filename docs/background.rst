Background correction
=====================

The `specreduce.background` module generates and subtracts a background image from
the input 2D spectral image. The `~specreduce.background.Background` object is
defined by one or more windows, where each window is a region parallel to a
`~specreduce.tracing.Trace`, offset from that `~specreduce.tracing.Trace` by a
specified separation in the cross-dispersion direction, and extending over a
specified width (also measured along the cross-dispersion axis) in pixels. The
object can be generated with:

* `~specreduce.background.Background`
* `Background.one_sided <specreduce.background.Background.one_sided>`
* `Background.two_sided <specreduce.background.Background.two_sided>`

The center of the window can either be passed as a float/integer or as a
`~specreduce.tracing.Trace`.

.. code-block:: python

  bg = specreduce.background.Background.one_sided(image, trace, separation=5, width=2)

or, equivalently

.. code-block:: python

  bg = specreduce.background.Background.one_sided(image, 15, separation=5, width=2)

The background image can be accessed via `~specreduce.background.Background.bkg_image`
and the background-subtracted image via `~specreduce.background.Background.sub_image`
(or ``image - bg``).

The background and trace steps can be done iteratively, to refine an automated
trace using the background-subtracted image as input.