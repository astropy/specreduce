########################
Specreduce Documentation
########################

The `specreduce <https://specreduce.readthedocs.io/en/latest/index.html>`_ package aims to provide a data reduction toolkit for optical
and infrared spectroscopy, on which applications such as pipeline processes for
specific instruments can be built. The scope of its functionality is limited to
basic spectroscopic reduction, with basic *image* processing steps (such as
bias subtraction) instead covered by `ccdproc <https://ccdproc.readthedocs.io/en/latest/>`_ and other packages, data
analysis by `specutils <https://specutils.readthedocs.io/en/latest/>`_ and visualization by `matplotlib <https://matplotlib.org/>`_. A few
examples will nevertheless be provided of its usage in conjunction with these
complementary packages.

.. note::

    Specreduce is currently an incomplete work-in-progress and is liable to
    change. Please feel free to contribute code and suggestions through github.


.. _spectral-extraction:

*******************
Spectral Extraction
*******************

.. toctree::
    :maxdepth: 2

    extraction_quickstart.rst

***********
Calibration
***********

.. toctree::
    :maxdepth: 1

    extinction.rst
    specphot_standards.rst


*****
Index
*****

.. toctree::
    :maxdepth: 1

    api.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


****************
Development Docs
****************

.. toctree::
    :maxdepth: 1

    process/index
