########################
Specreduce documentation
########################

The ``specreduce`` package aims to provide a data reduction toolkit for optical
and infrared spectroscopy, on which applications such as pipeline processes for
specific instruments can be built. The scope of its functionality is limited to
basic spectroscopic reduction, with basic *image* processing steps (such as
bias subtraction) instead covered by ``ccdproc`` and other packages, data
analysis by ``specutils`` and visualization by ``specviz`` or ``cubeviz``. A few
examples will nevertheless be provided of its usage in conjunction with these
complementary packages.

.. note::

    Specreduce is currently an incomplete work-in-progress and is liable to
    change. Please feel free to contribute code and suggestions through github.


.. _DR-process:

**********************
Data reduction process
**********************

.. toctree::
    :maxdepth: 1

    process/index

***********
Calibration
***********

.. toctree::
    :maxdepth: 1

    extinction.rst
    specphot_standards.rst

*************
Reference/API
*************

.. automodapi:: specreduce.core
    :no-inheritance-diagram:

.. automodapi:: specreduce.tracing
    :no-inheritance-diagram:

.. automodapi:: specreduce.background
    :no-inheritance-diagram:

.. automodapi:: specreduce.extract
    :no-inheritance-diagram:

.. automodapi:: specreduce.calibration_data
    :no-inheritance-diagram:
    :include-all-objects:

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
