1.1.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- ``peak_method`` as an optional argument to ``KosmosTrace`` [#115]

API Changes
^^^^^^^^^^^

- ``HorneExtract`` no longer requires ``mask`` and ``unit`` arguments [#105]
- ``BoxcarExtract`` and ``HorneExtract`` now accept parameters (and require the image and trace)
  at initialization, and allow overriding any input parameters when calling [#117]

Bug Fixes
^^^^^^^^^

- Corrected the default mask created in ``HorneExtract``/``OptimalExtract``
  when a user doesn't specify one and gives their image as a numpy array [#118]

1.0.0
-----

New Features
^^^^^^^^^^^^

- Added ``Trace`` classes
- Added basic synthetic data routines
- Added ``BoxcarExtract``
- Added ``HorneExtract``, a.k.a. ``OptimalExtract``
- Added basic ``Background`` subtraction

Bug Fixes
^^^^^^^^^

- Update ``codecov-action`` to ``v2``
- Change default branch from ``master`` to ``main``
- Test fixes; bump CI to python 3.8 and 3.9 and deprecate support for 3.7
