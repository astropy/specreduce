# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Specreduce is an Astropy coordinated package providing Python utilities for reducing and calibrating spectroscopic data. It offers modular, composable building blocks for spectral reduction workflows.

## Common Commands

### Testing
```bash
# Run all tests via tox
tox -e py314-test

# Run tests with coverage
pytest --pyargs specreduce docs --cov specreduce --cov-config=pyproject.toml

# Run a specific test file
pytest specreduce/tests/test_background.py -v

# Run tests with all optional dependencies
tox -e py314-test-alldeps-cov
```

### Code Style
```bash
# Check code style
tox -e codestyle

# Direct flake8
flake8 specreduce --count --extend-ignore E203
```

### Documentation
```bash
# Build HTML docs
cd docs && make html

# Or with sphinx directly
sphinx-build -b html docs docs/_build/html
```

### Development Setup
```bash
pip install -e ".[test,docs,all]"
```

## Architecture

The reduction workflow follows a modular pipeline: **Trace → Background → Extract → Wavelength Calibration → Flux Calibration**

### Core Modules

- **core.py**: Base class `SpecreduceOperation` and the `parse_image()` function for image format coercion (replaced the former `_ImageParser` class in v1.9). Handles Spectrum (specutils v2), CCDData, NDData, Quantity, and ndarray inputs. `MaskingOption` is a `typing.Literal` of the seven supported masking strategies.

- **tracing.py**: Trace determination classes
  - `Trace`: Base trace (center of image)
  - `FlatTrace`: Constant position trace
  - `ArrayTrace`: Custom trace from array
  - `FitTrace`: Fit trace to image features using peak detection (median, gaussian, centroid methods)

- **background.py**: `Background` class with `two_sided()` and `one_sided()` methods for background estimation. Supports multiple traces and custom aperture definitions.

- **extract.py**: 1D spectrum extraction
  - `BoxcarExtract`: Simple aperture extraction
  - `HorneExtract`/`OptimalExtract`: Optimal extraction with spatial profile fitting and variance propagation

- **wavecal1d.py**: Current wavelength calibration implementation using `WavelengthCalibration1D`. Supports automated line matching, template matching, and produces GWCS-based WCS objects.

- **wavesol1d.py**: `WavelengthSolution1D` — manages the pixel↔wavelength polynomial mapping underlying the calibration. Exports the solution as a lossless GWCS (`gwcs` property) or as a standard FITS WCS fitted with the Paper III grating dispersion function (`wcs()` method, `WAVE-GRI`/`AWAV-GRA`).

- **line_matching.py**: Helpers for matching detected lines to reference line lists.

- **tilt_correction.py**: `TiltCorrection` for correcting 2D spectral tilt/curvature.

- **tilt_solution.py**: `TiltSolution` encapsulating the polynomial tilt transformation.

- **table_utils.py**: Shared helpers for the QTable structures used across modules.

- **wavelength_calibration.py**: Legacy wavelength calibration (deprecated in v1.7.0, removal in v2.0)

- **fluxcal.py**: `FluxCalibration` class for flux calibration with magnitude-to-flux conversion and airmass extinction correction

- **calibration_data.py**: Spectrophotometric standards and line lists

- **utils/synth_data.py**: Synthetic spectroscopic data generation for testing. `SynthImage` is an immutable, chainable builder (`add_background`/`add_source`/`add_arcs`/`add_skylines`/`add_poisson_noise`/`add_read_noise`, then `to_array`/`to_ccddata`/`to_spectrum`). The old `make_2d_trace_image`/`make_2d_arc_image`/`make_2d_spec_image` functions are deprecated shims around it.

### Key Design Patterns

1. **Image Format Flexibility**: All operations accept multiple input formats via `parse_image()`
2. **Masking Strategies**: Seven masking options in the `MaskingOption` Literal (`apply`, `ignore`, `propagate`, `zero_fill`, `nan_fill`, `apply_mask_only`, `apply_nan_only`)
3. **Astropy Integration**: Uses Astropy models, units, and conventions throughout
4. **GWCS Support**: Wavelength calibration produces proper WCS objects

## Dependencies

- Python ≥3.11
- Core: numpy≥1.24, astropy≥6.0, scipy≥1.14, specutils≥2.0, matplotlib≥3.10, gwcs
- Optional: photutils≥1.0 (stellar profile fitting), synphot (synthetic photometry)

## Code Style

- Line length: 100 characters (flake8 and black)
- Black formatter targeting Python 3.11+
- Ignore E203 (whitespace before ':')
