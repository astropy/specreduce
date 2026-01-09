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

- **core.py**: Base class `SpecreduceOperation` and `_ImageParser` for image format coercion. Handles Spectrum1D, CCDData, NDData, Quantity, and ndarray inputs. Defines `MaskingOption` enum and masking strategies.

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

- **wavelength_calibration.py**: Legacy wavelength calibration (deprecated in v1.7.0, removal in v2.0)

- **fluxcal.py**: `FluxCalibration` class for flux calibration with magnitude-to-flux conversion and airmass extinction correction

- **calibration_data.py**: Spectrophotometric standards and line lists

- **utils/synth_data.py**: Synthetic spectroscopic data generation for testing

- **compat.py**: Compatibility layer for specutils v1.x and v2.x

### Key Design Patterns

1. **Image Format Flexibility**: All operations accept multiple input formats via `_ImageParser`
2. **Masking Strategies**: Seven masking options defined in `MaskingOption` enum
3. **Astropy Integration**: Uses Astropy models, units, and conventions throughout
4. **GWCS Support**: Wavelength calibration produces proper WCS objects

## Dependencies

- Python ≥3.11
- Core: numpy≥1.24, astropy≥5.3, scipy≥1.10, specutils≥1.9.1, matplotlib≥3.10, gwcs
- Optional: photutils≥1.0 (stellar profile fitting), synphot (synthetic photometry)

## Code Style

- Line length: 100 characters (flake8 and black)
- Black formatter targeting Python 3.11+
- Ignore E203 (whitespace before ':')
