from astropy.modeling.models import Linear1D, Gaussian1D
from astropy.modeling.fitting import LMLSQFitter
import astropy.units as u
from functools import cached_property
from gwcs import wcs
from gwcs import coordinate_frames as cf
import numpy as np
from specutils import SpectralRegion
from specutils.fitting import fit_lines


"""
Starting from Kyle's proposed pseudocode from https://github.com/astropy/specreduce/pull/152
"""

class CalibrationLine():

    def __init__(self, input_spectrum, wavelength, pixel, refinement_method=None,
                 refinement_kwargs={}):
        """
        input_spectrum: `~specutils.Spectrum1D`
            A one-dimensional Spectrum1D calibration spectrum from an arc lamp or similar.
        wavelength: `astropy.units.Quantity`
            The rest wavelength of the line.
        pixel: int
            Initial guess of the integer pixel location of the line center in the spectrum.
        refinement_method: str, optional
            None: use the actual provided pixel values as anchors for the fit without refining
            their location.
            'gradient': Simple gradient descent/ascent to nearest min/max value. Direction
            defined by ``direction`` in refinement_kwargs.
            'min'/'max': find min/max within pixel range provided by ``range`` kwarg.
            'gaussian': fit gaussian to spectrum within pixel range provided by ``range`` kwarg.
        refinement_kwargs: dict, optional
            Keywords to set the parameters for line location refinement. Distance on either side
            of line center to include on either side of gaussian fit is set by int ``range``.
            Gradient descent/ascent is determined by setting ``direction`` to 'min' or 'max'.
        """
        self.input_spectrum = input_spectrum
        self.wavelength = wavelength
        self.pixel = pixel
        self.refinement_method = refinement_method
        self.refinement_kwargs = refinement_kwargs

        if self.refinement_method in ("gaussian", "min", "max"):
            if 'range' not in self.refinement_kwargs:
                raise ValueError(f"You must define 'range' in refinement_kwargs to use "
                                  "{self.refinement_method} refinement.")
        elif self.refinement_method == "gradient" and 'direction' not in self.refinement_kwargs:
            raise ValueError("You must define 'direction' in refinement_kwargs to use "
                             "gradient refinement")

    @classmethod
    def by_name(cls, input_spectrum, line_name, pixel,
                refinement_method=None, refinement_kwargs={}):
	   # this does wavelength lookup and passes that wavelength to __init__ allowing the user
       # to call CalibrationLine.by_name(spectrum, 'H_alpha', pixel=40)
       # We could also have a type-check for ``wavelength`` above, this just feels more verbose
       return cls(...)

    def _do_refinement(self, input_spectrum):
            if self.refinement_method == 'gaussian':
                window_width = self.refinement_kwargs.get('range')
                window = SpectralRegion((self.pixel-window_width)*u.pix,
                                        (self.pixel+window_width)*u.pix)

                # Use specutils.fit_lines to do model fitting. Define window in u.pix based on kwargs
                input_model = Gaussian1D(mean=self.pixel, stddev=3,
                                         amplitude = self.input_spectrum.flux[self.pixel])

                fitted_model = fit_lines(self.input_spectrum, input_model, window=window)
                new_pixel = fitted_model.mean.value

            elif self.refinement_method == 'min':
                window_width = self.refinement_kwargs.get('range')
                new_pixel = np.argmin(self.input_spectrum.flux[self.pixel-window_width:
                                                               self.pixel+window_width+1])
                new_pixel += self.pixel - window_width

            elif self.refinement_method == 'max':
                window_width = self.refinement_kwargs.get('range')
                new_pixel = np.argmax(self.input_spectrum.flux[self.pixel-window_width:
                                                               self.pixel+window_width+1])
                new_pixel += self.pixel - window_width

            return new_pixel

    def refine(self, input_spectrum=None, return_object=False):
        # finds the center of the line according to refinement_method/kwargs and returns
        # either the pixel value or a CalibrationLine object with pixel changed to the refined value
        if self.refinement_method is None:
            return self.pixel
        input_spectrum = self.input_spectrum if input_spectrum is None else input_spectrum
        refined_pixel = self._do_refinement(input_spectrum)
        if return_object:
            return CalibrationLine(input_spectrum, self.wavelength, refined_pixel, ...)

        return refined_pixel

    @cached_property
    def refined_pixel(self):
        # returns the refined pixel for self.input_spectrum
        return self.refine()

    @property
    def with_refined_pixel(self):
        # returns a copy of this object, but with the pixel updated to the refined value
        return self.refine(return_object=True)

    def __str__(self):
        return(f"CalibrationLine: ({self.wavelength}, {self.pixel})")


class WavelengthCalibration1D():

    def __init__(self, input_spectrum, lines, model=Linear1D, spectral_unit=u.Angstrom,
                 fitter=LMLSQFitter(calc_uncertainties=True),
                 default_refinement_method=None, default_refinement_kwargs={}):
        """
        input_spectrum: `~specutils.Spectrum1D`
            A one-dimensional Spectrum1D calibration spectrum from an arc lamp or similar.
        lines: str, list
            List of lines to anchor the wavelength solution fit. List items are coerced to
            CalibrationLine objects if passed as tuples of (pixel, wavelength) with
            default_refinement_method and default_refinement_kwargs as defaults.
        model: `~astropy.modeling.Model`
            The model to fit for the wavelength solution. Defaults to a linear model.
        default_refinement_method: str, optional
            words
        default_refinement_kwargs: dict, optional
            words
        """
        self.input_spectrum = input_spectrum
        self.model = model
        self.spectral_unit = spectral_unit
        self.default_refinement_method = default_refinement_method
        self.default_refinement_kwargs = default_refinement_kwargs

        if self.default_refinement_method in ("gaussian", "min", "max"):
            if 'range' not in self.default_refinement_kwargs:
                raise ValueError(f"You must define 'range' in default_refinement_kwargs to use "
                                  "{self.refinement_method} refinement.")
        elif (self.default_refinement_method == "gradient" and 'direction' not in
                self.default_refinement_kwargs):
            raise ValueError("You must define 'direction' in default_refinement_kwargs to use "
                             "gradient refinement")

        self.lines = []
        if isinstance(lines, str):
            if lines in self.available_line_lists:
                raise ValueError(f"Line list '{lines}' is not an available line list.")
        else:
            for line in lines:
                if isinstance(line, CalibrationLine):
                    self.lines.append(line)
                else:
                    self.lines.append(CalibrationLine(self.input_spectrum, line[1], line[0],
                                      self.default_refinement_method,
                                      self.default_refinement_kwargs))


    @classmethod
    def autoidentify(cls, input_spectrum, line_list, model=Linear1D):
        # line_list could be a string ("common stellar") or an object
        # this builds a list of CalibrationLine objects and passes to __init__
        return cls(...)

    @property
    def refined_lines(self):
        return [l.with_refined_pixel for l in self.lines]

    @property
    def refined_pixels(self):
        # useful for plotting over spectrum to change input lines/refinement options
        return [(l.wavelength, l.refined_pixel) for l in self.lines]

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        x = np.array(self.refined_pixels)[:,0]
        y = np.array(self.refined_pixels)[:,1]

        # Fit the model
        self.model = fitter(self.model, x, y)

        # Build a GWCS pipeline from the fitted model
        pixel_frame = cf.CoordinateFrame(1, "SPECTRAL", [0,], axes_names=["x",], unit=[u.pix,])
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength",], unit=[self.spectral_unit,])

        pipeline = [(pixel_frame, model),(spectral_frame, None)]

        wcsobj = wcs.WCS(pipeline)

        return wcsobj

    def apply_to_spectrum(self, spectrum=None):
       # returns spectrum1d with wavelength calibration applied
       # actual line refinement and WCS solution should already be done so that this can
       # ibe called on multiple science sources
       spectrum = self.input_spectrum if apply_to_spectrum is None else apply_to_spectrum
       spectrum.wcs = self.wcs  # might need a deepcopy!
       return spectrum


class WavelengthCalibration2D():
    # input_spectrum must be 2-dimensional

    # lines are coerced to CalibrationLine objects if passed as tuples with default_refinement_method and default_refinement_kwargs as defaults
    def __init__(input_spectrum, trace, lines, model=Linear1D,
                 default_refinement_method=None, default_refinement_kwargs={}):
        pass

    @classmethod
    def autoidentify(cls, input_spectrum, trace, line_list, model=Linear1D):
        # does this do 2D identification, or just search a 1d spectrum exactly at the trace?
        # or should we require using WavelengthCalibration1D.autoidentify?
        return cls(...)

    @property
    def refined_lines(self):
        return [[(l.wavelength, l.refine(input_spectrum.get_row(row), return_object=False)) for l in self.lines] for row in rows]

    @property
    def refined_pixels(self):
        return [[(l.wavelength, l.refine(input_spectrum.get_row(row), return_object=True)) for l in self.lines] for row in rows]

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        return WCS(...)

    def __call__(self, apply_to_spectrum=None):
       # returns spectrum1d with wavelength calibration applied
       # actual line refinement and WCS solution should already be done so that this can be called on multiple science sources
       apply_to_spectrum = self.input_spectrum if apply_to_spectrum is None else apply_to_spectrum
       apply_to_spectrum.wcs = apply_to_spectrum  # might need a deepcopy!
       return apply_to_spectrum