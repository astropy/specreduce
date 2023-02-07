from astropy.modeling.models import Linear1D, Gaussian1D
from astropy.modeling.fitting import LMLSQFitter, LinearLSQFitter
import astropy.units as u
from functools import cached_property
from gwcs import wcs
from gwcs import coordinate_frames as cf
import numpy as np
from specutils import SpectralRegion, Spectrum1D
from specutils.fitting import fit_lines


__all__ = ['CalibrationLine', 'WavelengthCalibration1D']


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
        self._input_spectrum = input_spectrum
        self.wavelength = wavelength
        self.pixel = pixel
        self.refinement_method = refinement_method
        self.refinement_kwargs = refinement_kwargs
        self._cached_properties = ['refined_pixel', 'with_refined_pixel']

        if self.refinement_method in ("gaussian", "min", "max"):
            if 'range' not in self.refinement_kwargs:
                # We may want to adjust this default based on real calibration spectra
                self.refinement_kwargs['range'] = 10

    def _clear_cache(self, *attrs):
        """
        provide convenience function to clearing the cache for cached_properties
        """
        if not len(attrs):
            attrs = self._cached_properties
        for attr in attrs:
            if attr in self.__dict__:
                del self.__dict__[attr]

    @classmethod
    def by_name(cls, input_spectrum, line_name, pixel,
                refinement_method=None, refinement_kwargs={}):
        # this will do wavelength lookup and passes that wavelength to __init__ allowing the user
        # to call CalibrationLine.by_name(spectrum, 'H_alpha', pixel=40)
        # We could also have a type-check for ``wavelength`` above, this just feels more verbose
        return cls(...)

    def _do_refinement(self, input_spectrum):
        if self.refinement_method == 'gaussian':
            window_width = self.refinement_kwargs.get('range')
            window = SpectralRegion((self.pixel-window_width)*u.pix,
                                    (self.pixel+window_width)*u.pix)

            input_model = Gaussian1D(mean=self.pixel, stddev=3,
                                     amplitude=self.input_spectrum.flux[self.pixel])

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

        else:
            raise ValueError(f"Refinement method {self.refinement_method} is not implemented")

        return new_pixel

    def refine(self, input_spectrum=None, return_object=False):
        # finds the center of the line according to refinement_method/kwargs and returns
        # either the pixel value or a CalibrationLine object with pixel changed to the refined value
        if self.refinement_method is None:
            return self.pixel
        input_spectrum = self.input_spectrum if input_spectrum is None else input_spectrum
        refined_pixel = self._do_refinement(input_spectrum)
        if return_object:
            return CalibrationLine(input_spectrum, self.wavelength, refined_pixel,
                                   refinement_method=self.refinement_method,
                                   refinement_kwargs=self.refinement_kwargs)

        return refined_pixel

    @cached_property
    def refined_pixel(self):
        # returns the refined pixel for self.input_spectrum
        return self.refine()

    @cached_property
    def with_refined_pixel(self):
        # returns a copy of this object, but with the pixel updated to the refined value
        return self.refine(return_object=True)

    @property
    def input_spectrum(self):
        return self._input_spectrum

    @input_spectrum.setter
    def input_spectrum(self, new_spectrum):
        # We want to clear the refined locations if a new calibration spectrum is provided
        self._clear_cache()
        self._input_spectrum = new_spectrum

    def __str__(self):
        return f"CalibrationLine: ({self.wavelength}, {self.pixel})"

    def __repr__(self):
        return f"CalibrationLine({self.wavelength}, {self.pixel})"


class WavelengthCalibration1D():

    def __init__(self, input_spectrum, lines, model=Linear1D(), spectral_unit=u.Angstrom,
                 fitter=None, default_refinement_method=None, default_refinement_kwargs={}):
        """
        input_spectrum: `~specutils.Spectrum1D`
            A one-dimensional Spectrum1D calibration spectrum from an arc lamp or similar.
        lines: str, list
            List of lines to anchor the wavelength solution fit. List items are coerced to
            CalibrationLine objects if passed as tuples of (wavelength, pixel) with
            default_refinement_method and default_refinement_kwargs as defaults.
        model: `~astropy.modeling.Model`
            The model to fit for the wavelength solution. Defaults to a linear model.
        default_refinement_method: str, optional
            words
        default_refinement_kwargs: dict, optional
            words
        """
        self._input_spectrum = input_spectrum
        self._model = model
        self.spectral_unit = spectral_unit
        self.default_refinement_method = default_refinement_method
        self.default_refinement_kwargs = default_refinement_kwargs
        self._cached_properties = ['wcs',]
        self.fitter = fitter

        if self.default_refinement_method in ("gaussian", "min", "max"):
            if 'range' not in self.default_refinement_kwargs:
                # We may want to adjust this default based on real calibration spectra
                self.default_refinement_method['range'] = 10

        self._lines = []
        if isinstance(lines, str):
            if lines in self.available_line_lists:
                raise ValueError(f"Line list '{lines}' is not an available line list.")
        else:
            for line in lines:
                if isinstance(line, CalibrationLine):
                    self._lines.append(line)
                else:
                    self._lines.append(CalibrationLine(self.input_spectrum, line[0], line[1],
                                       self.default_refinement_method,
                                       self.default_refinement_kwargs))

    def _clear_cache(self, *attrs):
        """
        provide convenience function to clearing the cache for cached_properties
        """
        if not len(attrs):
            attrs = self._cached_properties
        for attr in attrs:
            if attr in self.__dict__:
                del self.__dict__[attr]

    @property
    def input_spectrum(self):
        return self._input_spectrum

    @input_spectrum.setter
    def input_spectrum(self, new_spectrum):
        # We want to clear the refined locations if a new calibration spectrum is provided
        self._clear_cache()
        for line in self.lines:
            line.input_spectrum = new_spectrum
        self._input_spectrum = new_spectrum

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, new_lines):
        self._clear_cache()
        self._lines = new_lines

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._clear_cache()
        self._model = new_model

    @property
    def refined_lines(self):
        return [line.with_refined_pixel for line in self.lines]

    @property
    def refined_pixels(self):
        # useful for plotting over spectrum to change input lines/refinement options
        return [(line.wavelength, line.refined_pixel) for line in self.lines]

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        x = [line[1] for line in self.refined_pixels] * u.pix
        y = [line[0].value for line in self.refined_pixels] * self.spectral_unit

        if self.fitter is None:
            # Flexible defaulting if self.fitter is None
            if self.model.linear:
                fitter = LinearLSQFitter(calc_uncertainties=True)
            else:
                fitter = LMLSQFitter(calc_uncertainties=True)
        else:
            fitter = self.fitter

        # Fit the model
        self._model = fitter(self._model, x, y)

        # Build a GWCS pipeline from the fitted model
        pixel_frame = cf.CoordinateFrame(1, "SPECTRAL", [0,], axes_names=["x",], unit=[u.pix,])
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength",], unit=[self.spectral_unit,])

        pipeline = [(pixel_frame, self.model), (spectral_frame, None)]

        wcsobj = wcs.WCS(pipeline)

        return wcsobj

    def apply_to_spectrum(self, spectrum=None):
        # returns spectrum1d with wavelength calibration applied
        # actual line refinement and WCS solution should already be done so that this can
        # be called on multiple science sources
        spectrum = self.input_spectrum if spectrum is None else spectrum
        updated_spectrum = Spectrum1D(spectrum.flux, wcs=self.wcs, mask=spectrum.mask,
                                      uncertainty=spectrum.uncertainty)
        return updated_spectrum


'''
# WavelengthCalibration2D is a planned future feature
class WavelengthCalibration2D():
    # input_spectrum must be 2-dimensional

    # lines are coerced to CalibrationLine objects if passed as tuples with
    # default_refinement_method and default_refinement_kwargs as defaults
    def __init__(input_spectrum, trace, lines, model=Linear1D,
                 default_refinement_method=None, default_refinement_kwargs={}):
        return NotImplementedError("2D wavelength calibration is not yet implemented")

    @classmethod
    def autoidentify(cls, input_spectrum, trace, line_list, model=Linear1D):
        # does this do 2D identification, or just search a 1d spectrum exactly at the trace?
        # or should we require using WavelengthCalibration1D.autoidentify?
        return cls(...)

    @property
    def refined_lines(self):
        return [[(l.wavelength, l.refine(input_spectrum.get_row(row), return_object=False))
                 for l in self.lines] for row in rows]

    @property
    def refined_pixels(self):
        return [[(l.wavelength, l.refine(input_spectrum.get_row(row), return_object=True))
                 for l in self.lines] for row in rows]

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        pass

    def __call__(self, apply_to_spectrum=None):
        # returns spectrum1d with wavelength calibration applied
        # actual line refinement and WCS solution should already be done so that this
        # can be called on multiple science sources
        apply_to_spectrum = self.input_spectrum if apply_to_spectrum is None else apply_to_spectrum
        apply_to_spectrum.wcs = apply_to_spectrum  # might need a deepcopy!
        return apply_to_spectrum
'''
