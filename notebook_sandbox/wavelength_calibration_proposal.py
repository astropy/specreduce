class CalibrationLine(fit_to_spectrum, wavelength, pixel,
                      refinement_method=None, refinement_kwargs={}):
    # fit_to_spectrum must be 1D spectrum1d object
    # refinement_method == None: do NOT refine, use actual values as anchors, no kwargs
    # refinement_method == 'uphill'/'downhill', go up/downhill from estimated pixel, no kwargs
    # refinement_method == 'min'/'max', find min/max within pixel range provided by range kwarg
    # refinement_method == 'gaussian', fit gaussian to spectrum within pixel range provided bye range kwarg

    @classmethod
    def by_name(cls, fit_to_spectrum, line_name, pixel,
                refinement_method=None, refinement_kwargs={}):
	   # this does wavelength lookup and passes that wavelength to __init__ allowing the user
       # to call CalibrationLine.by_name(spectrum, 'H_alpha', pixel=40)
       # We could also have a type-check for ``wavelength`` above, this just feels more verbose
       return cls(...)

    def refine(self, fit_to_spectrum=None, return_object=False):
        # finds the center of the line according to refinement_method/kwargs and returns
        # either the pixel value or a CalibrationLine object with pixel changed to the refined value
        fit_to_spectrum = self.fit_to_spectrum if fit_to_spectrum is None else fit_to_spectrum
        refined_pixel = do_refinement(fit_to_spectrum, self.refinement_method, **self.refinement_kwargs)
        if return_object:
            return CalibrationLine(fit_to_spectrum, self.wavelength, refined_pixel, ...)
        return refined_pixel

    @cached_property
    def refined_pixel(self):
        # returns the refined pixel for self.fit_to_spectrum
        return self.refine()

    @property
    def with_refined_pixel(self):
        # returns a copy of this object, but with the pixel updated to the refined value
        return self.refine(return_object=True)


class WavelengthCalibration1D(fit_to_spectrum, lines, model=Linear1D,
                              default_refinement_method=None, default_refinement_kwargs={}):
    # fit_to_spectrum must be 1-dimensional
    # lines are coerced to CalibrationLine objects if passed as tuples with default_refinement_method and default_refinement_kwargs as defaults
    @classmethod
    def autoidentify(cls, fit_to_spectrum, line_list, model=Linear1D, ...):
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
        return WCS(...)

    def __call__(self, apply_to_spectrum=None):
       # returns spectrum1d with wavelength calibration applied
       # actual line refinement and WCS solution should already be done so that this can be called on multiple science sources
       apply_to_spectrum = self.fit_to_spectrum if apply_to_spectrum is None else apply_to_spectrum
       apply_to_spectrum.wcs = apply_to_spectrum  # might need a deepcopy!
       return apply_to_spectrum 


class WavelengthCalibration2D(fit_to_spectrum, trace, lines, model=Linear1D,
                              default_refinement_method=None, default_refinement_kwargs={}):
    # fit_to_spectrum must be 2-dimensional
    # lines are coerced to CalibrationLine objects if passed as tuples with default_refinement_method and default_refinement_kwargs as defaults
    @classmethod
    def autoidentify(cls, fit_to_spectrum, trace, line_list, model=Linear1D, ...):
        # does this do 2D identification, or just search a 1d spectrum exactly at the trace?
        # or should we require using WavelengthCalibration1D.autoidentify?
        return cls(...)

    @property
    def refined_lines(self):
        return [[(l.wavelength, l.refine(fit_to_spectrum.get_row(row), return_object=False)) for l in self.lines] for row in rows]

    @property
    def refined_pixels(self):
        return [[(l.wavelength, l.refine(fit_to_spectrum.get_row(row), return_object=True)) for l in self.lines] for row in rows]

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        return WCS(...)

    def __call__(self, apply_to_spectrum=None):
       # returns spectrum1d with wavelength calibration applied
       # actual line refinement and WCS solution should already be done so that this can be called on multiple science sources
       apply_to_spectrum = self.fit_to_spectrum if apply_to_spectrum is None else apply_to_spectrum
       apply_to_spectrum.wcs = apply_to_spectrum  # might need a deepcopy!
       return apply_to_spectrum 



# various supported syntaxes for creating a calibration object (do we want this much flexibility?):
cal1d = WavelengthCalibration1D(spec1d_lamp, (CalibrationLine(5500, 20), CalibrationLine(6200, 32)))
cal1d = WavelengthCalibration1D(spec1d_lamp, (CalibrationLine.by_name('H_alpha', 20, 'uphill'), CalibrationLine.by_name('Lyman_alpha', 32, 'max', {'range': 10})))
cal1d = WavelengthCalibration1D(spec1d_lamp, ((5500, 20), ('Lyman_alpha', 32)))
cal1d = WavelengthCalibration1D(spec1d_lamp, ((5500, 20), (6000, 30)), default_refinement_method='gaussian')
cal1d = WavelengthCalibration1D.autoidentify(spec1d_lamp, common_stellar_line_list)

# once we have that object, we can access the fitted wavelength solution via the WCS and apply it to 
# a spectrum (either the same or separate from the spectrum used for fitting)
targ_wcs = cal1d.wcs
spec1d_targ1_cal = cal1d(spec1d_targ1)
spec1d_targ2_cal = cal1d(spec1d_targ2)


# we sould either start a 2D calibration the same way (except also passing the trace) or by passing
# the refined lines from the 1D calibration as the starting point
cal2d = WavelengthCalibration2D(spec2d_lamp, trace_lamp, ((5500, 20), (6000, 30)))
cal2d = WavelengthCalibration2D(spec2d_lamp, trace_lamp, cal1d.refined_lines)
spec2d_targ_cal = cal2d(spec2d_targ, trace_targ?)
