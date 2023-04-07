from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LMLSQFitter, LinearLSQFitter
from astropy.table import QTable, hstack
import astropy.units as u
from functools import cached_property
from gwcs import wcs
from gwcs import coordinate_frames as cf
import numpy as np
from specutils import Spectrum1D


__all__ = ['WavelengthCalibration1D']


def get_available_catalogs():
    """
    ToDo: Decide in what format to store calibration line catalogs (e.g., for lamps)
          and write this function to determine the list of available catalog names.
    """
    return []


def concatenate_catalogs():
    """
    ToDo: Code logic to combine the lines from multiple catalogs if needed
    """
    pass


class WavelengthCalibration1D():

    def __init__(self, input_spectrum, line_list, line_wavelengths=None, catalog=None,
                 model=Linear1D(), fitter=None):
        """
        input_spectrum: `~specutils.Spectrum1D`
            A one-dimensional Spectrum1D calibration spectrum from an arc lamp or similar.
        line_list: list, array, `~astropy.table.QTable`
            List or array of line pixel locations to anchor the wavelength solution fit.
            Will be converted to an astropy table internally if a list or array was input.
            Can also be input as an `~astropy.table.QTable` table with (minimally) a column
            named "pixel_center" and optionally a "wavelength" column with known line
            wavelengths populated.
        line_wavelengths: `~astropy.units.Quantity`, `~astropy.table.QTable`, optional
            `astropy.units.Quantity` array of line wavelength values corresponding to the
            line pixels defined in ``line_list``. Does not have to be in the same order (the lists will be sorted)
            but does currently need to be the same length as line_list. Can also be input
            as an `~astropy.table.QTable` with (minimally) a "wavelength" column.
        catalog: list, str, optional
            The name of a catalog of line wavelengths to load and use in automated and
            template-matching line matching.
        model: `~astropy.modeling.Model`
            The model to fit for the wavelength solution. Defaults to a linear model.
        fitter: `~astropy.modeling.fitting.Fitter`, optional
            The fitter to use in optimizing the model fit. Defaults to
            `~astropy.modeling.fitting.LinearLSQFitter` if the model to fit is linear
            or `~astropy.modeling.fitting.LMLSQFitter` if the model to fit is non-linear.
        """
        self._input_spectrum = input_spectrum
        self._model = model
        self._line_list = line_list
        self._cached_properties = ['wcs',]
        self.fitter = fitter
        self._potential_wavelengths = None
        self._catalog = catalog

        # ToDo: Implement having line catalogs
        self._available_catalogs = get_available_catalogs()

        if isinstance(line_list, (list, np.ndarray)):
            self._line_list = QTable([line_list], names=["pixel_center"])

        if self._line_list["pixel_center"].unit is None:
            self._line_list["pixel_center"].unit = u.pix

        # Make sure our pixel locations are sorted
        self._line_list.sort("pixel_center")

        if (line_wavelengths is None and catalog is None
                and "wavelength" not in self._line_list.columns):
            raise ValueError("You must specify at least one of line_wavelengths, "
                             "catalog, or 'wavelength' column in line_list.")

        # Sanity checks on line_wavelengths value
        if line_wavelengths is not None:
            if "wavelength" in line_list:
                raise ValueError("Cannot specify line_wavelengths separately if there is"
                                 "a 'wavelength' column in line_list.")
            if len(line_wavelengths) != len(line_list):
                raise ValueError("If line_wavelengths is specified, it must have the same "
                                 "length as line_pixels")
            if not isinstance(line_wavelengths, (u.Quantity, QTable)):
                raise ValueError("line_wavelengths must be specified as an astropy.units.Quantity"
                                 "array or as an astropy.table.QTable")
            if isinstance(line_wavelengths, u.Quantity):
                line_wavelengths.value.sort()
                self._line_list["wavelength"] = line_wavelengths
            elif isinstance(line_wavelengths, QTable):
                line_wavelengths.sort("wavelength")
                self._line_list = hstack(self._line_list, line_wavelengths)

        # Parse desired catalogs of lines for matching.
        if catalog is not None:
            # For now we avoid going into the later logic and just throw an error
            raise NotImplementedError("No catalogs are available yet, please input "
                                      "wavelengths with line_wavelengths or as a "
                                      "column in line_list")
            if isinstance(catalog, list):
                self._catalog = catalog
            else:
                self._catalog = [catalog]
            for cat in self._catalog:
                if isinstance(cat, str):
                    if cat not in self._available_catalogs:
                        raise ValueError(f"Line list '{cat}' is not an available catalog.")

            # Get the potential lines from any specified catalogs to use in matching
            self._potential_wavelengths = concatenate_catalogs(self._catalog)

    def identify_lines(self):
        """
        ToDo: Code matching algorithm between line pixel locations and potential line
        wavelengths from catalogs.
        """
        pass

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
    def available_catalogs(self):
        return self._available_catalogs

    @property
    def input_spectrum(self):
        return self._input_spectrum

    @input_spectrum.setter
    def input_spectrum(self, new_spectrum):
        # We want to clear the refined locations if a new calibration spectrum is provided
        self._clear_cache()
        self._input_spectrum = new_spectrum

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._clear_cache()
        self._model = new_model

    @cached_property
    def wcs(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        x = self._line_list["pixel_center"]
        y = self._line_list["wavelength"]

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
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength",],
                                          unit=[self._line_list["wavelength"].unit,])

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


