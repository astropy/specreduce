from functools import cached_property

import numpy as np
from astropy import units as u
from astropy.modeling.fitting import LMLSQFitter, LinearLSQFitter
from astropy.modeling.models import Linear1D
from astropy.table import QTable, hstack
from gwcs import coordinate_frames as cf
from gwcs import wcs
from specutils import Spectrum1D

__all__ = [
    'WavelengthCalibration1D'
]


def _check_arr_monotonic(arr):
    # returns True if ``arr`` is either strictly increasing or strictly
    # decreasing, otherwise returns False.

    sorted_increasing = np.all(arr[1:] >= arr[:-1])
    sorted_decreasing = np.all(arr[1:] <= arr[:-1])
    return sorted_increasing or sorted_decreasing


class WavelengthCalibration1D():

    def __init__(self, input_spectrum, matched_line_list=None, line_pixels=None,
                 line_wavelengths=None, catalog=None, input_model=Linear1D(), fitter=None):
        """
        input_spectrum: `~specutils.Spectrum1D`
            A one-dimensional Spectrum1D calibration spectrum from an arc lamp or similar.
        matched_line_list: `~astropy.table.QTable`, optional
            An `~astropy.table.QTable` table with (minimally) columns named
            "pixel_center" and "wavelength" with known corresponding line pixel centers
            and wavelengths populated.
        line_pixels: list, array, `~astropy.table.QTable`, optional
            List or array of line pixel locations to anchor the wavelength solution fit.
            Can also be input as an `~astropy.table.QTable` table with (minimally) a column
            named "pixel_center".
        line_wavelengths: `~astropy.units.Quantity`, `~astropy.table.QTable`, optional
            `astropy.units.Quantity` array of line wavelength values corresponding to the
            line pixels defined in ``line_list``, assumed to be in the same
            order. Can also be input as an `~astropy.table.QTable` with (minimally)
            a "wavelength" column.
        catalog: list, str, `~astropy.table.QTable`, optional
            The name of a catalog of line wavelengths to load and use in automated and
            template-matching line matching. NOTE: This option is currently not implemented.
        input_model: `~astropy.modeling.Model`
            The model to fit for the wavelength solution. Defaults to a linear model.
        fitter: `~astropy.modeling.fitting.Fitter`, optional
            The fitter to use in optimizing the model fit. Defaults to
            `~astropy.modeling.fitting.LinearLSQFitter` if the model to fit is linear
            or `~astropy.modeling.fitting.LMLSQFitter` if the model to fit is non-linear.

        Note that either ``matched_line_list`` or ``line_pixels`` must be specified,
        and if ``matched_line_list`` is not input, at least one of ``line_wavelengths``
        or ``catalog`` must be specified.
        """
        self._input_spectrum = input_spectrum
        self._input_model = input_model
        self._cached_properties = ['fitted_model', 'residuals', 'wcs']
        self.fitter = fitter
        self._potential_wavelengths = None
        self._catalog = catalog

        if not isinstance(input_spectrum, Spectrum1D):
            raise ValueError('Input spectrum must be Spectrum1D.')

        # We use either line_pixels or matched_line_list to create self._matched_line_list,
        # and check that various requirements are fulfilled by the input args.
        if matched_line_list is not None:
            pixel_arg = "matched_line_list"
            if not isinstance(matched_line_list, QTable):
                raise ValueError("matched_line_list must be an astropy.table.QTable.")
            self._matched_line_list = matched_line_list
        elif line_pixels is not None:
            pixel_arg = "line_pixels"
            if isinstance(line_pixels, (list, np.ndarray)):
                self._matched_line_list = QTable([line_pixels], names=["pixel_center"])
            elif isinstance(line_pixels, QTable):
                self._matched_line_list = line_pixels
        else:
            raise ValueError("Either matched_line_list or line_pixels must be specified.")

        if "pixel_center" not in self._matched_line_list.columns:
            raise ValueError(f"{pixel_arg} must have a 'pixel_center' column.")

        if self._matched_line_list["pixel_center"].unit is None:
            self._matched_line_list["pixel_center"].unit = u.pix

        # check that pixels are monotonic
        if not _check_arr_monotonic(self._matched_line_list["pixel_center"]):
            raise ValueError('Pixels must be strictly increasing or decreasing.')

        # now that pixels have been determined from input, figure out wavelengths.
        if (line_wavelengths is None and catalog is None
                and "wavelength" not in self._matched_line_list.columns):
            raise ValueError("You must specify at least one of line_wavelengths, "
                             "catalog, or 'wavelength' column in matched_line_list.")

        # Sanity checks on line_wavelengths value
        if line_wavelengths is not None:
            if (isinstance(self._matched_line_list, QTable) and
                    "wavelength" in self._matched_line_list.columns):
                raise ValueError("Cannot specify line_wavelengths separately if there is"
                                 " a 'wavelength' column in matched_line_list.")
            if len(line_wavelengths) != len(self._matched_line_list):
                raise ValueError("If line_wavelengths is specified, it must have the same "
                                 f"length as {pixel_arg}")
            if not isinstance(line_wavelengths, (u.Quantity, QTable)):
                raise ValueError("line_wavelengths must be specified as an astropy.units.Quantity"
                                 " array or as an astropy.table.QTable")

            # make sure wavelengths (or freq) are monotonic and add wavelengths
            # to _matched_line_list
            if isinstance(line_wavelengths, u.Quantity):
                if not _check_arr_monotonic(line_wavelengths):
                    if str(line_wavelengths.unit.physical_type) == "frequency":
                        raise ValueError('Frequencies must be strictly increasing or decreasing.')
                    raise ValueError('Wavelengths must be strictly increasing or decreasing.')

                self._matched_line_list["wavelength"] = line_wavelengths

            elif isinstance(line_wavelengths, QTable):
                if not _check_arr_monotonic(line_wavelengths['wavelength']):
                    raise ValueError('Wavelengths must be strictly increasing or decreasing.')
                self._matched_line_list = hstack([self._matched_line_list, line_wavelengths])

        # Parse desired catalogs of lines for matching.
        if catalog is not None:
            # For now we avoid going into the later logic and just throw an error
            raise NotImplementedError("No catalogs are available yet, please input "
                                      "wavelengths with line_wavelengths or as a "
                                      f"column in {pixel_arg}")

            if isinstance(catalog, QTable):
                if "wavelength" not in catalog.columns:
                    raise ValueError("Catalog table must have a 'wavelength' column.")
                self._catalog = catalog
            else:
                # This will need to be updated to match up with Tim's catalog code
                if isinstance(catalog, list):
                    self._catalog = catalog
                else:
                    self._catalog = [catalog]
                for cat in self._catalog:
                    if isinstance(cat, str):
                        if cat not in self._available_catalogs:
                            raise ValueError(f"Line list '{cat}' is not an available catalog.")

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
    def input_model(self):
        return self._input_model

    @input_model.setter
    def input_model(self, input_model):
        self._clear_cache()
        self._input_model = input_model

    @cached_property
    def fitted_model(self):
        # computes and returns WCS after fitting self.model to self.refined_pixels
        x = self._matched_line_list["pixel_center"]
        y = self._matched_line_list["wavelength"]

        if self.fitter is None:
            # Flexible defaulting if self.fitter is None
            if self.input_model.linear:
                fitter = LinearLSQFitter(calc_uncertainties=True)
            else:
                fitter = LMLSQFitter(calc_uncertainties=True)
        else:
            fitter = self.fitter

        # Fit the model
        return fitter(self.input_model, x, y)

    @cached_property
    def residuals(self):
        """
        calculate fit residuals between matched line list pixel centers and
        wavelengths and the evaluated fit model.
        """

        x = self._matched_line_list["pixel_center"]
        y = self._matched_line_list["wavelength"]

        # Get the fit residuals by evaulating model
        return y - self.fitted_model(x)

    @cached_property
    def wcs(self):
        # Build a GWCS pipeline from the fitted model
        pixel_frame = cf.CoordinateFrame(1, "SPECTRAL", [0,], axes_names=["x",], unit=[u.pix,])
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength",],
                                          unit=[self._matched_line_list["wavelength"].unit,])

        pipeline = [(pixel_frame, self.fitted_model), (spectral_frame, None)]

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
