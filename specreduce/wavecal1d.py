import warnings
from functools import cached_property
from typing import Sequence, Callable, Literal

import astropy.units as u
import gwcs
import numpy as np

from astropy.modeling import models, Model, fitting
from astropy.nddata import VarianceUncertainty
from gwcs import coordinate_frames
from matplotlib.pyplot import Axes, Figure, setp, subplots
from numpy import ndarray
from numpy.ma.core import MaskedArray
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

from specreduce.calibration_data import load_pypeit_calibration_lines
from specreduce.compat import Spectrum
from specreduce.line_matching import find_arc_lines

__all__ = ["WavelengthCalibration1D"]


def _format_linelist(lst):
    """Formats a line list for further processing.

    This function ensures that the provided line list adheres to certain shape and
    dimensionality constraints. It converts the input into a standardized format
    while applying necessary modifications such as tiling and masking.

    Parameters
    ----------
    lst
        Input array of centroids or centroids with amplitudes. Must be either:
            - A 1D array with a shape [n] for centroids.
            - A 2D array with a shape [n, 2] for centroids and amplitudes.

    Returns
    -------
    numpy.ma.MaskedArray
        Formatted and standardized line list array with shape [n, 2], where each row
        contains a line centroid and amplitude.

    Raises
    ------
    ValueError
        If the input line list does not meet the specified dimensional or shape
        requirements.
    """
    lst = MaskedArray(lst, copy=True)

    if (lst.ndim > 2) or (lst.ndim == 2 and lst.shape[1] > 2):
        raise ValueError(
            "Line lists must be 1D with a shape [n] (centroids) or "
            "2D with a shape [n, 2] (centroids and amplitudes)."
        )

    if lst.mask is np.False_:
        lst.mask = np.zeros(lst.shape[0], dtype=bool)

    if lst.ndim == 1:
        lst = np.tile(lst[:, None], [1, 2])
        lst[:, 1] = 0.0
        lst.mask[:, :] = lst.mask.any(axis=1)[:, None]

    sids = np.argsort(lst.data[:, 0])
    return lst[sids]


def _unclutter_text_boxes(labels: Sequence) -> None:
    """Remove overlapping labels from the plot.

    Removes overlapping text labels from a set of matplotlib label objects. The function iterates
    over all combinations of labels, checks for overlaps among their bounding boxes, and removes
    the label with the lower z-order in case of an overlap.

    Parameters
    ----------
    labels
        A list of matplotlib.text.Text objects.
    """
    to_remove = set()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1 = labels[i]
            l2 = labels[j]
            bbox1 = l1.get_window_extent()
            bbox2 = l2.get_window_extent()
            if bbox1.overlaps(bbox2):
                if l1.zorder < l2.zorder:
                    to_remove.add(l1)
                else:
                    to_remove.add(l2)

    for label in to_remove:
        label.remove()


def _diff_poly1d(m: models.Polynomial1D) -> models.Polynomial1D:
    """Compute the derivative of a Polynomial1D model.

    Computes the derivative of a Polynomial1D model and returns a new Polynomial1D
    model representing the derivative. The coefficients of the input model are
    used to calculate the coefficients of the derivative model. For a Polynomial1D
    of degree n, the derivative is a Polynomial1D of degree n-1.

    Parameters
    ----------
    m
        A Polynomial1D model for which the derivative is to be computed.

    Returns
    -------
    A new Polynomial1D model representing the derivative of the input Polynomial1D model.
    """
    coeffs = {f"c{i-1}": i * getattr(m, f"c{i}").value for i in range(1, m.degree + 1)}
    return models.Polynomial1D(m.degree - 1, **coeffs)


class WavelengthCalibration1D:
    def __init__(
        self,
        ref_pixel: float,
        unit: u.Unit = u.angstrom,
        degree: int = 3,
        line_lists: Sequence | None = None,
        arc_spectra: Spectrum | Sequence[Spectrum] | None = None,
        obs_lines: ndarray | Sequence[ndarray] | None = None,
        pix_bounds: tuple[int, int] | None = None,
        line_list_bounds: tuple[float, float] = (0, np.inf),
        wave_air: bool = False,
    ) -> None:
        """A class for wavelength calibration of one-dimensional spectral data.

        This class is designed to facilitate wavelength calibration of one-dimensional spectra,
        with support for both direct input of line lists and observed spectra. It uses a polynomial
        model for fitting the wavelength solution and offers features to incorporate catalog lines
        and observed line positions.

        Parameters
        ----------
        ref_pixel
            The reference pixel in which the wavelength solution will be centered.

        unit
            The unit of the wavelength calibration, by default ``astropy.units.Angstrom``.

        degree
            The polynomial degree for the wavelength solution, by default 3.

        line_lists
            Catalogs of spectral line wavelengths for wavelength calibration. Provide either an
            array of line wavelengths or a list of catalog names. If `None`, no line lists are used.

        arc_spectra
            Arc spectra provided as ``Spectrum`` objects for wavelength fitting, by default
            None. This parameter and ``obs_lines`` cannot be provided simultaneously.

        obs_lines
            Pixel positions of observed spectral lines for wavelength fitting, by default None. This
            parameter and ``arc_spectra`` cannot be provided simultaneously.

        pix_bounds
            Lower and upper pixel bounds for fitting, defined as a 2-tuple (min, max). If
            ``obs_lines`` is provided, this parameter is mandatory.

        line_list_bounds
            Wavelength bounds (inclusive) as a range (min, max) for filtering usable spectral
            lines from the provided line lists, by default (0, np.inf).

        wave_air
            Boolean indicating whether the input wavelengths correspond to air rather than vacuum;
            by default `False`, meaning vacuum wavelengths.
        """
        self.unit = unit
        self._unit_str = unit.to_string("latex")
        self.degree = degree
        self.ref_pixel = ref_pixel
        self.nframes = 0

        if degree < 1:
            raise ValueError("Degree must be at least 1.")
        if ref_pixel < 0:
            raise ValueError("Reference pixel must be positive.")

        self.arc_spectra: list[Spectrum] | None = None
        self.bounds_pix: tuple[int, int] | None = pix_bounds
        self.bounds_wav: tuple[float, float] | None = None
        self._cat_lines: list[MaskedArray] | None = None
        self._obs_lines: list[MaskedArray] | None = None
        self._trees: Sequence[KDTree] | None = None

        self._fit: optimize.OptimizeResult | None = None
        self._wcs: gwcs.wcs.WCS | None = None

        self._p2w: Model | None = None  # pixel -> wavelength model
        self._w2p: Callable | None = None  # wavelength -> pixel model
        self._p2w_dldx: Model | None = None  # delta lambda / delta pixel

        # Read and store the observational data if given. The user can provide either a list of arc
        # spectra as Spectrum1D objects or a list of line pixel position arrays. An attempt to give
        # both raises an error.
        if arc_spectra is not None and obs_lines is not None:
            raise ValueError("Only one of arc_spectra or obs_lines can be provided.")

        if arc_spectra is not None:
            self.arc_spectra = [arc_spectra] if isinstance(arc_spectra, Spectrum) else arc_spectra
            self.nframes = len(self.arc_spectra)
            for s in self.arc_spectra:
                if s.data.ndim > 1:
                    raise ValueError("The arc spectrum must be one dimensional.")
            self.bounds_pix = (0, self.arc_spectra[0].shape[0])

        elif obs_lines is not None:
            self.observed_lines = obs_lines
            self.nframes = len(self._obs_lines)
            if self.bounds_pix is None:
                raise ValueError("Must give pixel bounds when providing observed line positions.")

        # Read the line lists if given. The user can provide an array of line wavelength positions
        # or a list of line list names (used by `load_pypeit_calibration_lines`) for each arc
        # spectrum.
        if line_lists is not None:
            if not isinstance(line_lists, (tuple, list)):
                line_lists = [line_lists]

            if len(line_lists) != self.nframes:
                raise ValueError("The number of line lists must match the number of arc spectra.")
            self._read_linelists(line_lists, line_list_bounds=line_list_bounds, wave_air=wave_air)

    def _init_model(self):
        self._p2w = models.Shift(-self.ref_pixel) | models.Polynomial1D(self.degree)

    def _read_linelists(
        self,
        line_lists: Sequence,
        line_list_bounds: tuple[float, float] = (0.0, np.inf),
        wave_air: bool = False,
    ):
        """Read and processes line lists and organize them for further use.

        Parameters
        ----------
        line_lists
            A collection of line lists that can either be arrays of wavelengths or ``pypeit``
            lamp names.

        line_list_bounds
            A tuple specifying the minimum and maximum wavelength bounds. Only wavelengths
            within this range are retained.

        wave_air
             If True, convert the vacuum wavelengths used by ``pypeit`` to air wavelengths.
        """

        lines_wav = []
        for lst in line_lists:
            if isinstance(lst, ndarray):
                lines_wav.append(lst)
            else:
                lines = []
                if isinstance(lst, str):
                    lst = [lst]
                for ll in lst:
                    lines.append(
                        load_pypeit_calibration_lines(ll, wave_air=wave_air)["wavelength"]
                        .to(self.unit)
                        .value
                    )
                lines_wav.append(np.ma.masked_array(np.sort(np.concatenate(lines))))
        for i, lst in enumerate(lines_wav):
            lines_wav[i] = lst[(lst >= line_list_bounds[0]) & (lst <= line_list_bounds[1])]

        self.catalog_lines = lines_wav
        self._create_trees()

    def _create_trees(self) -> None:
        """Initialize the KDTree instances for the current set of catalog line locations."""
        self._trees = [KDTree(lst.compressed()[:, None]) for lst in self.catalog_line_locations]

    def find_lines(self, fwhm: float, noise_factor: float = 1.0) -> None:
        """Find lines in the provided arc spectra.

        Determines the spectral lines within each spectrum of the arc spectra based on the
        provided initial guess for the line Full Width at Half Maximum (FWHM).

        Parameters
        ----------
        fwhm
            Initial guess for the FWHM for the spectral lines, used as a parameter in
            the ``find_arc_lines`` function to locate and identify spectral arc lines.

        noise_factor
            The factor to multiply the uncertainty by to determine the noise threshold
            in the `~specutils.fitting.find_lines_threshold` routine.
        """
        if self.arc_spectra is None:
            raise ValueError("Must provide arc spectra to find lines.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            line_lists = []
            for i, arc in enumerate(self.arc_spectra):
                lines = find_arc_lines(arc, fwhm, noise_factor=noise_factor)

                # Find the line amplitudes. We could also use the fitted amplitudes
                # returned by `find_arc_lines`, but these can be unreliable, even when
                # the centroids would be fine. The amplitudes are used only for plotting,
                # so selecting the maximum flux near the line centroid is the best approach.
                amplitudes = np.zeros(len(lines))
                for j, line in enumerate(lines["centroid"].value):
                    line = int(np.floor(line))
                    if line < 0 or line >= arc.shape[0]:
                        raise ValueError(
                            "Error in arc line identification. Try increasing ``noise_factor``."
                        )
                    amplitudes[j] = arc.flux.value[line - 2 : line + 2].max()
                line_lists.append(
                    np.ma.masked_array(np.transpose([lines["centroid"].value, amplitudes]))
                )
            self.observed_lines = line_lists

    def fit_lines(
        self,
        pixels: Sequence,
        wavelengths: Sequence,
        match_obs: bool = False,
        match_cat: bool = False,
        refine_fit: bool = True,
        refine_max_distance: float = 5.0,
    ) -> None:
        """Fit the wavelength solution model using provided line pairs.

        Fits the pixel-to-wavelength transformation model using explicitly
        provided pairs of pixel coordinates and their corresponding wavelengths.
        This method uses linear least squares fitting.

        Optionally, the provided pixel and wavelength values can be "snapped"
        to the nearest values present in the internally stored observed line
        list and catalog line list, respectively. This can correct for small
        inaccuracies in the input pairs if the internal lists are populated.

        Parameters
        ----------
        pixels
            A sequence of pixel positions corresponding to known spectral lines.

        wavelengths
            A sequence of the same size as ``pixels``, containing the known
            wavelengths corresponding to the given pixel positions.

        match_obs
            If True, snap the input ``pixels`` values to the nearest
            pixel values found in ``self.observed_line_locations`` (if available). This helps
            ensure the fit uses the precise centroids detected by `find_lines`
            or provided initially.

        match_cat
            If True, snap the input ``wavelengths`` values to the
            nearest wavelength values found in ``self.catalog_line_locations`` (if available).
            This ensures the fit uses the precise catalog wavelengths.

        refine_fit
            If True (default), automatically call the ``refine_fit`` method
            immediately after the global optimization to improve the solution
            using a least-squares fit on matched lines.

        refine_max_distance
            The maximum allowed distance between the catalog and observed lines for them to be
            considered a match.
        """
        pixels = np.asarray(pixels)
        wavelengths = np.asarray(wavelengths)
        if pixels.size != wavelengths.size:
            raise ValueError("The sizes of pixel and wavelength arrays must match.")
        nlines = pixels.size
        if nlines < 2:
            raise ValueError("Need at least two lines for a fit")

        if self.bounds_pix is None:
            raise ValueError("Cannot fit without pixel bounds set.")

        if self._p2w is None:
            self._init_model()

        # Match the input wavelengths to catalog lines.
        if match_cat:
            if self._cat_lines is None:
                raise ValueError("Cannot fit without catalog lines set.")
            tree = KDTree(
                np.concatenate([c.compressed() for c in self.catalog_line_locations])[:, None]
            )
            ix = tree.query(wavelengths[:, None])[1]
            wavelengths = tree.data[ix][:, 0]

        # Match the input pixel values to observed pixel values.
        if match_obs:
            if self._obs_lines is None:
                raise ValueError("Cannot fit without observed lines set.")
            tree = KDTree(
                np.concatenate([c.compressed() for c in self.observed_line_locations])[:, None]
            )
            ix = tree.query(pixels[:, None])[1]
            pixels = tree.data[ix][:, 0]

        fitter = fitting.LinearLSQFitter()
        m = self._p2w[1]
        if m.degree > nlines:
            for i in range(nlines, m.degree + 1):
                m.fixed[f"c{i}"] = True
        m = fitter(m, pixels - self.ref_pixel, wavelengths)
        for i in range(m.degree + 1):
            m.fixed[f"c{i}"] = False
        self._p2w = self._p2w[0] | m

        can_match = self._cat_lines is not None and self._obs_lines is not None
        if refine_fit and can_match:
            self.refine_fit(refine_max_distance)
        else:
            self._calculate_p2w_derivative()
            self._calculate_p2w_inverse()
            if can_match:
                self.match_lines()

    def fit_global(
        self,
        wavelength_bounds: tuple[float, float],
        dispersion_bounds: tuple[float, float],
        higher_order_limits: Sequence[float] | None = None,
        popsize: int = 30,
        max_distance: float = 100,
        refine_fit: bool = True,
    ) -> None:
        """Calculate a wavelength solution using global optimization.

        Determines an initial functional relationship between pixel positions
        and wavelengths using a global optimization algorithm (differential
        evolution). This method does not require pre-matched pixel-wavelength
        pairs. It works by finding model parameters that minimize the
        distance between the predicted wavelengths of observed lines and their nearest
        catalog counterparts accessed via KDTree).

        Optionally, this initial solution can be immediately refined using a
        least-squares fit on automatically matched lines.

        Parameters
        ----------
        wavelength_bounds
            Bounds (min, max) for the wavelength value at the ``ref_pixel``.
            Used as a constraint in the optimization.

        dispersion_bounds
            Bounds (min, max) for the dispersion (d_wavelength / d_pixel)
            at the ``ref_pixel``. Used as a constraint in the optimization.

        higher_order_limits
            Limit for the absolute value of the higher-order coefficients.
            The optimization bounds for each coefficient will be set to
            [-limit, limit].

        popsize
            Population size for the ``scipy.optimize.differential_evolution``
            optimizer. Larger values increase the likelihood of finding the
            global minimum but also increase computation time.

        max_distance
            Maximum distance (in wavelength units) allowed when associating
            an observed line with a theoretical line during the optimization's
            cost function evaluation. Points beyond this distance contribute
            this maximum value to the cost, preventing outliers from having
            excessive influence.

        refine_fit
            If True (default), automatically call the ``refine_fit`` method
            immediately after the global optimization to improve the solution
            using a least squares fit on matched lines.
        """

        if self._p2w is None:
            self._init_model()
        model = self._p2w

        obs_lines = self.observed_line_locations

        def minfun(x):
            total_distance = 0.0
            for t, l in zip(self._trees, obs_lines):
                transformed_lines = model.evaluate(l, -self.ref_pixel, *x)[:, None]
                total_distance += np.clip(t.query(transformed_lines)[0], 0, max_distance).sum()
            return total_distance

        # Define bounds for differential_evolution.
        bounds = [np.asarray(wavelength_bounds), np.asarray(dispersion_bounds)]

        if higher_order_limits is not None:
            if len(higher_order_limits) != model[1].degree - 1:
                raise ValueError(
                    "The number of higher-order limits must match the degree of the polynomial "
                    "model minus one."
                )
            for v in higher_order_limits:
                bounds.append(np.asarray([v, v]))
        else:
            for i in range(2, model[1].degree + 1):
                bounds.append(np.array([-1, 1]) * 10**(np.log10(np.mean(dispersion_bounds)) - 2))
        bounds = np.array(bounds)

        self._fit = fit = optimize.differential_evolution(
            minfun,
            bounds=bounds,
            popsize=popsize,
        )
        self._p2w = models.Shift(-self.ref_pixel) | models.Polynomial1D(
            self.degree, **{f"c{i}": fit.x[i] for i in range(fit.x.size)}
        )

        # Update the model with the best-fit parameters found
        if refine_fit:
            self.refine_fit()
        else:
            self._calculate_p2w_derivative()
            self._calculate_p2w_inverse()
            self.match_lines()

    def refine_fit(self, max_match_distance: float = 5.0, max_iter: int = 5) -> None:
        """Refine the fit of the pixel-to-wavelength transformation.

        Refines the fit of a polynomial model to data by performing a fitting operation
        using matched pixel and wavelength data points. The method uses a linear least
        squares fitter to optimize the model parameters based on the match.

        Parameters
        ----------

        max_match_distance
            Maximum allowable distance used to identify matched pixel and wavelength
            data points. Points exceeding the bound will not be considered in the fit.

        max_iter
            Maximum number of fitting iterations.
        """

        model = self._p2w[1]
        fitter = fitting.LinearLSQFitter()
        rms = np.nan
        for i in range(max_iter):
            self.match_lines(max_match_distance)
            matched_pix = np.ma.concatenate(self.observed_line_locations).compressed()
            matched_wav = np.ma.concatenate(self.catalog_line_locations).compressed()
            rms_new = np.sqrt(((matched_wav - self.pix_to_wav(matched_pix)) ** 2).mean())
            if rms_new == rms:
                break
            else:
                self._p2w = self._p2w[0].copy() | fitter(
                    model, matched_pix - self.ref_pixel, matched_wav
                )
                rms = rms_new
        self._calculate_p2w_derivative()
        self._calculate_p2w_inverse()
        self.match_lines(max_match_distance)

    def _calculate_p2w_derivative(self) -> None:
        """Calculate (d wavelength) / (d pixel) for the pixel-to-wavelength transformation."""
        if self._p2w is not None:
            self._p2w_dldx = models.Shift(self._p2w.offset_0) | _diff_poly1d(self._p2w[1])

    def _calculate_p2w_inverse(self) -> None:
        """Compute the wavelength-to-pixel mapping from the pixel-to-wavelength transformation."""
        p = np.arange(self.bounds_pix[0] - 2, self.bounds_pix[1] + 2)
        self._w2p = interp1d(self._p2w(p), p, bounds_error=False, fill_value=np.nan)
        self.bounds_wav = self._p2w(self.bounds_pix)

    def resample(
        self,
        spectrum: Spectrum,
        nbins: int | None = None,
        wlbounds: tuple[float, float] | None = None,
        bin_edges: Sequence[float] | None = None,
    ) -> Spectrum:
        """Bin the given pixel-space 1D spectrum to a wavelength space conserving the flux.

        This method bins a pixel-space spectrum to a wavelength space using the computed
        pixel-to-wavelength and wavelength-to-pixel transformations and their derivatives with
        respect to the spectral axis. The binning is exact and conserves the total flux.

        Parameters
        ----------
        spectrum
            A Spectrum1D instance containing the flux to be resampled over the wavelength space.

        nbins
            The number of bins for resampling. If not provided, it defaults to the size of the
            input spectrum.

        wlbounds
            A tuple specifying the starting and ending wavelengths for resampling. If not
            provided, the wavelength bounds are inferred from the object's methods and the
            entire flux array is used.

        bin_edges
            Explicit bin edges in the wavelength space. If provided, ``nbins`` and ``wlbounds``
            are ignored.

        Returns
        -------
            1D spectrum binned to the specified wavelength bins.
        """
        if self._p2w is None:
            raise ValueError("Wavelength solution not yet computed.")

        if nbins is not None and nbins < 0:
            raise ValueError("Number of bins must be positive.")

        flux = spectrum.flux.value
        if spectrum.uncertainty is not None:
            ucty = spectrum.uncertainty.represent_as(VarianceUncertainty).array
            ucty_type = type(spectrum.uncertainty)
        else:
            ucty = np.zeros_like(flux)
            ucty_type = VarianceUncertainty
        npix = flux.size
        nbins = npix if nbins is None else nbins
        if wlbounds is None:
            l1 = self._p2w(0) - self._p2w_dldx(0)
            l2 = self._p2w(npix) + self._p2w_dldx(npix)
        else:
            l1, l2 = wlbounds

        bin_edges_wav = bin_edges if bin_edges is not None else np.linspace(l1, l2, num=nbins + 1)
        bin_edges_pix = np.clip(self._w2p(bin_edges_wav) + 0.5, 0, npix - 1e-12)
        bin_edge_ix = np.floor(bin_edges_pix).astype(int)
        bin_edge_w = bin_edges_pix - bin_edge_ix
        bin_centers_wav = 0.5 * (bin_edges_wav[:-1] + bin_edges_wav[1:])
        flux_wl = np.zeros(nbins)
        ucty_wl = np.zeros(nbins)
        weights = np.zeros(npix)

        dldx = self._p2w_dldx(np.arange(npix))
        n = np.nansum(flux) / np.nansum((dldx * flux))
        for i in range(nbins):
            i1, i2 = bin_edge_ix[i : i + 2]
            weights[:] = 0
            if i1 != i2:
                weights[i1 + 1 : i2] = 1.0
                weights[i1] = 1 - bin_edge_w[i]
                weights[i2] = bin_edge_w[i + 1]
                flux_wl[i] = (weights[i1 : i2 + 1] * flux[i1 : i2 + 1] * dldx[i1 : i2 + 1]).sum()
                ucty_wl[i] = (weights[i1 : i2 + 1] * ucty[i1 : i2 + 1] * dldx[i1 : i2 + 1]).sum()
            else:
                flux_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * flux[i1] * dldx[i1]
                ucty_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * ucty[i1] * dldx[i1]
        flux_wl = (flux_wl * n) * spectrum.flux.unit
        ucty_wl = VarianceUncertainty(ucty_wl * n).represent_as(ucty_type)
        return Spectrum(flux_wl, bin_centers_wav * u.angstrom, uncertainty=ucty_wl)

    def pix_to_wav(self, pix: MaskedArray | ndarray | float) -> ndarray | float:
        """Map pixel values into wavelength values.

        Parameters
        ----------
        pix
            Input pixel value(s) to be transformed into wavelength.

        Returns
        -------
        Transformed wavelength value(s) corresponding to the input pixel value(s).
        """
        if isinstance(pix, MaskedArray):
            wav = self._p2w(pix.data)
            return np.ma.masked_array(wav, mask=pix.mask)
        else:
            return self._p2w(pix)

    def wav_to_pix(self, wav: MaskedArray | ndarray | float) -> ndarray | float:
        """Map wavelength values into pixel values.

        Parameters
        ----------
        wav
            The wavelength value(s) to be converted into pixel value(s).

        Returns
        -------
        The corresponding pixel value(s) for the input wavelength(s).
        """
        if isinstance(wav, MaskedArray):
            pix = self._w2p(wav.data)
            return np.ma.masked_array(pix, mask=wav.mask)
        else:
            return self._w2p(wav)

    @property
    def observed_lines(self) -> None | list[MaskedArray]:
        """Pixel positions and amplitudes of the observed lines as a list of masked arrays."""
        return self._obs_lines

    @cached_property
    def observed_line_locations(self) -> None | list[MaskedArray]:
        """Pixel positions of the observed lines as a list of masked arrays."""
        if self._obs_lines is None:
            return None
        else:
            return [line[:, 0] for line in self._obs_lines]

    @cached_property
    def observed_line_amplitudes(self) -> None | list[MaskedArray]:
        """Amplitudes of the observed lines as a list of masked arrays."""
        if self._obs_lines is None:
            return None
        else:
            return [line[:, 1] for line in self._obs_lines]

    @observed_lines.setter
    def observed_lines(self, line_lists: MaskedArray | ndarray | list[MaskedArray] | list[ndarray]):
        if not isinstance(line_lists, Sequence):
            line_lists = [line_lists]
        self._obs_lines = []
        for lst in line_lists:
            self._obs_lines.append(_format_linelist(lst))
        if hasattr(self, "observed_line_locations"):
            del self.observed_line_locations
        if hasattr(self, "observed_line_amplitudes"):
            del self.observed_line_amplitudes

    @property
    def catalog_lines(self) -> None | list[MaskedArray]:
        """Catalog line wavelengths as a list of masked arrays."""
        return self._cat_lines

    @cached_property
    def catalog_line_locations(self) -> None | list[MaskedArray]:
        """Pixel positions of the catalog lines as a list of masked arrays."""
        if self._cat_lines is None:
            return None
        else:
            return [line[:, 0] for line in self._cat_lines]

    @cached_property
    def catalog_line_amplitudes(self) -> None | list[MaskedArray]:
        """Amplitudes of the catalog lines as a list of masked arrays."""
        if self._obs_lines is None:
            return None
        else:
            return [line[:, 1] for line in self._cat_lines]

    @catalog_lines.setter
    def catalog_lines(self, line_lists: MaskedArray | ndarray | list[MaskedArray] | list[ndarray]):
        if not isinstance(line_lists, Sequence):
            line_lists = [line_lists]
        self._cat_lines = []
        for lst in line_lists:
            self._cat_lines.append(_format_linelist(lst))
        if hasattr(self, "catalog_line_locations"):
            del self.catalog_line_locations
        if hasattr(self, "catalog_line_amplitudes"):
            del self.catalog_line_amplitudes

    @property
    def gwcs(self) -> gwcs.wcs.WCS:
        """GWCS object defining the mapping between pixel and spectral coordinate frames."""
        pixel_frame = coordinate_frames.CoordinateFrame(
            1,
            "SPECTRAL",
            (0,),
            axes_names=[
                "x",
            ],
            unit=[u.pix],
        )
        spectral_frame = coordinate_frames.SpectralFrame(
            axes_names=("wavelength",),
            unit=[self.unit],
        )
        pipeline = [(pixel_frame, self._p2w), (spectral_frame, None)]
        self._wcs = gwcs.wcs.WCS(pipeline)
        return self._wcs

    def match_lines(self, max_distance: float = 5) -> None:
        """Match the observed lines to theoretical lines.

        Parameters
        ----------
        max_distance
            The maximum allowed distance between the catalog and observed lines for them to be
            considered a match.
        """

        for iframe, tree in enumerate(self._trees):
            l, ix = tree.query(
                self._p2w(self.observed_line_locations[iframe].data)[:, None],
                distance_upper_bound=max_distance,
            )
            m = np.isfinite(l)

            # Check for observed lines that match a catalog line.
            # Remove all but the nearest match. This isn't an optimal solution,
            # we could also iterate the match by removing the currently matched
            # lines, but this works for now.
            uix, cnt = np.unique(ix[m], return_counts=True)
            if any(n := cnt > 1):
                for i, c in zip(uix[n], cnt[n]):
                    s = ix == i
                    r = np.zeros(c, dtype=bool)
                    r[np.argmin(l[s])] = True
                    m[s] = r

            self._cat_lines[iframe].mask[:, :] = True
            self._cat_lines[iframe].mask[ix[m], :] = False
            self._obs_lines[iframe].mask[:, :] = ~m[:, None]

    def remove_unmatched_lines(self) -> None:
        """Remove unmatched lines from observation and catalog line data."""
        self.observed_lines = [lst.compressed().reshape([-1, 2]) for lst in self._obs_lines]
        self.catalog_lines = [lst.compressed().reshape([-1, 2]) for lst in self._cat_lines]
        self._create_trees()

    def rms(self, space: Literal["pixel", "wavelength"] = "wavelength") -> float:
        """Compute the RMS of the residuals between matched lines in the pixel or wavelength space.

        Parameters
        ----------
        space
            The space in which to calculate the RMS residual. If 'wavelength',
            the calculation is performed in the wavelength space. If 'pixel',
            it is performed in the pixel space. Default is 'wavelength'.

        Returns
        -------
        float
        """
        self.match_lines()
        mpix = np.ma.concatenate(self.observed_line_locations).compressed()
        mwav = np.ma.concatenate(self.catalog_line_locations).compressed()
        if space == "wavelength":
            return np.sqrt(((mwav - self.pix_to_wav(mpix)) ** 2).mean())
        elif space == "pixel":
            return np.sqrt(((mpix - self.wav_to_pix(mwav)) ** 2).mean())
        else:
            raise ValueError("Space must be either 'pixel' or 'wavelength'")

    def _plot_lines(
        self,
        kind: Literal["observed", "catalog"],
        frames: int | Sequence[int] | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_labels: bool | Sequence[bool] = True,
        map_x: bool = False,
        label_kwargs: dict | None = None,
    ) -> Figure:
        """
        Plot lines with optional features such as wavelength mapping and label customization.

        Parameters
        ----------
        kind
            Specifies the line list to plot.

        frames
            Frame indices to plot. If None, all frames are plotted.

        axes
            Axes object(s) where the lines should be plotted. If None, new Axes are generated.

        figsize
            Size of the figure to use if creating new Axes. Ignored if axes are provided.

        plot_labels
            Flag(s) indicating whether to display labels for the lines. If a single value is
            provided, it is applied to all frames.

        map_x
            If True, maps the x-axis values between pixel and wavelength space.

        label_kwargs
            Additional keyword arguments to customize the label style.

        Returns
        -------
        Figure
            The Figure object containing the plotted spectral lines.
        """
        largs = dict(backgroundcolor="w", rotation=90, size="small")
        if label_kwargs is not None:
            largs.update(label_kwargs)

        if frames is None:
            frames = np.arange(self.nframes)
        else:
            frames = np.atleast_1d(frames)

        if axes is None:
            fig, axes = subplots(
                frames.size, 1, figsize=figsize, constrained_layout=True, sharex="all"
            )
        elif isinstance(axes, Axes):
            fig = axes.figure
            axes = [axes]
        else:
            fig = axes[0].figure
        axes = np.atleast_1d(axes)

        if isinstance(plot_labels, bool):
            plot_labels = np.full(frames.size, plot_labels, dtype=bool)

        if map_x and self._p2w is None:
            raise ValueError("Cannot map between pixels and wavelengths without a fitted model.")

        if kind == "observed":
            transform = self.pix_to_wav if map_x else lambda x: x
            linelists = self.observed_lines
            spectra = self.arc_spectra
            lc = "C0"
        else:
            transform = self.wav_to_pix if map_x else lambda x: x
            linelists = self.catalog_lines
            spectra = None
            lc = "C1"

        ypad = 1.3
        labels = []
        for iframe, (ax, frame) in enumerate(zip(axes, frames)):
            if spectra is not None:
                spc = self.arc_spectra[iframe]
                vmax = np.nanmax(spc.flux.value)
                ax.plot(transform(spc.spectral_axis.value), spc.flux.value / vmax, "k")
            else:
                vmax = 1.0

            if linelists is not None:
                labels.append([])

                # Loop over individual lines in the line list.
                for i in range(linelists[iframe].shape[0]):
                    c, a = linelists[iframe].data[i]
                    ls = "-" if linelists[iframe].mask[i, 0] == 0 else ":"

                    ax.plot(transform([c, c]), [a / vmax + 0.1, 1.27], c=lc, ls=ls, zorder=-100)
                    if plot_labels[iframe]:
                        lloc = transform(c)
                        labels[-1].append(
                            ax.text(
                                lloc,
                                ypad,
                                np.round(lloc, 4 - 1 - int(np.floor(np.log10(lloc)))),
                                ha="center",
                                va="top",
                                **largs,
                            )
                        )
                        labels[-1][-1].set_clip_on(True)
                        labels[-1][-1].zorder = a

        if (kind == "observed" and not map_x) or (kind == "catalog" and map_x):
            xlabel = "Pixel"
        else:
            xlabel = f"Wavelength {self._unit_str}"

        if kind == "catalog":
            axes[0].xaxis.set_label_position("top")
            axes[0].xaxis.tick_top()
            setp(axes[0], xlabel=xlabel)
            for ax in axes[1:]:
                ax.set_xticklabels([])
        else:
            setp(axes[-1], xlabel=xlabel)
            for ax in axes[:-1]:
                ax.set_xticklabels([])

        xlims = np.array([ax.get_xlim() for ax in axes])
        setp(axes, xlim=(xlims[:, 0].min(), xlims[:, 1].max()), yticks=[])

        if linelists is not None:
            fig.canvas.draw()
            for i in range(len(frames)):
                if plot_labels[i]:

                    # Calculate the label bounding box upper limits and adjust the y-axis limits.
                    tr_to_data = axes[i].transData.inverted()
                    ymax = -np.inf
                    for lb in labels[i]:
                        ymax = max(ymax, tr_to_data.transform(lb.get_window_extent().p1)[1])
                    setp(axes[i], ylim=(-0.04, ymax * 1.06))

                    # Remove the overlapping labels prioritizing the high-amplitude lines.
                    _unclutter_text_boxes(labels[i])

        return fig

    def plot_catalog_lines(
        self,
        frames: int | Sequence[int] | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_labels: bool | Sequence[bool] = True,
        map_to_pix: bool = False,
        label_kwargs: dict | None = None,
    ) -> Figure:
        """Plot the catalog lines.

        Parameters
        ----------
        frames
            Specifies the frames to be plotted. If an integer, only one frame is plotted.
            If a sequence, the specified frames are plotted. If None, default selection
            or all frames are plotted.

        axes
            The matplotlib axes where catalog data will be plotted. If provided, the function
            will plot on these axes. If None, new axes will be created.

        figsize
            Specifies the dimensions of the figure as (width, height). If None, the default
            dimensions are used.

        plot_labels
            If True, the numerical values associated with the catalog data will be displayed
            in the plot. If False, only the graphical representation of the lines will be shown.

        map_to_pix
            Indicates whether the catalog data should be mapped to pixel coordinates
            before plotting. If True, the data is converted to pixel coordinates.

        label_kwargs
            Specifies the keyword arguments for the line label text objects.

        Returns
        -------
        Figure
            The matplotlib figure containing the plotted catalog lines.

        """
        return self._plot_lines(
            "catalog",
            frames=frames,
            axes=axes,
            figsize=figsize,
            plot_labels=plot_labels,
            map_x=map_to_pix,
            label_kwargs=label_kwargs,
        )

    def plot_observed_lines(
        self,
        frames: int | Sequence[int] | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_labels: bool | Sequence[bool] = True,
        map_to_wav: bool = False,
        label_kwargs: dict | None = None,
    ) -> Figure:
        """Plot observed spectral lines for the given arc spectra.

        Parameters
        ----------
        frames
            Specifies the frame(s) for which the plot is to be generated. If None, all frames
            are plotted. When an integer is provided, a single frame is used. For a sequence
            of integers, multiple frames are plotted.

        axes
            Axes object(s) to plot the spectral lines on. If None, new axes are created.

        figsize
            Dimensions of the figure to be created, specified as a tuple (width, height). Ignored
            if ``axes`` is provided.

        plot_labels
            If True, plots the numerical values of the observed lines at their respective
            locations on the graph.

        map_to_wav
            Determines whether to map the x-axis values to wavelengths.

        label_kwargs
            Specifies the keyword arguments for the line label text objects.

        Returns
        -------
        Figure
            The matplotlib figure containing the observed lines plot.
        """

        fig = self._plot_lines(
            "observed",
            frames=frames,
            axes=axes,
            figsize=figsize,
            plot_labels=plot_labels,
            map_x=map_to_wav,
            label_kwargs=label_kwargs,
        )

        for ax in fig.axes:
            ax.autoscale(True, "x", tight=True)
        return fig

    def plot_fit(
        self,
        frames: Sequence[int] | int | None = None,
        figsize: tuple[float, float] | None = None,
        plot_labels: bool = True,
        obs_to_wav: bool = False,
        cat_to_pix: bool = False,
        label_kwargs: dict | None = None,
    ) -> Figure:
        """Plot the fitted catalog and observed lines for the specified arc spectra.

        Parameters
        ----------
        frames
            The indices of the frames to plot. If `None`, all frames from 0 to
            ``self.nframes - 1`` are plotted.

        figsize
            Defines the width and height of the figure in inches. If `None`, the
            default size is used.

        plot_labels
            If `True`, print line locations over the plotted lines. Can also be a list with
            the same length as ``frames``.

        obs_to_wav
            If `True`, transform the x-axis of observed lines to the wavelength domain
            using `self._p2w`, if available.

        cat_to_pix
            If `True`, transforms catalog data points to pixel values before plotting.

        label_kwargs
            Specifies the keyword arguments for the line label text objects.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the generated subplots.
        """
        if frames is None:
            frames = np.arange(self.nframes)
        else:
            frames = np.atleast_1d(frames)

        fig, axs = subplots(2 * frames.size, 1, constrained_layout=True, figsize=figsize)
        self.plot_catalog_lines(
            frames,
            axs[0::2],
            plot_labels=plot_labels,
            map_to_pix=cat_to_pix,
            label_kwargs=label_kwargs,
        )
        self.plot_observed_lines(
            frames,
            axs[1::2],
            plot_labels=plot_labels,
            map_to_wav=obs_to_wav,
            label_kwargs=label_kwargs,
        )

        xlims = np.array([ax.get_xlim() for ax in axs[::2]])
        if obs_to_wav:
            setp(axs, xlim=(xlims[:, 0].min(), xlims[:, 1].max()))
        else:
            setp(axs[::2], xlim=(xlims[:, 0].min(), xlims[:, 1].max()))

        setp(axs[0], yticks=[], xlabel=f"Wavelength [{self._unit_str}]")
        for ax in axs[1:-1]:
            ax.set_xlabel("")
            ax.set_xticklabels("")

        axs[0].xaxis.set_label_position("top")
        axs[0].xaxis.tick_top()
        return fig

    def plot_residuals(
        self,
        ax: Axes | None = None,
        space: Literal["pixel", "wavelength"] = "wavelength",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot the residuals of pixel-to-wavelength or wavelength-to-pixel transformation.

        Parameters
        ----------
        ax
            Matplotlib Axes object to plot on. If None, a new figure and axes are created.

        space
            The reference space used for plotting residuals. Options are 'pixel' for residuals
            in pixel space or 'wavelength' for residuals in wavelength space.

        figsize
            The size of the figure in inches, if a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure

        self.match_lines()
        mpix = np.ma.concatenate(self.observed_line_locations).compressed()
        mwav = np.ma.concatenate(self.catalog_line_locations).compressed()

        if space == "wavelength":
            twav = self.pix_to_wav(mpix)
            ax.plot(mwav, mwav - twav, ".")
            ax.text(
                0.98,
                0.95,
                f"RMS = {np.sqrt(((mwav - twav) ** 2).mean()):4.2f} {self._unit_str}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            setp(
                ax,
                xlabel=f"Wavelength [{self._unit_str}]",
                ylabel=f"Residuals [{self._unit_str}]",
            )
        elif space == "pixel":
            tpix = self.wav_to_pix(mwav)
            ax.plot(mpix, mpix - tpix, ".")
            ax.text(
                0.98,
                0.95,
                f"RMS = {np.sqrt(((mpix - tpix) ** 2).mean()):4.2f} pix",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            setp(ax, xlabel="Pixel", ylabel="Residuals [pix]")
        else:
            raise ValueError("Invalid space specified for plotting residuals.")
        ax.axhline(0, c="k", lw=1, ls="--")
        return fig
