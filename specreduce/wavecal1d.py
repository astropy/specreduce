import warnings
from typing import Sequence, Callable, Literal

import astropy.units as u
import numpy as np
from astropy.modeling import models, Model, fitting
from astropy.nddata import VarianceUncertainty
from gwcs import coordinate_frames as cf
from gwcs import wcs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import setp, subplots
from numpy import ndarray
from numpy.ma.core import MaskedArray
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from specutils import Spectrum1D

from specreduce.calibration_data import load_pypeit_calibration_lines
from specreduce.line_matching import find_arc_lines


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


class WavelengthSolution1D:
    def __init__(
        self,
        ref_pixel: float,
        unit: u.Unit = u.angstrom,
        degree: int = 3,
        line_lists=None,
        arc_spectra: Spectrum1D | Sequence[Spectrum1D] | None = None,
        obs_lines: ndarray | Sequence[ndarray] | None = None,
        pix_bounds: tuple[int, int] | None = None,
        line_list_bounds: tuple[float, float] = (0, np.inf),
        wave_air: bool = False,
    ) -> None:

        self.unit = unit
        self._unit_str = unit.to_string("latex")
        self.degree = degree
        self.ref_pixel = ref_pixel

        self.bounds_pix: tuple[int, int] | None = pix_bounds
        self.bounds_wav: tuple[float, float] | None = None
        self._cat_lines: list[MaskedArray] | None = None
        self._obs_lines: list[MaskedArray] | None = None
        self._trees: Sequence[KDTree] | None = None

        self._fit: optimize.OptimizeResult | None = None
        self._wcs: wcs.WCS | None = None

        self._p2w: Model | None = None  # pixel -> wavelength model
        self._w2p: Callable | None = None  # wavelength -> pixel model
        self._p2w_dldx: Model | None = None  # delta lambda / delta pixel

        # Read and store the observational data if given. The user can provide either a list of arc
        # spectra as Spectrum1D objects, or a list of line pixel position arrays. Attempts to give
        # both raises an error.
        if arc_spectra is not None and obs_lines is not None:
            raise ValueError("Only one of arc_spectra or obs_lines can be provided.")

        if arc_spectra is not None:
            self.arc_spectra = [arc_spectra] if isinstance(arc_spectra, Spectrum1D) else arc_spectra
            self.nframes = len(self.arc_spectra)
            for s in self.arc_spectra:
                if s.data.ndim > 1:
                    raise ValueError("The arc spectra must be 1 dimensional.")
            self.bounds_pix = (0, self.arc_spectra[0].shape[0])

        elif obs_lines is not None:
            self.lines_pix = obs_lines
            self.nframes = len(self._obs_lines)
            if self.bounds_pix is None:
                raise ValueError("Must give pixel bounds when providing observed line positions.")

        # Read the line lists if given. The user can provide an array of line wavelength positions
        # or a list of line list names (used by `load_pypeit_calibration_lines`) for each arc
        # spectrum.
        if line_lists is not None:
            if len(line_lists) != self.nframes:
                raise ValueError("The number of line lists must match the number of arc spectra.")
            self._read_linelists(line_lists, line_list_bounds=line_list_bounds, wave_air=wave_air)

    def _init_model(self):
        self._p2w = models.Shift(-self.ref_pixel) | models.Polynomial1D(self.degree)

    def _read_linelists(
        self,
        line_lists,
        line_list_bounds: tuple[float, float] = (0.0, np.inf),
        wave_air: bool = False,
    ):
        """Load and filter calibration line lists for specified lamps and wavelength bounds.

        Notes
        -----
        This method uses the `load_pypeit_calibration_lines` function to load the
        line lists for each lamp. It filters the loaded data to only include lines
        within the defined wavelength boundaries. The filtered data is stored in
        `self.linelists`, and the wavelengths are extracted and stored in
        `self.lines_wav`. Additionally, KDTree objects are created for each extracted
        wavelength list and stored in `self._trees` for quick nearest-neighbor
        queries of line wavelengths.
        """
        if not isinstance(line_lists, (tuple, list)):
            line_lists = [line_lists]

        lines_wav = []
        for l in line_lists:
            if isinstance(l, ndarray):
                lines_wav.append(l)
            else:
                lines = []
                if isinstance(l, str):
                    l = [l]
                for ll in l:
                    lines.append(
                        load_pypeit_calibration_lines(ll, wave_air=wave_air)["wavelength"]
                        .to(self.unit)
                        .value
                    )
                lines_wav.append(np.ma.masked_array(np.sort(np.concatenate(lines))))
        for i, l in enumerate(lines_wav):
            lines_wav[i] = l[(l >= line_list_bounds[0]) & (l <= line_list_bounds[1])]

        self.lines_wav = lines_wav
        self._trees = [KDTree(l.data[:, None]) for l in self._cat_lines]

    def find_lines(self, fwhm: float, noise_factor: float = 1.0) -> None:
        """Find lines in the provided arc spectra.

        This method determines the spectral lines within each spectrum of the arc spectra
        based on the provided initial guess for the line Full Width at Half Maximum (FWHM).

        Parameters
        ----------
        fwhm
            Initial guess for the FWHM for the spectral lines, used as a parameter in
            the ``find_arc_lines`` function to locate and identify spectral arc lines.
        noise_factor
            The factor to multiply the uncertainty by to determine the noise threshold
            in the `~specutils.fitting.find_lines_threshold` routine.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines_obs = [
                find_arc_lines(sp, fwhm, noise_factor=noise_factor) for sp in self.arc_spectra
            ]
        self._obs_lines = [np.ma.masked_array(lo["centroid"].value) for lo in lines_obs]

    def fit_lines(
        self,
        pixels: Sequence,
        wavelengths: Sequence,
        match_pix: bool = True,
        match_wav: bool = True,
    ) -> None:
        """Fit wavelength calibration lines to a model.

        This method takes input arrays of pixel values and their corresponding wavelengths and fits
        them to a calibration model, updating the pixel-to-wavelength transformation for the system.
        Optionally, the method can match the provided pixel or wavelength values to known catalog
        or observed values for better fitting.

        Parameters
        ----------
        pixels
            A sequence of pixel positions to be used for fitting.
        wavelengths
            A sequence of associated wavelengths corresponding to the pixel positions.
        match_pix
            A flag indicating whether to match the input pixel values to observed pixel values.
        match_wav
            A flag indicating whether to match the input wavelengths to catalog values.
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
        if match_wav:
            tree = KDTree(np.concatenate([c.data for c in self._cat_lines])[:, None])
            ix = tree.query(wavelengths[:, None])[1]
            wavelengths = tree.data[ix][:, 0]

        # Match the input pixel values to observed pixel values.
        if match_pix:
            tree = KDTree(np.concatenate([c.data for c in self._obs_lines])[:, None])
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
        self._calculate_p2w_derivative()
        self._calculate_p2w_inverse()

    def fit(
        self,
        wavelength_bounds: tuple[float, float],
        dispersion_bounds: tuple[float, float],
        popsize: int = 30,
        max_distance: float = 100,
        refine_fit: bool = True,
    ):
        """Calculate and refine the pixel-to-wavelength and wavelength-to-pixel transformations.

        This method determines the functional relationships between pixel positions and
        wavelengths in a spectrum, including their derivatives, by fitting calibration data
        to given constraints. The transformations include forward (pixel-to-wavelength)
        and backward (wavelength-to-pixel) mappings as well as their respective derivatives
        across the spectral axis.

        Parameters
        ----------
        wavelength_bounds
            Initial bounds for the wavelength value at the reference pixel.
        dispersion_bounds
            Initial bounds for the dispersion value at the reference pixel.
        popsize
            The population size for differential evolution optimization.
        max_distance
            The maximum allowable distance when querying the tree in the optimization process.
        refine_fit
            Refine the global fit of the pixel-to-wavelength transformation.

        Raises
        ------
        ValueError
            Raised if the optimization or fitting process fails due to invalid inputs or constraints.
        """

        if self._p2w is None:
            self._init_model()
        model = self._p2w

        def minfun(x):
            total_distance = 0.0
            for t, l in zip(self._trees, self._obs_lines):
                transformed_lines = model.evaluate(l, -self.ref_pixel, *x)[:, None]
                total_distance += np.clip(t.query(transformed_lines)[0], 0, max_distance).sum()
            return total_distance

        bounds = np.concatenate(
            [
                [wavelength_bounds, dispersion_bounds, [-1e-3, 1e-3]],
                np.zeros((model[1].degree - 2, 2)),
            ]
        )
        self._fit = fit = optimize.differential_evolution(
            minfun,
            bounds=bounds,
            popsize=popsize,
        )
        self._p2w = models.Shift(-self.ref_pixel) | models.Polynomial1D(
            self.degree, **{f"c{i}": fit.x[i] for i in range(fit.x.size)}
        )

        if refine_fit:
            self.refine_fit()
        self._calculate_p2w_derivative()
        self._calculate_p2w_inverse()

    def refine_fit(self, match_distance_bound: float = 5.0):
        """Refine the global fit of the pixel-to-wavelength transformation.

        Refines the fit of a polynomial model to data by performing a fitting operation
        using matched pixel and wavelength data points. The method uses a linear least
        squares fitter to optimize the model parameters based on the match.

        Parameters
        ----------
        degree : int, optional
            The degree of the polynomial model used for fitting. Higher values
            allow for more complex polynomial models. Default is 4.

        match_distance_bound : float, optional
            Maximum allowable distance used to identify matched pixel and wavelength
            data points. Points exceeding the bound will not be considered in the fit.
            Default is 5.0.
        """
        self.match_lines(match_distance_bound)
        matched_pix = np.ma.concatenate(self._obs_lines).compressed()
        matched_wav = np.ma.concatenate(self._cat_lines).compressed()
        model = self._p2w[1]
        fitter = fitting.LinearLSQFitter()
        self._p2w = self._p2w[0].copy() | fitter(model, matched_pix - self.ref_pixel, matched_wav)
        self._calculate_p2w_derivative()
        self._calculate_p2w_inverse()
        self.match_lines(match_distance_bound)

    def _calculate_p2w_derivative(self):
        """Calculate (d wavelength) / (d pixel) for the pixel-to-wavelength transformation."""
        if self._p2w is not None:
            self._p2w_dldx = models.Shift(self._p2w.offset_0) | _diff_poly1d(self._p2w[1])

    def _calculate_p2w_inverse(self) -> None:
        """Compute the wavelength-to-pixel mapping from the pixel-to-wavelength transformation."""
        p = np.arange(*self.bounds_pix)
        self._w2p = interp1d(self._p2w(p), p, bounds_error=False, fill_value=np.nan)
        self.bounds_wav = self._p2w(self.bounds_pix)

    def resample(
        self,
        spectrum: Spectrum1D,
        nbins: int | None = None,
        wlbounds: tuple[float, float] | None = None,
        bin_edges: Sequence[float] | None = None,
    ) -> Spectrum1D:
        """Bin the given pixel-space 1D spectrum to a wavelength space conserving the flux.

        This method bins a pixel-space spectrum to a wavelength space using the computed pixel-to-wavelength and
        wavelength-to-pixel transformations and their derivatives with respect to the spectral axis. The binning is
        exact and conserves the total flux.

        Parameters
        ----------
        spectrum
            A Spectrum1D instance containing the flux to be resampled over the wavelength space.
        nbins
            The number of bins for resampling. If not provided, it defaults to the size of the input spectrum.
        wlbounds
            A tuple specifying the starting and ending wavelengths for resampling. If not provided, the
            wavelength bounds are inferred from the object's methods and the entire flux array is used.
        bin_edges
            Explicit bin edges in the wavelength space. If provided, `nbins` and `wlbounds` are ignored.

        Returns
        -------
            1D spectrum binned to the specified wavelength bins.
        """
        flux = spectrum.flux.value
        ucty = spectrum.uncertainty.represent_as(VarianceUncertainty).array
        npix = flux.size
        nbins = npix if nbins is None else nbins
        if wlbounds is None:
            l1 = self._p2w(0)
            l2 = self._p2w(npix - 1)
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
        n = flux.sum() / (dldx * flux).sum()
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
        ucty_wl = VarianceUncertainty(ucty_wl * n).represent_as(type(spectrum.uncertainty))
        return Spectrum1D(flux_wl, bin_centers_wav * u.angstrom, uncertainty=ucty_wl)

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
    def lines_pix(self) -> list[MaskedArray]:
        """List of pixel positions of identified spectral lines."""
        return self._obs_lines

    @lines_pix.setter
    def lines_pix(self, lines_pix: MaskedArray | ndarray | list[MaskedArray] | list[ndarray]):
        if not isinstance(lines_pix, Sequence):
            lines_pix = [lines_pix]
        self._obs_lines = []
        for l in lines_pix:
            if isinstance(l, MaskedArray) and l.mask is not np.False_:
                self._obs_lines.append(l)
            else:
                self._obs_lines.append(np.ma.masked_array(l, mask=np.zeros(l.size, bool)))

    @property
    def lines_wav(self) -> list[MaskedArray]:
        """List of wavelength positions of theoretical spectral lines."""
        return self._cat_lines

    @lines_wav.setter
    def lines_wav(self, lines_wav: MaskedArray | ndarray | list[MaskedArray] | list[ndarray]):
        if not isinstance(lines_wav, Sequence):
            lines_wav = [lines_wav]
        self._cat_lines = []
        for l in lines_wav:
            if isinstance(l, MaskedArray) and l.mask is not np.False_:
                self._cat_lines.append(l)
            else:
                self._cat_lines.append(np.ma.masked_array(l, mask=np.zeros(l.size, bool)))

    @property
    def wcs(self):
        pixel_frame = cf.CoordinateFrame(
            1,
            "SPECTRAL",
            (0,),
            axes_names=[
                "x",
            ],
            unit=[u.pix],
        )
        spectral_frame = cf.SpectralFrame(
            axes_names=("wavelength",),
            unit=[self.unit],
        )
        pipeline = [(pixel_frame, self._p2w), (spectral_frame, None)]
        self._wcs = wcs.WCS(pipeline)
        return self._wcs

    def match_lines(self, upper_bound: float = 5) -> None:
        """Match the observed lines to theoretical lines.

        Parameters
        ----------
        upper_bound
            The maximum allowed distance between the query points and the KD-tree
            data points for them to be considered a match.
        """
        matched_lines_wav = []
        matched_lines_pix = []
        for iframe, tree in enumerate(self._trees):
            l, ix = tree.query(
                self._p2w(self._obs_lines[iframe].data)[:, None],
                distance_upper_bound=upper_bound,
            )
            m = np.isfinite(l)

            # Check for observed lines that match to a same catalog line.
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

            matched_lines_wav.append(np.ma.masked_array(tree.data[:, 0], mask=True))
            matched_lines_wav[-1].mask[ix[m]] = False
            matched_lines_pix.append(np.ma.masked_array(self._obs_lines[iframe].data, mask=~m))

        self._obs_lines = matched_lines_pix
        self._cat_lines = matched_lines_wav

        if any([o.count() != c.count() for o, c in zip(self._obs_lines, self._cat_lines)]):
            warnings.warn(
                "Line matching failed, the number of matched catalog lines != "
                "the number of matched observed lines."
            )

    def remove_ummatched_lines(self):
        self._obs_lines = [np.ma.masked_array(l.compressed()) for l in self._obs_lines]
        self._cat_lines = [np.ma.masked_array(l.compressed()) for l in self._cat_lines]

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
        mpix = np.ma.concatenate(self.lines_pix).compressed()
        mwav = np.ma.concatenate(self.lines_wav).compressed()
        if space == "wavelength":
            return np.sqrt(((mwav - self.pix_to_wav(mpix)) ** 2).mean())
        else:
            return np.sqrt(((mpix - self.wav_to_pix(mwav)) ** 2).mean())

    def _plot_lines(
        self,
        kind: Literal["observed", "catalog"],
        frames: int | Sequence[int] | None = None,
        axs: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_values: bool | Sequence[bool] = True,
        map_x: bool = False,
    ) -> Figure:

        if frames is None:
            frames = np.arange(self.nframes)
        else:
            frames = np.atleast_1d(frames)

        if axs is None:
            fig, axs = subplots(frames.size, 1, figsize=figsize, constrained_layout=True)
        elif isinstance(axs, Axes):
            fig = axs.figure
            axs = [axs]
        else:
            fig = axs[0].figure

        if isinstance(plot_values, bool):
            plot_values = np.full(frames.size, plot_values, dtype=bool)

        if map_x and self._p2w is None:
            raise ValueError("Cannot map between pixels and wavelengths without a fitted model.")

        if kind == "observed":
            transform = self.pix_to_wav if map_x else lambda x: x
            linelists = self._obs_lines
        else:
            transform = self.wav_to_pix if map_x else lambda x: x
            linelists = self._cat_lines

        if linelists is not None:
            for iax, (ax, frame) in enumerate(zip(axs, frames)):
                lines = linelists[frame]
                ax.vlines(transform(lines[lines.mask].data), 0, 1, ls=":")
                ax.vlines(transform(lines[~lines.mask].data), 0, 1)
                if plot_values[iax]:
                    for i, l in enumerate(transform(lines.data)):
                        if np.isfinite(l):
                            ax.text(
                                l,
                                0.25 + 0.25 * (i % 3),
                                f"{l:.0f}",
                                rotation=90,
                                ha="right",
                                va="center",
                                bbox=dict(alpha=0.8, fc="w", lw=0),
                                size="small",
                            )

        if (kind == "observed" and not map_x) or (kind == "catalog" and map_x):
            xlabel = "Pixel"
        else:
            xlabel = f"Wavelength {self._unit_str}"

        if kind == "catalog":
            axs[0].xaxis.set_label_position("top")
            axs[0].xaxis.tick_top()
            setp(axs[0], xlabel=xlabel)
            for ax in axs[1:]:
                ax.set_xticklabels([])
        else:
            setp(axs[-1], xlabel=xlabel)
            for ax in axs[:-1]:
                ax.set_xticklabels([])
        xlims = np.array([ax.get_xlim() for ax in axs])
        setp(axs, xlim=(xlims[:, 0].min(), xlims[:, 1].max()), yticks=[])
        return fig

    def plot_catalog_lines(
        self,
        frames: int | Sequence[int] | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_values: bool = True,
        map_to_pix: bool = False,
    ) -> Figure:
        return self._plot_lines(
            "catalog",
            frames=frames,
            axs=axes,
            figsize=figsize,
            plot_values=plot_values,
            map_x=map_to_pix,
        )

    def plot_observed_lines(
        self,
        frames: int | Sequence[int] | None = None,
        axes: Axes | Sequence[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        plot_values: bool = True,
        plot_spectra: bool = True,
        map_to_wav: bool = False,
    ) -> Figure:

        fig = self._plot_lines(
            "observed",
            frames=frames,
            axs=axes,
            figsize=figsize,
            plot_values=plot_values,
            map_x=map_to_wav,
        )

        if axes is None:
            axes = fig.axes

        if self.arc_spectra is not None and plot_spectra:
            if frames is None:
                frames = np.arange(self.nframes)
            elif np.isscalar(frames):
                frames = [frames]

            transform = self._p2w if map_to_wav else lambda x: x
            for i, frame in enumerate(frames):
                axes[i].plot(
                    transform(self.arc_spectra[frame].spectral_axis.value),
                    self.arc_spectra[frame].data / (1.2 * self.arc_spectra[frame].data.max()),
                    c="k",
                    zorder=-10,
                )
            setp(
                axes,
                xlim=transform(
                    [
                        self.arc_spectra[0].spectral_axis.min().value,
                        self.arc_spectra[0].spectral_axis.max().value,
                    ]
                ),
            )
        return fig

    def plot_fit(
        self,
        frames: Sequence[int] | int | None = None,
        figsize: tuple[float, float] | None = None,
        plot_values: bool = True,
        obs_to_wav: bool = False,
        cat_to_pix: bool = False,
    ) -> Figure:

        if frames is None:
            frames = np.arange(self.nframes)
        else:
            frames = np.atleast_1d(frames)

        if self._p2w is not None and obs_to_wav:
            transform = self._p2w
        else:
            transform = lambda x: x

        fig, axs = subplots(2 * frames.size, 1, constrained_layout=True, figsize=figsize)
        self.plot_catalog_lines(frames, axs[::2], plot_values=plot_values, map_to_pix=cat_to_pix)
        self.plot_observed_lines(frames, axs[1::2], plot_values=plot_values, map_to_wav=obs_to_wav)

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
            Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
        space
            The reference space used for plotting residuals. Options are 'pixel' for residuals in pixel space or
            'wavelength' for residuals in wavelength space.
        figsize
            The size of the figure in inches, if a new figure is created. Default is None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure

        self.match_lines()
        mpix = np.ma.concatenate(self.lines_pix).compressed()
        mwav = np.ma.concatenate(self.lines_wav).compressed()

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
        else:
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
        ax.axhline(0, c="k", lw=1, ls="--")
        return fig
