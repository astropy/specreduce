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
        self._unit_str = unit.to_string('latex')
        self.degree = degree
        self.ref_pixel = ref_pixel

        self.bounds_pix: tuple[int, int] | None = pix_bounds
        self.bounds_wav: tuple[float, float] | None = None
        self._lines_wav: list[MaskedArray] | None = None
        self._lines_pix: list[MaskedArray] | None = None
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
            self._lines_pix = [obs_lines] if isinstance(obs_lines, ndarray) else obs_lines
            self.nframes = len(self._lines_pix)
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

        self._lines_wav = []
        for l in line_lists:
            if isinstance(l, ndarray):
                self._lines_wav.append(l)
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
                self._lines_wav.append(np.ma.masked_array(np.sort(np.concatenate(lines))))
        for i, l in enumerate(self._lines_wav):
            self._lines_wav[i] = l[(l >= line_list_bounds[0]) & (l <= line_list_bounds[1])]
        self._trees = [KDTree(l.data[:, None]) for l in self._lines_wav]

    def find_lines(self, fwhm: float):
        """Finds lines in the provided arc spectra.

        This method determines the spectral lines within each spectrum of the arc spectra
        based on the provided initial guess for the line Full Width at Half Maximum (FWHM).

        Parameters
        ----------
        fwhm
            Initial guess for the FWHM for the spectral lines, used as a parameter in
            the ``find_arc_lines`` function to locate and identify spectral arc lines.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines_obs = [find_arc_lines(sp, fwhm) for sp in self.arc_spectra]
        self._lines_pix = [np.ma.masked_array(lo["centroid"].value) for lo in lines_obs]

    def fit_lines(self, pixels: Sequence, wavelengths: Sequence) -> None:
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
        ref_pixel: float,
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
        ref_pixel
            The reference pixel position used as the zero point for the transformation.
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
        model = models.Shift(-ref_pixel) | models.Polynomial1D(3)

        def minfun(x):
            total_distance = 0.0
            for t, l in zip(self._trees, self._lines_pix):
                transformed_lines = model.evaluate(l, -ref_pixel, *x)[:, None]
                total_distance += np.clip(t.query(transformed_lines)[0], 0, max_distance).sum()
            return total_distance

        self._fit = fit = optimize.differential_evolution(
            minfun,
            bounds=[wavelength_bounds, dispersion_bounds, [-1e-3, 1e-3], [-1e-4, 1e-4]],
            popsize=popsize,
        )
        self._p2w = models.Shift(-ref_pixel) | models.Polynomial1D(
            3, **{f"c{i}": fit.x[i] for i in range(fit.x.size)}
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
        matched_pix = np.ma.concatenate(self._lines_pix).compressed()
        matched_wav = np.ma.concatenate(self._lines_wav).compressed()
        model = self._p2w[1]
        fitter = fitting.LinearLSQFitter()
        self._p2w = self._p2w[0].copy() | fitter(
            model, matched_pix - self.ref_pixel, matched_wav
        )
        self._calculate_p2w_derivative()
        self._calculate_p2w_inverse()

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

    def pix_to_wav(self, pix: ndarray | float) -> ndarray | float:
        """Map pixel values into wavelength values.

        Parameters
        ----------
        pix
            Input pixel value(s) to be transformed into wavelength.

        Returns
        -------
        Transformed wavelength value(s) corresponding to the input pixel value(s).
        """
        return self._p2w(pix)

    def wav_to_pix(self, wav: ndarray | float) -> ndarray | float:
        """Map wavelength values into pixel values.

        Parameters
        ----------
        wav
            The wavelength value(s) to be converted into pixel value(s).

        Returns
        -------
        The corresponding pixel value(s) for the input wavelength(s).
        """
        return self._w2p(wav)

    @property
    def lines_pix(self) -> list[MaskedArray]:
        """List of pixel positions of identified spectral lines."""
        return self._lines_pix

    @property
    def lines_wav(self) -> list[MaskedArray]:
        """List of wavelength positions of theoretical spectral lines."""
        return self._lines_wav

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
        pipeline = [(pixel_frame, self.fitted_model), (spectral_frame, None)]
        self._wcs = wcs.WCS(pipeline)
        return self._wcs

    def match_lines(self, upper_bound: float = 5, concatenate_frames: bool = True) -> None:
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
                self._p2w(self._lines_pix[iframe])[:, None], distance_upper_bound=upper_bound
            )
            m = np.isfinite(l)
            matched_lines_wav.append(np.ma.masked_array(tree.data[:, 0], mask=True))
            matched_lines_wav[-1].mask[ix[m]] = False
            matched_lines_pix.append(np.ma.masked_array(self._lines_pix[iframe], mask=~m))

        self._lines_pix = matched_lines_pix
        self._lines_wav = matched_lines_wav

    def remove_ummatched_lines(self):
        self._lines_pix = [np.ma.masked_array(l.compressed()) for l in self._lines_pix]
        self._lines_wav = [np.ma.masked_array(l.compressed()) for l in self._lines_wav]

    def rms(self, space: Literal["pixel", "wavelength"] = "wavelength") -> float:
        """Compute the RMS of the residuals between matched lines in either the pixel or wavelength space.

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
        mpix, mwav = self.match_lines()
        if space == "wavelength":
            return np.sqrt(((mwav - self.pix_to_wav(mpix)) ** 2).mean())
        else:
            return np.sqrt(((mpix - self.wav_to_pix(mwav)) ** 2).mean())

    def plot(
        self,
        axes: Sequence[Axes] | None = None,
        frames: Sequence[int] | int | None = None,
        figsize: tuple[float, float] | None = None,
        map_x: bool = False,
        plot_obs_lines: bool = True,
        plot_listed_lines: bool = True,
        plot_line_values: bool = True,
    ) -> Figure:
        """Plot the arc spectra and mark identified line positions for all the data sets.

        This method plots the spectral data stored in `arc_spectra` for each dataset
        in separate subplots. It also marks identified spectral line positions within
        the plots.

        Parameters
        ----------
        axes
            Pre-configured Matplotlib axes to be used for plotting. If `None`, new
            axes objects are created automatically, arranged in a single-column layout.
        figsize
            Specifies the width and height of the figure in inches when creating new
            axes. Ignored if `axes` is provided.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if frames is None:
            frames = np.arange(self.nframes)
        elif np.isscalar(frames):
            frames = [frames]
        nframes = len(frames)

        if axes is None:
            fig, axes = subplots(
                nframes, 1, figsize=figsize, sharex="all", constrained_layout=True, squeeze=False
            )
        else:
            axes = np.atleast_2d([axes])
            fig = axes[0, 0].figure

        if map_x:
            xlabel = f"Wavelength [{self.unit.to_string('latex')}]"
            transform = self._p2w
        else:
            xlabel = "Pixel"
            transform = lambda x: x

        for iax, ifr in enumerate(frames):
            if self.arc_spectra is not None:
                axes[iax, 0].plot(
                    transform(self.arc_spectra[ifr].spectral_axis.value),
                    self.arc_spectra[ifr].data / (1.2 * self.arc_spectra[ifr].data.max()),
                    c="k",
                    zorder=10,
                )

            if plot_obs_lines and self._lines_pix is not None:
                lpix = self._lines_pix[ifr].data
                axes[iax, 0].vlines(transform(lpix), 0.0, 1, alpha=0.5)
                axes[iax, 0].vlines(transform(lpix), 0.9, 1, zorder=14)
                if plot_line_values:
                    for i, l in enumerate(sorted(lpix)):
                        axes[iax, 0].text(
                            transform(l),
                            0.25 + 0.25 * (i % 3),
                            f"{l:.0f}",
                            rotation=90,
                            ha="right",
                            va="center",
                            bbox=dict(alpha=0.8, fc="w", lw=0),
                            zorder=20,
                            size='small'
                        )

            if plot_listed_lines and map_x:
                lwav = self._lines_wav[ifr].data
                axes[iax, 0].vlines(
                    lwav, 0.0, 1, alpha=0.5, color="C1", ls="--", zorder=11
                )
                axes[iax, 0].vlines(lwav, 0.95, 1, color="C1", lw=4, zorder=13)

            axes[iax, 0].autoscale(enable=True, axis="x", tight=True)
            setp(axes[iax, 0], yticks=[])
        setp(axes[-1], xlabel=xlabel)
        return fig

    def plot_fit(self, frame: int = 0, figsize: tuple[float, float] | None = None,
                 plot_values: bool = True, transform_x: bool = False) -> Figure:
        fig, axs = subplots(2, 1, constrained_layout=True, figsize=figsize)

        if self._lines_wav is not None:
            axs[0].vlines(self._lines_wav[frame].data, 0, 1)
            if plot_values:
                for i, l in enumerate(sorted(self._lines_wav[frame].data)):
                    axs[0].text(l, 0.25 + 0.25 * (i % 3), f"{l:.0f}", rotation=90, ha='right',
                                va='center', bbox=dict(alpha=0.8, fc='w', lw=0), size='small')

        if transform_x and self._p2w is None:
            raise ValueError("Pixel to wavelength transform does not exist.")
        self.plot(axs[1:], plot_line_values=plot_values, map_x=transform_x, plot_listed_lines=False)
        if transform_x:
            axs[1].set_xlim(axs[0].get_xlim())

        setp(axs[0], yticks=[], xlabel=f'Wavelength [{self._unit_str}]')
        axs[0].xaxis.set_label_position('top')
        axs[0].xaxis.tick_top()
        return fig

    def plot_solution(
        self,
        axes: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        model: Callable | None = None,
    ) -> Figure:
        """Plot the wavelength solution applied to the provided spectra and overlay it for visualization.

        This method generates plots for the given arc spectra, showcasing the results of the model
        fit. Each subplot represents a single spectrum with overlaid model predictions and markers
        indicating the expected wavelengths.

        Parameters
        ----------
        axes
            Array of Matplotlib Axes where the spectra and their corresponding solutions will be plotted.
            If None, new Axes objects will be created for visualization. Must be provided in the case of
            external figure context.
        figsize
            Tuple specifying the dimensions of the figure in inches if new Axes are created. Ignored if
            `axes` is provided. Must follow the structure (width, height).
        model
            The model function to be applied for prediction on the spectral axis values. If None, the
            already fitted model within the class instance (`self.fitted_model`) will be utilized.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if axes is None:
            fig, axes = subplots(
                self.nframes,
                1,
                figsize=figsize,
                sharex="all",
                constrained_layout=True,
                squeeze=False,
            )
        else:
            fig = axes[0].figure

        model = model if model is not None else self._p2w

        for i in range(self.nframes):
            if self.arc_spectra is not None:
                sp = self.arc_spectra[i]
                axes[i, 0].plot(model(sp.spectral_axis.value), sp.data / (1.2 * sp.data.max()))
            axes[i, 0].vlines(self._lines_wav[i], 0.0, 1.0, alpha=0.3, ec="darkorange", zorder=0)
            axes[i, 0].vlines(model(self._lines_pix[i]), 0.9, 1.0, alpha=1)
            axes[i, 0].autoscale(enable=True, axis="x", tight=True)
        setp(axes[-1], xlabel=f'Wavelength [{self.unit.to_string(format="latex")}]')
        return fig

    def plot_transforms(
        self, figsize: tuple[float, float] | None = None, plim: tuple[int, int] | None = None
    ) -> Figure:
        """Plot and visualize transformation functions between pixel and wavelength space.

        This method generates a grid of subplots to illustrate the transformations
        between pixel and wavelength spaces in two directions: Pixel -> Wavelength
        and Wavelength -> Pixel. It also includes visualizations of the derivatives
        of these transformations with respect to the spectral axis.

        Parameters
        ----------
        figsize
            Width, height in inches to control the size of the figure.
        plim
            Lower and upper limits for pixel values used for plotting.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axs = subplots(2, 2, figsize=figsize, constrained_layout=True, sharex="col")
        if self.arc_spectra is not None and plim is None:
            xpix = self.arc_spectra[0].spectral_axis.value
        else:
            xpix = np.arange(*(plim or (0, 2000)))

        xwav = np.linspace(*self.pix_to_wav(xpix[[0, -1]]), num=xpix.size)
        axs[0, 0].plot(xpix, self._p2w(xpix), "k")
        axs[1, 0].plot(xpix[:-1], np.diff(self._p2w(xpix)) / np.diff(xpix), lw=4, c="k")
        axs[1, 0].plot(xpix, self._p2w_dldx(xpix), ls="--", lw=2, c="w")
        axs[0, 1].plot(xwav, self._w2p(xwav), "k")
        axs[1, 1].plot(xwav[:-1], np.diff(self._w2p(xwav)) / np.diff(xwav), lw=4, c="k")
        axs[1, 1].plot(xwav, self._w2p_dxdl(xwav), ls="--", lw=2, c="w")
        setp(axs[1, 0], xlabel="Pixel", ylabel=r"d$\lambda$/dx")
        setp(axs[0, 0], ylabel=rf"$\lambda$ [{self.unit}]")
        setp(axs[1, 1], xlabel=rf"$\lambda$ [{self.unit}]", ylabel=r"dx/d$\lambda$")
        setp(axs[0, 1], ylabel="Pixel")
        axs[0, 0].set_title("Pixel -> wavelength")
        axs[0, 1].set_title("Wavelength -> pixel")
        fig.align_labels()
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

        mpix, mwav = self.match_lines()

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
                xlabel=f'Wavelength [{self._unit_str}]',
                ylabel=f'Residuals [{self._unit_str}]',
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
