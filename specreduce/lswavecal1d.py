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
from scipy import optimize
from scipy.spatial import KDTree

from numpy import ndarray
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
    coeffs = {f'c{i-1}': i*getattr(m, f'c{i}').value for i in range(1, m.degree+1)}
    return models.Polynomial1D(m.degree-1, **coeffs)


class WavelengthSolution1D:
    def __init__(self, *, line_lists,
                 arc_spectra: Spectrum1D | Sequence[Spectrum1D] | None = None,
                 obs_lines: ndarray | Sequence[ndarray] | None = None,
                 wlbounds: tuple[float, float] = (0, np.inf),
                 unit: u.Unit = u.angstrom,
                 wave_air: bool = False):

        if arc_spectra is None and obs_lines is None:
            raise ValueError("Either arc_spectra or obs_lines must be provided.")

        if arc_spectra is not None and obs_lines is not None:
            raise ValueError("Only one of arc_spectra or obs_lines can be provided.")

        self.wlbounds: tuple[float, float] = wlbounds
        self.unit: u.Unit = unit
        self.wave_air: bool = wave_air

        self.arc_spectra: Sequence[Spectrum1D] = [arc_spectra] if isinstance(arc_spectra, Spectrum1D) else arc_spectra
        self.lines_pix: Sequence[ndarray] = [obs_lines] if isinstance(obs_lines, ndarray) else obs_lines
        self.ndata: int = len(self.arc_spectra) if self.arc_spectra is not None else len(self.lines_pix)

        self.lines_wav: Sequence[ndarray] | None = None
        self._trees: Sequence[KDTree] | None = None
        self._read_linelists(line_lists)

        self._fit: optimize.OptimizeResult | None = None
        self._wcs: wcs.WCS | None = None

        self._p2w: Model | None = None           # The fitted pixel -> wavelength model
        self._w2p: Model | None = None           # The fitted wavelength -> pixel model
        self._p2w_dldx: Model | None = None      # delta lambda / delta pixel
        self._w2p_dxdl: Model | None = None      # delta pixel / delta lambda

    def _read_linelists(self, line_lists):
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

        self.lines_wav = []
        for l in line_lists:
            if isinstance(l, ndarray):
                self.lines_wav.append(l)
            else:
                lines = []
                if isinstance(l, str):
                    l = [l]
                for ll in l:
                    lines.append(load_pypeit_calibration_lines(ll, wave_air=self.wave_air)['wavelength'].to(self.unit).value)
                self.lines_wav.append(np.concatenate(lines))
        for i, l in enumerate(self.lines_wav):
            self.lines_wav[i] = l[(l >= self.wlbounds[0]) & (l <= self.wlbounds[1])]
        self._trees = [KDTree(l[:, None]) for l in self.lines_wav]

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
            warnings.simplefilter('ignore')
            lines_obs = [find_arc_lines(sp, fwhm) for sp in self.arc_spectra]
        self.lines_pix = [lo['centroid'].value for lo in lines_obs]

    def fit(self, ref_pixel: float,
            wavelength_bounds: tuple[float, float],
            dispersion_bounds: tuple[float, float],
            popsize: int = 30,
            max_distance: float = 100,
            refine_fit: bool = True):
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
            for t, l in zip(self._trees, self.lines_pix):
                transformed_lines = model.evaluate(l, -ref_pixel, *x)[:, None]
                total_distance += np.clip(t.query(transformed_lines)[0], 0, max_distance).sum()
            return total_distance

        self._fit = fit = optimize.differential_evolution(minfun,
                                                          bounds=[wavelength_bounds,
                                                                  dispersion_bounds,
                                                                  [-1e-3, 1e-3],
                                                                  [-1e-4, 1e-4]],
                                                          popsize=popsize)
        self._p2w = (models.Shift(-ref_pixel) |
                     models.Polynomial1D(3, **{f'c{i}': fit.x[i] for i in range(fit.x.size)}))

        if refine_fit:
            self._refine_fit()
        self._calculate_inverse()
        self._calculate_derivatives()

    def _calculate_inverse(self):
        """Compute the wavelength-to-pixel mapping from the pixel-to-wavelength transformation.

        Compute and set the inverse mapping from wavelength reference system to pixel reference system
        through polynomial fitting. This method uses the DogBoxLSQFitter to fit the data and produces
        a transformation model to establish the inverse relation.
        """
        vpix = self.arc_spectra[0].spectral_axis.value
        vwav = self._p2w(vpix)
        w2p = models.Polynomial1D(6, c0=-self._p2w.offset_0, c1=1/self._p2w.c1_1, fixed={'c0': True})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter = fitting.DogBoxLSQFitter()
            self._w2p = models.Shift(-self._p2w.c0_1) | fitter(w2p, vwav - self._p2w.c0_1.value, vpix)

    def _calculate_derivatives(self):
        """Calculate the forward and reverse mapping derivatives with respect to the spectral axis.
        """
        if self._p2w is not None:
            self._p2w_dldx = models.Shift(self._p2w.offset_0) | _diff_poly1d(self._p2w[1])
        if self._w2p is not None:
            self._w2p_dxdl = models.Shift(self._w2p.offset_0) | _diff_poly1d(self._w2p[1])


    def _refine_fit(self, degree: int = 4, match_distance_bound: float = 5.0):
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
        matched_pix, matched_wav = self.match_lines(match_distance_bound)
        model = models.Polynomial1D(degree, **{n: getattr(self._p2w[-1], n).value for n in self._p2w[-1].param_names})
        fitter = fitting.LinearLSQFitter()
        self._p2w = self._p2w[0].copy() | fitter(model, matched_pix + self._p2w.offset_0.value, matched_wav)


    def resample(self, spectrum: Spectrum1D,
                 nbins: int | None = None,
                 wlbounds: tuple[float, float] | None = None,
                 bin_edges: Sequence[float] | None = None) -> Spectrum1D:
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
            i1, i2 = bin_edge_ix[i:i + 2]
            weights[:] = 0
            if i1 != i2:
                weights[i1 + 1:i2] = 1.0
                weights[i1] = 1 - bin_edge_w[i]
                weights[i2] = bin_edge_w[i + 1]
                flux_wl[i] = (weights[i1:i2 + 1] * flux[i1:i2 + 1] * dldx[i1:i2 + 1]).sum()
                ucty_wl[i] = (weights[i1:i2 + 1] * ucty[i1:i2 + 1] * dldx[i1:i2 + 1]).sum()
            else:
                flux_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * flux[i1] * dldx[i1]
                ucty_wl[i] = (bin_edges_pix[i + 1] - bin_edges_pix[i]) * ucty[i1] * dldx[i1]
        flux_wl = (flux_wl * n) * spectrum.flux.unit
        ucty_wl = VarianceUncertainty(ucty_wl * n).represent_as(type(spectrum.uncertainty))
        return Spectrum1D(flux_wl, bin_centers_wav * u.angstrom, uncertainty=ucty_wl)

    def pix_to_wav(self, pix: ndarray |float) -> ndarray | float:
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
    def wcs(self):
        pixel_frame = cf.CoordinateFrame(1, "SPECTRAL", [0, ], axes_names=["x", ], unit=[u.pix])
        spectral_frame = cf.SpectralFrame(axes_names=["wavelength", ], unit=[self.unit])
        pipeline = [(pixel_frame, self.fitted_model), (spectral_frame, None)]
        self._wcs = wcs.WCS(pipeline)
        return self._wcs

    def match_lines(self, upper_bound: float = 5) -> tuple[ndarray, ndarray]:
        """Match the observed lines to theoretical lines.
s.
        Parameters
        ----------
        upper_bound
            The maximum allowed distance between the query points and the KD-tree
            data points for them to be considered a match.

        Returns
        -------
        A tuple containing two concatenated arrays:
        - An array of matched line positions in pixel coordinates.
        - An array of matched line positions in wavelength coordinates.
        """
        matched_lines_wav = []
        matched_lines_pix = []
        for iframe, tree in enumerate(self._trees):
            l, ix = tree.query(self._p2w(self.lines_pix[iframe])[:, None], distance_upper_bound=upper_bound)
            m = np.isfinite(l)
            matched_lines_wav.append(tree.data[ix[m], 0])
            matched_lines_pix.append(self.lines_pix[iframe][m])
        return np.concatenate(matched_lines_pix), np.concatenate(matched_lines_wav)

    def rms(self, space: Literal['pixel', 'wavelength'] = 'wavelength') -> float:
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
        if space == 'wavelength':
            return np.sqrt(((mwav - self.pix_to_wav(mpix)) ** 2).mean())
        else:
            return np.sqrt(((mpix - self.wav_to_pix(mwav)) ** 2).mean())

    def plot_lines(self, axes: Axes | None = None, figsize: tuple[float, float] | None = None) -> Figure:
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
        if axes is None:
            fig, axes = subplots(self.ndata, 1, figsize=figsize, sharex='all',
                                 constrained_layout=True, squeeze=False)
        else:
            fig = axes[0].figure
        for i, sp in enumerate(self.arc_spectra):
            axes[i,0].plot(sp.data / (1.2 * sp.data.max()))
            axes[i,0].vlines(self.lines_pix[i], 0.0, 1, alpha=0.1)
            axes[i,0].vlines(self.lines_pix[i], 0.9, 1)
            axes[i,0].autoscale(enable=True, axis='x', tight=True)
        setp(axes[-1], xlabel='Pixel')
        return fig

    def plot_solution(self, axes: Axes | None = None, figsize: tuple[float, float] | None = None,
                      model: Callable | None = None) -> Figure:
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
            fig, axes = subplots(len(self.arc_spectra), 1, figsize=figsize, sharex='all',
                                 constrained_layout=True, squeeze=False)
        else:
            fig = axes[0].figure

        model = model if model is not None else self._p2w

        for i, sp in enumerate(self.arc_spectra):
            axes[i,0].plot(model(sp.spectral_axis.value), sp.data / (1.2 * sp.data.max()))
            axes[i,0].vlines(self.lines_wav[i], 0.0, 1.0, alpha=0.3, ec='darkorange', zorder=0)
            axes[i,0].vlines(model(self.lines_pix[i]), 0.9, 1.0, alpha=1)
            axes[i,0].autoscale(enable=True, axis='x', tight=True)
        setp(axes[-1], xlabel=f'Wavelength [{self.unit.to_string(format="latex")}]')
        return fig

    def plot_transforms(self, figsize: tuple[float, float] | None = None) -> Figure:
        """ Plot and visualize transformation functions between pixel and wavelength space.

        This method generates a grid of subplots to illustrate the transformations
        between pixel and wavelength spaces in two directions: Pixel -> Wavelength
        and Wavelength -> Pixel. It also includes visualizations of the derivatives
        of these transformations with respect to the spectral axis.

        Parameters
        ----------
        figsize
            Width, height in inches to control the size of the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axs = subplots(2, 2, figsize=figsize, constrained_layout=True, sharex='col')
        xpix = self.arc_spectra[0].spectral_axis.value
        xwav = np.linspace(*self.pix_to_wav(xpix[[0, -1]]), num=xpix.size)
        axs[0, 0].plot(xpix, self._p2w(xpix), 'k')
        axs[1, 0].plot(xpix[:-1], np.diff(self._p2w(xpix)) / np.diff(xpix), lw=4, c='k')
        axs[1, 0].plot(xpix, self._p2w_dldx(xpix), ls='--', lw=2, c='w')
        axs[0, 1].plot(xwav, self._w2p(xwav), 'k')
        axs[1, 1].plot(xwav[:-1], np.diff(self._w2p(xwav)) / np.diff(xwav), lw=4, c='k')
        axs[1, 1].plot(xwav, self._w2p_dxdl(xwav), ls='--', lw=2, c='w')
        setp(axs[1,0], xlabel='Pixel', ylabel=r'd$\lambda$/dx')
        setp(axs[0,0], ylabel=fr'$\lambda$ [{self.unit}]')
        setp(axs[1,1], xlabel=fr'$\lambda$ [{self.unit}]', ylabel=r'dx/d$\lambda$')
        setp(axs[0,1], ylabel='Pixel')
        axs[0,0].set_title('Pixel -> wavelength')
        axs[0,1].set_title('Wavelength -> pixel')
        fig.align_labels()
        return fig

    def plot_residuals(self, ax: Axes | None = None, space: Literal['pixel', 'wavelength'] = 'wavelength',
                       figsize: tuple[float, float] | None = None) -> Figure:
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

        if space == 'wavelength':
            twav = self.pix_to_wav(mpix)
            ax.plot(mwav, mwav - twav, '.')
            ax.text(0.98, 0.95,
                    f"RMS = {np.sqrt(((mwav - twav) ** 2).mean()):4.2f} {self.unit.to_string(format="latex")}",
                    transform=ax.transAxes, ha='right', va='top')
            setp(ax,
                 xlabel=f'Wavelength [{self.unit.to_string(format="latex")}]',
                 ylabel=f'Residuals [{self.unit.to_string(format="latex")}]')
        else:
            tpix = self.wav_to_pix(mwav)
            ax.plot(mpix, mpix - tpix, '.')
            ax.text(0.98, 0.95, f"RMS = {np.sqrt(((mpix - tpix) ** 2).mean()):4.2f} pix", transform=ax.transAxes,
                    ha='right', va='top')
            setp(ax, xlabel='Pixel', ylabel='Residuals [pix]')
        ax.axhline(0, c='k', lw=1, ls='--')
        return fig
