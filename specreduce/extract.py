# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import NDData, VarianceUncertainty
from numpy import ndarray
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline

from specutils import Spectrum
from specreduce.core import SpecreduceOperation, ImageLike, MaskingOption, parse_image
from specreduce.tracing import Trace

__all__ = ["BoxcarExtract", "HorneExtract", "OptimalExtract"]


def _get_boxcar_weights(center, hwidth, npix):
    """
    Compute weights given an aperture center, half width,
    and number of pixels.

    Based on `get_boxcar_weights()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    center : float, required
        The index of the aperture's center pixel on the larger image's
        cross-dispersion axis.

    hwidth : float, required
        Half of the aperture's width in the cross-dispersion direction.

    npix : float, required
        The number of pixels in the larger image's cross-dispersion
        axis.

    Returns
    -------
    weights : `~numpy.ndarray`
        A 2D image with weights assigned to pixels that fall within the
        defined aperture.
    """
    weights = np.zeros((npix))
    if hwidth == 0:
        # the logic below would return all zeros anyways, so might as well save the time
        # (negative widths should be avoided by earlier logic!)
        return weights

    if center - hwidth > npix - 0.5 or center + hwidth < -0.5:
        # entire window is out-of-bounds
        return weights

    lower_edge = max(-0.5, center - hwidth)  # where -0.5 is lower bound of the image
    upper_edge = min(center + hwidth, npix - 0.5)  # where npix-0.5 is upper bound of the image

    # let's avoid recomputing the round repeatedly
    int_round_lower_edge = int(round(lower_edge))
    int_round_upper_edge = int(round(upper_edge))

    # inner pixels that get full weight
    # the round in conjunction with the +1 handles the half-pixel "offset",
    # the upper bound doesn't have the +1 because array slicing is inclusive on the lower index and
    # exclusive on the upper-index
    # NOTE: round(-0.5) == 0, which is helpful here for the case where lower_edge == -0.5
    weights[int_round_lower_edge + 1 : int_round_upper_edge] = 1

    # handle edge pixels (for cases where an edge pixel is fully-weighted, this will set it again,
    # but should still compute a weight of 1.  By using N:N+1, we avoid index errors if the edge
    # is outside the image bounds.  But we do need to avoid negative indices which would count
    # from the end of the array.
    if int_round_lower_edge >= 0:
        weights[int_round_lower_edge : int_round_lower_edge + 1] = (
            round(lower_edge) + 0.5 - lower_edge
        )
    weights[int_round_upper_edge : int_round_upper_edge + 1] = upper_edge - (
        round(upper_edge) - 0.5
    )

    return weights


def _ap_weight_image(trace, width, disp_axis, crossdisp_axis, image_shape):
    """
    Create a weight image that defines the desired extraction aperture.

    Based on `ap_weight_images()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    trace : `~specreduce.tracing.Trace`, required
        trace object
    width : float, required
        width of extraction aperture in pixels
    disp_axis : int, required
        dispersion axis
    crossdisp_axis : int, required
        cross-dispersion axis
    image_shape : tuple with 2 elements, required
        size (shape) of image

    Returns
    -------
    wimage : `~numpy.ndarray`
        a 2D weight image defining the aperture
    """
    wimage = np.zeros(image_shape)
    hwidth = 0.5 * width
    image_sizes = image_shape[crossdisp_axis]

    # loop in dispersion direction and compute weights.
    for i in range(image_shape[disp_axis]):
        # TODO trace must handle transposed data (disp_axis == 0)
        # pass trace.trace.data[i] to avoid any mask if part of the regions is out-of-bounds

        # ArrayTrace can have nonfinite or masked data in trace, and this will fail,
        # so figure out how to handle that...

        wimage[:, i] = _get_boxcar_weights(trace.trace.data[i], hwidth, image_sizes)

    return wimage


@dataclass
class BoxcarExtract(SpecreduceOperation):
    """
    Standard boxcar extraction along a trace.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        extract = BoxcarExtract(image, trace)
        spectrum = extract(width=width)


    Parameters
    ----------
    image
        image with 2-D spectral image data
    trace_object
        trace object
    width
        width of extraction aperture in pixels
    disp_axis
        dispersion axis
    crossdisp_axis
        cross-dispersion axis
    mask_treatment
        Specifies how to handle masked or non-finite values in the input image.
        The accepted values are:

        - ``apply``: The image remains unchanged, and any existing mask is combined\
            with a mask derived from non-finite values.
        - ``ignore``: The image remains unchanged, and any existing mask is dropped.
        - ``propagate``: The image remains unchanged, and any masked or non-finite pixel\
            causes the mask to extend across the entire cross-dispersion axis.
        - ``zero_fill``: Pixels that are either masked or non-finite are replaced with 0.0,\
            and the mask is dropped.
        - ``nan_fill``:  Pixels that are either masked or non-finite are replaced with nan,\
            and the mask is dropped.
        - ``apply_mask_only``: The  image and mask are left unmodified.
        - ``apply_nan_only``: The  image is left unmodified, the old mask is dropped, and a\
            new mask is created based on non-finite values.

    Returns
    -------
    spec : `~specutils.Spectrum`
        The extracted 1d spectrum expressed in DN and pixel units
    """

    image: ImageLike
    trace_object: Trace
    width: float = 5
    disp_axis: int = 1
    crossdisp_axis: int = 0
    mask_treatment: MaskingOption = "apply"
    _valid_mask_treatment_methods = (
        "apply",
        "ignore",
        "propagate",
        "zero_fill",
        "nan_fill",
        "apply_mask_only",
        "apply_nan_only",
    )

    @property
    def spectrum(self):
        return self.__call__()

    def __call__(
        self,
        image: ImageLike | None = None,
        trace: Trace | None = None,
        width: float | None = None,
        disp_axis: int | None = None,
        crossdisp_axis: int | None = None,
    ) -> Spectrum:
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image
            The image with 2-D spectral image data
        trace
            The trace object
        width
            The width of extraction aperture in pixels
        disp_axis
            The dispersion axis
        crossdisp_axis
            The cross-dispersion axis

        Returns
        -------
        spec
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        image = image if image is not None else self.image
        trace = trace or self.trace_object
        width = width or self.width
        disp_axis = disp_axis or self.disp_axis
        cdisp_axis = crossdisp_axis or self.crossdisp_axis

        if width <= 0:
            raise ValueError("The window width must be positive")

        self.image = parse_image(
            image, disp_axis=disp_axis, mask_treatment=self.mask_treatment
        )

        # Spectrum extraction
        # ===================
        # Assign no weight to non-finite pixels outside the window. Non-finite pixels inside
        # the window will be propagated to the sum if mask treatment is either ``ignore`` or
        # ``propagate`` or excluded if the chosen mask treatment option is ``apply``. In the
        # latter case, the flux is calculated as the average of the non-masked pixels inside
        # the window multiplied by the window width.
        window_weights = _ap_weight_image(trace, width, disp_axis, cdisp_axis, self.image.shape)

        # Extract variance for uncertainty propagation (if available)
        if self.image.uncertainty is not None:
            variance = self.image.uncertainty.represent_as(VarianceUncertainty).array
            orig_uncty_type = type(self.image.uncertainty)
        else:
            variance = None
            orig_uncty_type = None

        extracted_variance = None
        if self.mask_treatment == "apply":
            flux = np.where(~self.image.mask, self.image.data * window_weights, 0.0)
            weights = np.where(~self.image.mask, window_weights, 0.0)
            weights_sum = weights.sum(axis=cdisp_axis)
            window_sum = window_weights.sum(axis=cdisp_axis)
            extracted_flux = flux.sum(axis=cdisp_axis) / weights_sum * window_sum

            if variance is not None:
                extracted_variance = (
                    np.where(~self.image.mask, variance * window_weights**2, 0.0).sum(
                        axis=cdisp_axis
                    )
                    / weights_sum**2
                    * window_sum**2
                )
        else:
            flux = np.where(window_weights, self.image.data * window_weights, 0.0)
            extracted_flux = flux.sum(axis=cdisp_axis)

            if variance is not None:
                variance = np.where(window_weights, variance * window_weights**2, 0.0)
                extracted_variance = np.sum(variance, axis=cdisp_axis)

        if extracted_variance is not None:
            spectrum_uncty = VarianceUncertainty(
                extracted_variance * self.image.unit**2
            ).represent_as(orig_uncty_type)
        else:
            spectrum_uncty = None

        return Spectrum(
            extracted_flux * self.image.unit,
            spectral_axis=self.image.spectral_axis,
            uncertainty=spectrum_uncty,
        )


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    There are two options for fitting the spatial profile used for
    extraction - by default, a 1D gaussian is fit and as a uniform profile
    across the spectrum. Alternativley, the ``self profile`` option may be
    chosen - when this option is chosen, the spatial profile will be sampled
    (using a default of 10 sample bins, but can be modified with
    ``spatial_profile``) and interpolated between to produce a smoothly varying
    spatial profile across the spectrum.

    If using the Gaussian option for the spatial profile, a background profile
    is fit simultaneously with the Gaussian so that a residual background does
    not distort the profile fit. By default, this is done with a 2nd degree
    polynomial. The background model only serves the profile fit: it is neither
    subtracted from the data nor included in the extraction kernel, so the
    input image is expected to be background-subtracted. The
    ``interpolated_profile`` option does not use a background model.

    The extraction kernel is evaluated directly on the image grid at every
    pixel's cross-dispersion offset from the trace, so curved traces are
    followed at sub-pixel precision without resampling the image (Horne 1986,
    Sect. II.A). The optional ``window`` restricts both the profile fit and the
    extraction to pixels within a given distance of the trace, which keeps
    other sources on the slit out of the profile fit.

    Following Horne (1986), the pixel variances used to weight the extraction
    are by default re-estimated from the extraction model rather than taken
    directly from the input. Weighting by a variance derived from the noisy
    data itself (e.g. the Poisson variance of the observed counts) biases the
    extracted flux low, by roughly the inverse of the counts per pixel. The
    re-estimation fits, column by column, a linear relation between the input
    variance and the model flux, which recovers the read-noise and gain terms
    without requiring them as inputs. Set ``model_variance=False`` to weight
    with the input variances as given.


    Parameters
    ----------

    image : `~astropy.nddata.NDData`-like or array-like, required
        The input 2D spectrum from which to extract a source. An
        NDData object must specify uncertainty and a mask. An array
        requires use of the ``variance``, ``mask``, & ``unit`` arguments.

    trace_object : `~specreduce.tracing.Trace`, required
        The associated 1D trace object created for the 2D image.

    disp_axis : int, optional
        The index of the image's dispersion axis. [default: 1]

    crossdisp_axis : int, optional
        The index of the image's cross-dispersion axis. [default: 0]

    bkgrd_prof : `~astropy.modeling.Model` or None, optional
        A model for the residual background fit together with the ``gaussian``
        spatial profile. ``None`` fits the Gaussian alone. The
        ``interpolated_profile`` option ignores this argument.
        [default: ``models.Polynomial1D(2)``]

    spatial_profile : str or dict, optional
        The shape of the object profile. The first option is 'gaussian' to fit
        a uniform 1D gaussian to the average of pixels in the cross-dispersion
        direction. The other option is 'interpolated_profile' - when this
        option is used, the profile is sampled in bins, and these samples are
        interpolated between to construct a continuously varying, empirical
        spatial profile for extraction. For this option, if passed in as a
        string (i.e., spatial_profile='interpolated_profile') the default values
        for the number of bins used (10) and degree of interpolation
        (linear in x and y, by default) will be used. To set these parameters,
        pass in a dictionary with the keys 'n_bins_interpolated_profile' (which
        accepts an integer number of bins) and 'interp_degree' (which accepts an
        int, or tuple of ints for x and y degree, respectively).
        [default: gaussian]

    variance : `~numpy.ndarray`, optional
        (Only used if ``image`` is not an NDData object.)
        The associated variances for each pixel in the image. Must
        have the same dimensions as ``image``. If all zeros, the variance
        will be ignored and treated as all ones.  If any zeros, those
        elements will be excluded via masking.  If any negative values,
        an error will be raised. [default: None]

    mask : `~numpy.ndarray`, optional
        (Only used if ``image`` is not an NDData object.)
        Whether to mask each pixel in the image. Must have the same
        dimensions as ``image``. If blank, all non-NaN pixels are
        unmasked. [default: None]

    unit : `~astropy.units.Unit` or str, optional
        (Only used if ``image`` is not an NDData object.)
        The associated unit for the data in ``image``. If blank,
        fluxes are interpreted in DN. [default: None]

    model_variance : bool, optional
        If True, re-estimate the pixel variances from the extraction model
        before the final extraction, as described above. If False, weight the
        extraction with the input variances as given. [default: True]

    window : float or None, optional
        Half-width, in pixels, of the region around the trace used for the
        profile fit and the extraction. Pixels further from the trace are
        treated as masked. ``None`` uses the full cross-dispersion extent.
        [default: None]

    """

    image: NDData
    trace_object: Trace
    bkgrd_prof: "Model | None" = field(default_factory=lambda: models.Polynomial1D(2))
    spatial_profile: str | dict = "gaussian"
    variance: np.ndarray = field(default=None)
    mask: np.ndarray = field(default=None)
    unit: np.ndarray = field(default=None)
    disp_axis: int = 1
    crossdisp_axis: int = 0
    model_variance: bool = True
    window: float | None = None
    # TODO: should disp_axis and crossdisp_axis be defined in the Trace object?

    @property
    def spectrum(self):
        return self.__call__()

    def _parse_image(self, image, variance=None, mask=None, unit=None, disp_axis=1):
        """
        Convert all accepted image types to a consistently formatted
        Spectrum object.

        HorneExtract needs its own version of this method because it is
        more stringent in its requirements for input images. The extra
        arguments are needed to handle cases where these parameters were
        specified as arguments and those where they came as attributes
        of the image object.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, required
            The image to be parsed. If None, defaults to class' own
            image attribute.
        variance : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as ``image``. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.
        mask : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as ``image``. If blank, all non-NaN pixels are
            unmasked.
        unit : `~astropy.units.Unit` or str, optional
            (Only used if ``image`` is not an NDData object.)
            The associated unit for the data in ``image``. If blank,
            fluxes are interpreted in DN.
        disp_axis : int, optional
            The index of the image's dispersion axis. Should not be
            changed until operations can handle variable image
            orientations. [default: 1]
        """

        if isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, u.quantity.Quantity):
            img = image.value
        else:  # NDData, including CCDData and Spectrum
            img = image.data

        # mask is set as None when not specified upon creating a Spectrum
        # object, so we must check whether it is absent *and* whether it's
        # present but set as None
        if getattr(image, "mask", None) is not None:
            mask = image.mask
        elif mask is not None:
            pass
        else:
            # if user provides no mask at all, don't mask anywhere
            mask = np.zeros_like(img)

        if img.shape != mask.shape:
            raise ValueError("image and mask shapes must match.")

        # Process uncertainties, converting to variances when able and throwing
        # an error when uncertainties are missing or less easily converted
        if hasattr(image, "uncertainty") and image.uncertainty is not None:
            if image.uncertainty.uncertainty_type == "var":
                variance = image.uncertainty.array
            elif image.uncertainty.uncertainty_type == "std":
                variance = image.uncertainty.array**2
            elif image.uncertainty.uncertainty_type == "ivar":
                variance = 1 / image.uncertainty.array
            else:
                # other options are InverseUncertainty and UnknownUncertainty
                raise ValueError(
                    "image NDData object has unexpected "
                    "uncertainty type. instead, try "
                    "VarianceUncertainty or StdDevUncertainty."
                )
        elif hasattr(image, "uncertainty") and image.uncertainty is None:
            # ignore variance arg to focus on updating NDData object
            raise ValueError("image NDData object lacks uncertainty")
        else:
            if variance is None:
                raise ValueError(
                    "if image is a numpy or Quantity array, a "
                    "variance must be specified. consider "
                    "wrapping it into one object by instead "
                    "passing an NDData image."
                )
            elif image.shape != variance.shape:
                raise ValueError("image and variance shapes must match")

        if np.any(variance < 0):
            raise ValueError("variance must be fully positive")
        if np.all(variance == 0):
            # technically would result in infinities, but since they're all
            # zeros, we can override ones to simulate an unweighted case
            variance = np.ones_like(variance)
        if np.any(variance == 0):
            # exclude such elements by editing the input mask
            mask[variance == 0] = True
            # replace the variances to avoid a divide by zero warning
            variance[variance == 0] = np.nan

        variance = VarianceUncertainty(variance)

        unit = getattr(image, "unit", u.Unit(unit) if unit is not None else u.Unit("DN"))

        spectral_axis = getattr(image, "spectral_axis", np.arange(img.shape[disp_axis]) * u.pix)

        return Spectrum(img * unit, spectral_axis=spectral_axis, uncertainty=variance, mask=mask)

    def _fit_gaussian_spatial_profile(
        self, img: ndarray, mask: ndarray, offsets: ndarray, bkgrd_prof: "Model | None"
    ):
        """Fit a 1D Gaussian spatial profile in trace-relative coordinates.

        The valid pixels of ``img`` are binned by their cross-dispersion offset
        from the trace in tenth-of-a-pixel bins and averaged, which co-adds the
        spectrum along the dispersion axis without resampling the image. Each
        bin is placed at the mean offset of the pixels it holds. A Gaussian
        centred on the trace (mean fixed at zero offset) and an optional
        background model are then fit to the binned profile. Returns the fitted
        compound model.
        """
        bin_width = 0.1
        valid = ~mask
        u = offsets[valid]
        u0 = np.floor(u.min())
        idx = ((u - u0) / bin_width).astype(int)
        counts = np.bincount(idx)
        populated = counts > 0
        centres = np.bincount(idx, weights=u)[populated] / counts[populated]
        coadd = np.bincount(idx, weights=img[valid])[populated] / counts[populated]

        gauss_prof = models.Gaussian1D(
            amplitude=coadd.max(), mean=0.0, stddev=2, fixed={"mean": True}
        )
        if bkgrd_prof is not None:
            ext_prof = gauss_prof + bkgrd_prof
        else:
            # add a trivial constant model so attribute names are the same
            ext_prof = gauss_prof + models.Const1D(0, fixed={"amplitude": True})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = fitting.LMLSQFitter()
            return fitter(ext_prof, centres, coadd)

    def _fit_spatial_profile(
        self, img: ndarray, mask: ndarray, offsets: ndarray, n_bins: int, kx: int, ky: int
    ) -> "_InterpolatedProfile":
        """
        Fit an empirical spatial profile by sampling the median profile along the
        dispersion direction.

        The image is split into ``n_bins`` bins along the dispersion axis. In each
        bin the valid pixels are grouped by their cross-dispersion offset from the
        trace, rounded to the nearest pixel, and the median of each group gives one
        sample of the spatial profile. The samples are normalized to unit sum and
        interpolated with a bivariate spline in dispersion pixel and offset, so the
        profile can be evaluated at any fractional offset from the trace.

        Parameters
        ----------
        img
            The 2D array of spectral data, with the cross-dispersion axis first.
        mask
            Boolean mask of the same shape; ``True`` marks pixels to ignore.
        offsets
            Cross-dispersion offset of every pixel from the trace.
        n_bins
            The number of bins along the dispersion axis.
        kx, ky
            Spline degrees along the dispersion axis and the offset axis.

        Returns
        -------
        _InterpolatedProfile
            Callable profile of dispersion pixel and trace-relative offset.
        """
        ncross, ndisp = img.shape
        if n_bins > ndisp:
            raise ValueError(
                f"n_bins_interpolated_profile ({n_bins}) exceeds the number of "
                f"dispersion pixels ({ndisp})."
            )
        img = np.where(~mask, img, np.nan)
        k = np.round(np.where(np.isfinite(offsets), offsets, 0.0)).astype(int)
        kmin, kmax = k[~mask].min(), k[~mask].max()
        grid = np.arange(kmin, kmax + 1)

        edges = np.linspace(0, ndisp, n_bins + 1).astype(int)
        bin_centres = (edges[:-1] + edges[1:]) // 2
        samples = np.zeros((n_bins, grid.size))
        for i in range(n_bins):
            cols = np.arange(edges[i], edges[i + 1])
            block, kb = img[:, cols], k[:, cols] - kmin
            ok = np.isfinite(block) & (kb >= 0) & (kb < grid.size)
            rows, cc = np.nonzero(ok)
            scatter = np.full((grid.size, cols.size), np.nan)
            scatter[kb[rows, cc], cc] = block[rows, cc]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(scatter, axis=1)
            median = np.where(np.isfinite(median), median, 0.0)
            samples[i] = median / median.sum()

        spline = RectBivariateSpline(x=bin_centres, y=grid, z=samples, kx=kx, ky=ky)
        return _InterpolatedProfile(spline, grid)

    def __call__(
        self,
        image=None,
        trace_object=None,
        disp_axis=None,
        crossdisp_axis=None,
        bkgrd_prof=None,
        spatial_profile=None,
        n_bins_interpolated_profile=None,
        interp_degree_interpolated_profile=None,
        variance=None,
        mask=None,
        unit=None,
        model_variance=None,
        window=None,
    ):
        """
        Run the Horne calculation on a region of an image and extract a 1D spectrum.

        Parameters
        ----------

        image : `~astropy.nddata.NDData`-like or array-like, required
            The input 2D spectrum from which to extract a source. An
            NDData object must specify uncertainty and a mask. An array
            requires use of the ``variance``, ``mask``, & ``unit`` arguments.

        trace_object : `~specreduce.tracing.Trace`, required
            The associated 1D trace object created for the 2D image.

        disp_axis : int, optional
            The index of the image's dispersion axis.

        crossdisp_axis : int, optional
            The index of the image's cross-dispersion axis.

        bkgrd_prof : `~astropy.modeling.Model`, optional
            A model for the residual background fit together with the ``gaussian``
            spatial profile. Overrides the model given at initialization; the
            ``interpolated_profile`` option ignores it.

        spatial_profile : str or dict, optional
            The shape of the object profile. The first option is 'gaussian' to fit
            a uniform 1D gaussian to the average of pixels in the cross-dispersion
            direction. The other option is 'interpolated_profile'  - when this
            option is used, the profile is sampled in bins and these samples are
            interpolated between to construct a continuously varying, empirical
            spatial profile for extraction. For this option, if passed in as a
            string (i.e spatial_profile='interpolated_profile') the default values
            for the number of bins used (10) and degree of interpolation
            (linear in x and y, by default) will be used. To set these parameters,
            pass in a dictionary with the keys 'n_bins_interpolated_profile' (which
            accepts an integer number of bins) and 'interp_degree' (which accepts an
            int, or tuple of ints for x and y degree, respectively).
            [default: gaussian]

        variance : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            The associated variances for each pixel in the image. Must
            have the same dimensions as ``image``. If all zeros, the variance
            will be ignored and treated as all ones.  If any zeros, those
            elements will be excluded via masking.  If any negative values,
            an error will be raised.

        mask : `~numpy.ndarray`, optional
            (Only used if ``image`` is not an NDData object.)
            Whether to mask each pixel in the image. Must have the same
            dimensions as ``image``. If blank, all non-NaN pixels are
            unmasked.

        unit : `~astropy.units.Unit` or str, optional
            (Only used if ``image`` is not an NDData object.)
            The associated unit for the data in ``image``. If blank,
            fluxes are interpreted in DN.

        model_variance : bool, optional
            Whether to re-estimate the pixel variances from the extraction model
            before the final extraction. Overrides the value given at initialization.

        window : float or None, optional
            Half-width, in pixels, of the region around the trace used for the
            profile fit and the extraction. Overrides the value given at
            initialization.


        Returns
        -------
        spec_1d : `~specutils.Spectrum`
            The final, Horne extracted 1D spectrum.
        """
        image = image if image is not None else self.image
        trace_object = trace_object if trace_object is not None else self.trace_object
        disp_axis = disp_axis if disp_axis is not None else self.disp_axis
        crossdisp_axis = crossdisp_axis if crossdisp_axis is not None else self.crossdisp_axis
        bkgrd_prof = bkgrd_prof if bkgrd_prof is not None else self.bkgrd_prof
        profile = spatial_profile if spatial_profile is not None else self.spatial_profile
        variance = variance if variance is not None else self.variance
        mask = mask if mask is not None else self.mask
        unit = unit if unit is not None else self.unit
        model_variance = model_variance if model_variance is not None else self.model_variance
        window = window if window is not None else self.window

        profile_choices = ("gaussian", "interpolated_profile")

        if not isinstance(profile, (str, dict)):
            raise ValueError("spatial_profile must be a string or dictionary.")
        if isinstance(profile, str):
            profile = dict(name=profile)

        profile_type = profile["name"].lower()
        if profile_type not in profile_choices:
            raise ValueError("spatial_profile must be one of" f"{', '.join(profile_choices)}")

        n_bins_interpolated_profile = profile.get("n_bins_interpolated_profile", 10)
        interp_degree_interpolated_profile = profile.get("interp_degree_interpolated_profile", 1)

        # Store original uncertainty type BEFORE parsing (parsing converts to VarianceUncertainty)
        if hasattr(image, "uncertainty") and image.uncertainty is not None:
            orig_uncty_type = type(image.uncertainty)
        else:
            orig_uncty_type = VarianceUncertainty  # default if variance passed separately

        self.image = self._parse_image(image, variance, mask, unit, disp_axis)

        variance = self.image.uncertainty.represent_as(VarianceUncertainty).array
        mask = self.image.mask.astype(bool) | (~np.isfinite(self.image.data))
        unit = self.image.unit
        flux = self.image.data

        # work with the cross-dispersion axis first
        if disp_axis == 0:
            flux, variance, mask = flux.T, variance.T, mask.T
        ncross, ndisp = flux.shape

        # cross-dispersion offset of every pixel from the trace; columns without a
        # finite trace position, and pixels outside the window, are masked
        trace = np.ma.filled(np.ma.asarray(trace_object.trace, dtype=float), np.nan)
        offsets = np.arange(ncross)[:, None] - trace[None, :]
        mask = mask | ~np.isfinite(offsets)
        if window is not None:
            mask = mask | ~(np.abs(offsets) <= window)
        if not np.any(~mask):
            raise ValueError("no valid pixels to extract from.")

        if profile_type == "gaussian":
            fit_ext_kernel = self._fit_gaussian_spatial_profile(flux, mask, offsets, bkgrd_prof)
            # The background component only stabilises the profile fit; the
            # extraction kernel is the Gaussian alone, evaluated at each pixel's
            # offset from the trace.
            amplitude = fit_ext_kernel.amplitude_0.value
            stddev = abs(fit_ext_kernel.stddev_0.value)
            with np.errstate(invalid="ignore"):
                kernel_vals = amplitude * np.exp(-0.5 * (offsets / stddev) ** 2)
            norms = np.full(ndisp, amplitude * stddev * np.sqrt(2 * np.pi))
        else:  # interpolated_profile
            # determine interpolation degree from input and make tuple if int
            # this can also be moved to another method to parse the input
            # 'spatial_profile' arg, eventually
            if isinstance(interp_degree_interpolated_profile, int):
                kx = ky = interp_degree_interpolated_profile
            else:  # if input is tuple of ints
                if not isinstance(interp_degree_interpolated_profile, tuple):
                    raise ValueError(
                        "``interp_degree_interpolated_profile`` must be ",
                        "an integer or tuple of integers.",
                    )
                if not all(isinstance(x, int) for x in interp_degree_interpolated_profile):
                    raise ValueError(
                        "``interp_degree_interpolated_profile`` must be ",
                        "an integer or tuple of integers.",
                    )
                kx, ky = interp_degree_interpolated_profile

            interp_spatial_prof = self._fit_spatial_profile(
                flux, mask, offsets, n_bins_interpolated_profile, kx, ky
            )
            # add private attribute to save fit profile. should this be public?
            self._interp_spatial_prof = interp_spatial_prof

            disp_pix = np.broadcast_to(np.arange(ndisp), offsets.shape)
            kernel_vals = interp_spatial_prof.ev(disp_pix, offsets)
            # normalization of the profile over its sampled offset range
            norms = trapezoid(
                interp_spatial_prof(np.arange(ndisp), interp_spatial_prof.grid), dx=1, axis=1
            )

        kernel_vals = np.where(np.isfinite(kernel_vals), kernel_vals, 0.0)
        valid = ~mask
        crossdisp_axis = 0

        extracted_flux, extracted_variance = _horne_sum(
            flux, kernel_vals, variance, valid, norms, crossdisp_axis
        )
        if model_variance:
            variance = _model_variance(
                kernel_vals, norms, extracted_flux, variance, valid, crossdisp_axis
            )
            extracted_flux, extracted_variance = _horne_sum(
                flux, kernel_vals, variance, valid, norms, crossdisp_axis
            )

        spectrum_uncty = VarianceUncertainty(
            extracted_variance * self.image.unit**2
        ).represent_as(orig_uncty_type)

        return Spectrum(
            extracted_flux * unit,
            spectral_axis=self.image.spectral_axis,
            uncertainty=spectrum_uncty,
        )


class _InterpolatedProfile:
    """
    Empirical spatial profile as a function of dispersion pixel and offset from the trace.

    Wraps a `~scipy.interpolate.RectBivariateSpline` sampled on ``grid``, the integer
    offsets covered by the image, and evaluates to zero outside that range instead
    of extrapolating.
    """

    def __init__(self, spline: RectBivariateSpline, grid: ndarray):
        self.spline = spline
        self.grid = grid

    def _inside(self, offsets):
        return (offsets >= self.grid[0]) & (offsets <= self.grid[-1])

    def __call__(self, disp, offsets):
        """Evaluate on the grid spanned by ``disp`` and ``offsets``."""
        offsets = np.asarray(offsets)
        return self.spline(disp, offsets) * self._inside(offsets)[None, :]

    def ev(self, disp, offsets):
        """Evaluate at the pointwise coordinates ``disp`` and ``offsets``."""
        offsets = np.asarray(offsets)
        inside = self._inside(offsets)
        values = self.spline.ev(disp, np.where(inside, offsets, self.grid[0]))
        return np.where(inside, values, 0.0)


def _horne_sum(flux, kernel_vals, variance, valid, norms, crossdisp_axis):
    """
    Evaluate the Horne (1986) weighted sums for every dispersion element.

    Returns the extracted flux ``norms * sum(f P / V) / sum(P**2 / V)`` and its
    variance ``norms**2 / sum(P**2 / V)``, with the sums restricted to ``valid``
    pixels.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sum(np.where(valid, flux * kernel_vals / variance, 0.0), axis=crossdisp_axis)
        den = np.sum(np.where(valid, kernel_vals**2 / variance, 0.0), axis=crossdisp_axis)
        return (num / den) * norms, norms**2 / den


def _model_variance(kernel_vals, norms, extracted_flux, variance, valid, crossdisp_axis):
    """
    Re-estimate the pixel variances from the extraction model (Horne 1986).

    Weighting the extraction by a variance derived from the noisy data itself
    biases the extracted flux low, because pixels that fluctuate upwards get
    less weight than pixels that fluctuate downwards. Horne's remedy is to
    compute the variance from the noise-free model instead. Without knowing
    the gain and read noise, the same is achieved by fitting, for each
    dispersion element, the linear relation ``V = intercept + slope * model``
    between the input variances and the model flux ``extracted_flux * P``
    over the valid pixels. The slope recovers the inverse gain and the
    intercept the read-noise and background terms. Model variances are floored
    at the smallest valid input variance of the element, and elements without
    a finite extracted flux keep their input variances.
    """
    expand = lambda a: np.expand_dims(a, crossdisp_axis)  # noqa: E731

    with np.errstate(invalid="ignore"):
        model = kernel_vals * expand(extracted_flux / norms)
    m = np.where(valid, model, 0.0)
    v = np.where(valid, variance, 0.0)
    n = np.sum(valid, axis=crossdisp_axis)
    sx, sy = m.sum(axis=crossdisp_axis), v.sum(axis=crossdisp_axis)
    sxx, sxy = (m * m).sum(axis=crossdisp_axis), (m * v).sum(axis=crossdisp_axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        det = n * sxx - sx * sx
        slope = np.where(det > 0, (n * sxy - sx * sy) / det, 0.0)
        slope = np.clip(slope, 0.0, None)
        intercept = np.where(n > 0, (sy - slope * sx) / n, np.nan)
        floor = np.min(np.where(valid, variance, np.inf), axis=crossdisp_axis)
        modelled = np.maximum(expand(intercept) + expand(slope) * model, expand(floor))

    usable = np.isfinite(extracted_flux) & (n > 0)
    return np.where(expand(usable), modelled, variance)


def _align_along_trace(img, trace_array, disp_axis=1, crossdisp_axis=0):
    """
    Given an arbitrary trace ``trace_array`` (an np.ndarray), roll
    all columns of ``nddata`` to shift the NDData's pixels nearest
    to the trace to the center of the spatial dimension of the
    NDData.
    """
    # TODO: this workflow does not support extraction for >2D spectra
    if not (disp_axis == 1 and crossdisp_axis == 0):
        # take the transpose to ensure the rows are the cross-disp axis:
        img = img.T

    n_rows, n_cols = img.shape

    # indices of all columns, in their original order
    rows = np.broadcast_to(np.arange(n_rows)[:, None], img.shape)
    cols = np.broadcast_to(np.arange(n_cols), img.shape)

    # we want to "roll" each column so that the trace sits in
    # the central row of the final image
    shifts = trace_array.astype(int) - n_rows // 2

    # we wrap the indices so we don't index out of bounds
    shifted_rows = np.mod(rows + shifts[None, :], n_rows)

    return img[shifted_rows, cols]


@dataclass
class OptimalExtract(HorneExtract):
    """
    An alias for `HorneExtract`.
    """

    __doc__ += HorneExtract.__doc__
    pass
