# Licensed under a 3-clause BSD style license - see LICENSE.rst

from dataclasses import dataclass, field, InitVar

import numpy as np
from astropy import units as u
from astropy.modeling import Model, models, fitting
from astropy.nddata import VarianceUncertainty
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline
from specutils import Spectrum1D

from specreduce.core import SpecreduceOperation
from specreduce.image import SRImage, as_image, DISP_AXIS, CROSSDISP_AXIS
from specreduce.tracing import Trace, FlatTrace

__all__ = ['BoxcarExtract', 'HorneExtract', 'OptimalExtract']


def _get_boxcar_weights(center: float, hwidth: float, npix: int):
    """
    Compute weights given an aperture center, half width,
    and number of pixels.

    Based on `get_boxcar_weights()` from a JDAT Notebook by Karl Gordon:
    https://github.com/spacetelescope/jdat_notebooks/blob/main/notebooks/MIRI_LRS_spectral_extraction/miri_lrs_spectral_extraction.ipynb

    Parameters
    ----------
    center
        The index of the aperture's center pixel on the larger image's
        cross-dispersion axis.

    hwidth
        Half of the aperture's width in the cross-dispersion direction.

    npix
        The number of pixels in the larger image's cross-dispersion
        axis.

    Returns
    -------
    weights : `~numpy.ndarray`
        A 2D image with weights assigned to pixels that fall within the
        defined aperture.
    """
    weights = np.zeros(npix)
    if hwidth == 0:
        # the logic below would return all zeros anyways, so might as well save the time
        # (negative widths should be avoided by earlier logic!)
        return weights

    if center-hwidth > npix-0.5 or center+hwidth < -0.5:
        # entire window is out-of-bounds
        return weights

    lower_edge = max(-0.5, center-hwidth)  # where -0.5 is lower bound of the image
    upper_edge = min(center+hwidth, npix-0.5)  # where npix-0.5 is upper bound of the image

    # let's avoid recomputing the round repeatedly
    int_round_lower_edge = int(round(lower_edge))
    int_round_upper_edge = int(round(upper_edge))

    # inner pixels that get full weight
    # the round in conjunction with the +1 handles the half-pixel "offset",
    # the upper bound doesn't have the +1 because array slicing is inclusive on the lower index and
    # exclusive on the upper-index
    # NOTE: round(-0.5) == 0, which is helpful here for the case where lower_edge == -0.5
    weights[int_round_lower_edge+1:int_round_upper_edge] = 1

    # handle edge pixels (for cases where an edge pixel is fully-weighted, this will set it again,
    # but should still compute a weight of 1.  By using N:N+1, we avoid index errors if the edge
    # is outside the image bounds.  But we do need to avoid negative indices which would count
    # from the end of the array.
    if int_round_lower_edge >= 0:
        weights[int_round_lower_edge:int_round_lower_edge+1] = round(lower_edge) + 0.5 - lower_edge
    weights[int_round_upper_edge:int_round_upper_edge+1] = upper_edge - (round(upper_edge) - 0.5)

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
    Does a standard boxcar extraction.

    Example: ::

        trace = FlatTrace(image, trace_pos)
        extract = BoxcarExtract(image, trace)
        spectrum = extract(width=width)


    Parameters
    ----------
    image : `~astropy.nddata.NDData`-like or array-like, required
        image with 2-D spectral image data
    trace_object : Trace, required
        trace object
    width : float, optional
        width of extraction aperture in pixels
    disp_axis : int, optional
        dispersion axis
    crossdisp_axis : int, optional
        cross-dispersion axis

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted 1d spectrum expressed in DN and pixel units
    """
    image: SRImage
    trace_object: Trace
    width: float = 5
    disp_axis: InitVar[int] = 1
    crossdisp_axis: InitVar[int | None] = None
    mask_treatment: str = 'filter'
    _valid_mask_treatment_methods = ('filter', 'omit', 'zero-fill')

    def __post_init__(self, disp_axis, crossdisp_axis):
        self.image = as_image(self.image,
                              disp_axis=disp_axis,
                              crossdisp_axis=crossdisp_axis)

    @property
    def spectrum(self):
        return self.__call__()

    def __call__(self, image=None, trace_object=None, width=None,
                 disp_axis=1, crossdisp_axis=None,
                 mask_treatment: str = 'filter') -> Spectrum1D:
        """
        Extract the 1D spectrum using the boxcar method.

        Parameters
        ----------
        image : `~astropy.nddata.NDData`-like or array-like, required
            image with 2-D spectral image data
        trace_object : Trace, required
            trace object
        width : float, optional
            width of extraction aperture in pixels [default: 5]
        disp_axis : int, optional
            dispersion axis [default: 1]
        crossdisp_axis : int, optional
            cross-dispersion axis [default: 0]
        mask_treatment : string, optional
            The method for handling masked or non-finite data. Choice of `filter`,
            `omit`, or `zero-fill`. If `filter` is chosen, masked/non-finite data
            will be filtered during the fit to each bin/column (along disp. axis) to
            find the peak. If `omit` is chosen, columns along disp_axis with any
            masked/non-finite data values will be fully masked (i.e, 2D mask is
            collapsed to 1D and applied). If `zero-fill` is chosen, masked/non-finite
            data will be replaced with 0.0 in the input image, and the mask will then
            be dropped. For all three options, the input mask (optional on input
            NDData object) will be combined with a mask generated from any non-finite
            values in the image data. Also note that because binning is an option in
            FitTrace, that masked data will contribute zero to the sum when binning
            adjacent columns.
            [default: ``filter``]


        Returns
        -------
        spec : `~specutils.Spectrum1D`
            The extracted 1d spectrum with flux expressed in the same
            units as the input image, or u.DN, and pixel units
        """
        trace_object = trace_object if trace_object is not None else self.trace_object
        width = width if width is not None else self.width

        # Parse image, including masked/nonfinite data handling based on
        # choice of `mask_treatment`, which for BoxcarExtract can be filter, zero-fill, or
        # omit. non-finite data will be masked, always. Returns a Spectrum1D.
        # self.image = self._parse_image(image, disp_axis=disp_axis,
        #                                mask_treatment=self.mask_treatment)
        # TODO: Mask treatment
        self.image = as_image(image if image is not None else self.image,
                              disp_axis, crossdisp_axis)

        # # _parse_image returns a Spectrum1D. convert this to a masked array
        # # for ease of calculations here (even if there is no masked data).
        # img = np.ma.masked_array(self.image.data, self.image.mask)

        if width <= 0:
            raise ValueError("width must be positive")

        # weight image to use for extraction
        wimg = _ap_weight_image(trace_object,
                                width,
                                self.image.disp_axis,
                                self.image.crossdisp_axis,
                                self.image.shape)

        # extract, assigning no weight to non-finite pixels outside the window
        # (non-finite pixels inside the window will still make it into the sum)
        image_windowed = np.where(wimg, self.image.data*wimg, 0)
        ext1d = np.sum(image_windowed, axis=self.image.crossdisp_axis)
        return Spectrum1D(ext1d * self.image.unit)


@dataclass
class HorneExtract(SpecreduceOperation):
    """
    Perform a Horne (a.k.a. optimal) extraction on a two-dimensional
    spectrum.

    There are two options for fitting the spatial profile used for
    extraction - by default, a 1D gaussian is fit and as a uniform profile
    across the spectrum. Alternativley, the ``self profile`` option may be
    chosen - when this option is chosen, the spatial profile will be sampled
    at various locations (set by <>) and interpolated between to produce a
    smoothly varying spatial profile across the spectrum.

    If using the Gaussian option for the spatial profile, a background profile
    may be fit (but not subtracted) simultaneously to the data. By default,
    this is done with a 2nd degree polynomial. If using the
    ``interpolated_profile`` option, the background model must be set to None.


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
        A model for the image's background flux. If ``spatial_profile`` is set
        to ``interpolated_profile``, then ``bkgrd_prof`` must be set to None.
        [default: models.Polynomial1D(2)].

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

    """
    image: SRImage
    trace_object: Trace
    bkgrd_prof: Model = field(default=models.Polynomial1D(2))
    spatial_profile: str | dict = 'gaussian'  # can actually be str, dict
    variance: InitVar[np.ndarray | None] = field(default=None)
    mask: InitVar[np.ndarray | None] = field(default=None)
    unit: InitVar[u.Unit | None] = field(default=None)
    disp_axis: InitVar[int] = field(default=1)
    crossdisp_axis: InitVar[int | None] = field(default=None)

    def __post_init__(self, variance, mask, unit, disp_axis, crossdisp_axis):
        self.image = SRImage(self.image,
                        disp_axis=disp_axis,
                        crossdisp_axis=crossdisp_axis,
                        unit=unit,
                        mask=mask,
                        uncertainty=variance,
                        uncertainty_type='var',
                        ensure_var_uncertainty=False,
                        require_uncertainty=False)


    @property
    def spectrum(self):
        return self.__call__()

    def _fit_gaussian_spatial_profile(self, img, disp_axis, crossdisp_axis,
                                      or_mask, bkgrd_prof):

        """
            Fits an 1D Gaussian profile to spectrum in `img`. Takes the mean
            of  ``img`` along the cross-dispersion axis (i.e, takes the mean of
            each row for a horizontal trace). Any columns with non-finite values
            are omitted from the fit. Background model (optional) is fit
            simultaneously. Returns an `astropy.model.Gaussian1D` (or compound
            model, if `bkgrd_prof` is supplied) fit to data.
        """

        # co-add signal in each image row
        nrows = img.shape[crossdisp_axis]
        ncols = img.shape[disp_axis]
        xd_pixels = np.arange(nrows)

        # for now, mask row with any non-finite value.
        row_mask = np.logical_or.reduce(or_mask, axis=disp_axis)
        coadd = np.ma.masked_array(np.sum(img, axis=disp_axis) / ncols,
                                   mask=row_mask)

        # use the sum of brightest row as an inital guess for Gaussian amplitude,
        # the the location of the brightest row as an initial guess for the mean
        gauss_prof = models.Gaussian1D(amplitude=coadd.max(),
                                       mean=coadd.argmax(), stddev=2)

        # Fit extraction kernel (Gaussian + background model) to coadded rows
        # with combined model (must exclude masked indices manually;
        # LevMarLSQFitter does not)
        if bkgrd_prof is not None:
            ext_prof = gauss_prof + bkgrd_prof
        else:
            # add a trivial constant model so attribute names are the same
            ext_prof = gauss_prof + models.Const1D(0, fixed={'amplitude': True})

        fitter = fitting.LevMarLSQFitter()
        fit_ext_kernel = fitter(ext_prof, xd_pixels[~row_mask], coadd[~row_mask])

        return fit_ext_kernel

    def _fit_self_spatial_profile(self, img, disp_axis, crossdisp_axis, or_mask,
                                  n_bins_interpolated_profile, kx, ky):

        """
            Fit a spatial profile to spectrum by sampling the median profile in
            bins (number of which set be `n_bins_interpolated_profile` along the
            dispersion direction, and interpolating between
            samples. Columns (assuming horizontal trace) with any non-finite
            values will be omitted from the fit. Returns an interpolator object
            (RectBivariateSpline) that can be evaluated at any x,y.
        """

        # boundaries of bins for sampling profile.
        sample_locs = np.linspace(0, img.shape[disp_axis]-1,
                                  n_bins_interpolated_profile+1, dtype=int)
        # centers of these bins, roughly
        bin_centers = [(sample_locs[i]+sample_locs[i+1]) // 2 for i in
                       range(len(sample_locs) - 1)]

        # for now, since fitting isn't enabled for arrays with any nans
        # mask out the columns with any non-finite values to make sure
        # these don't contribute to the fit profile
        col_mask = np.logical_or.reduce(or_mask, axis=crossdisp_axis)

        # make a full mask for the image based on which cols have nans
        img_col_mask = np.tile(col_mask, (img.shape[0], 1))

        # need to make a new masked array since this mask is different
        # omit all columns with nans from contributing to the fit
        new_masked_arr = np.ma.array(img.data.copy(), mask=img_col_mask)

        # sample at these locations, normalize to area so this just reflects the
        # shape of the spectrum any shifts in center location should be corrected
        # by _align_along_trace (this should be addressed later with a better way
        # to flatten the trace, because trace can be wiggly even if 'flat'...)
        samples = []
        for i in range(n_bins_interpolated_profile):
            slicee = new_masked_arr[:, sample_locs[i]:sample_locs[i+1]]
            bin_median = np.ma.median(slicee, axis=disp_axis)
            bin_median_sum = np.ma.sum(bin_median)
            samples.append(bin_median / bin_median_sum)

        interp_2d = RectBivariateSpline(x=bin_centers,
                                        y=np.arange(img.shape[crossdisp_axis]),
                                        z=samples, kx=kx, ky=ky)

        return interp_2d

    def __call__(self, image=None, trace_object=None,
                 disp_axis=None, crossdisp_axis=None,
                 bkgrd_prof=None, spatial_profile=None,
                 n_bins_interpolated_profile=None,
                 interp_degree_interpolated_profile=None,
                 variance=None, mask=None, unit=None):
        """
        Run the Horne calculation on a region of an image and extract a
        1D spectrum.

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
            A model for the image's background flux.

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


        Returns
        -------
        spec_1d : `~specutils.Spectrum1D`
            The final, Horne extracted 1D spectrum.
        """
        image = image if image is not None else self.image

        if variance is not None:
            uncertainty = VarianceUncertainty(variance)
        else:
            uncertainty = getattr(image, 'uncertainty', None)
            if uncertainty is None:
                raise ValueError('variance must be specified if the image data does '
                                 'not contain uncertainty information.')

        image = SRImage(image,
                        disp_axis=disp_axis or self.image.disp_axis,
                        crossdisp_axis=crossdisp_axis or self.image.crossdisp_axis,
                        unit=unit or self.image.unit,
                        mask=mask if mask is not None else self.image.mask,
                        uncertainty=uncertainty,
                        uncertainty_type='var',
                        ensure_var_uncertainty=True,
                        require_uncertainty=True)

        trace_object = trace_object if trace_object is not None else self.trace_object
        bkgrd_prof = bkgrd_prof if bkgrd_prof is not None else self.bkgrd_prof
        spatial_profile = (spatial_profile if spatial_profile is not None else
                           self.spatial_profile)

        # figure out what 'spatial_profile' was provided
        # put this parsing into another method at some point, its a lot..
        interp_degree_interpolated_profile = None
        n_bins_interpolated_profile = None
        spatial_profile_choices = ('gaussian', 'interpolated_profile')

        if isinstance(spatial_profile, str):
            spatial_profile = spatial_profile.lower()
            if spatial_profile not in spatial_profile_choices:
                raise ValueError("spatial_profile must be one of"
                                 f"{', '.join(spatial_profile_choices)}")
            if spatial_profile == 'interpolated_profile':  # use defaults
                bkgrd_prof = None
                n_bins_interpolated_profile = 10
                interp_degree_interpolated_profile = 1
        elif isinstance(spatial_profile, dict):
            # first, figure out what type of profile is indicated
            # right now, the only type that should use a dictionary is 'interpolated_profile'
            # but this may be extended in the future hence the 'name' key. also,
            # the gaussian option could be supplied as a single-key dict

            # will raise key error if not present, and also fail on .lower if not
            # a string - think this is informative enough
            spatial_profile_type = spatial_profile['name'].lower()

            if spatial_profile_type not in spatial_profile_choices:
                raise ValueError("spatial_profile must be one of"
                                 f"{', '.join(spatial_profile_choices)}")
            if spatial_profile_type == 'gaussian':
                spatial_profile = 'gaussian'
            else:
                if 'n_bins_interpolated_profile' in spatial_profile.keys():
                    n_bins_interpolated_profile = \
                        spatial_profile['n_bins_interpolated_profile']
                else:  # use default
                    n_bins_interpolated_profile = 10
                if 'interp_degree_interpolated_profile' in spatial_profile.keys():
                    interp_degree_interpolated_profile = \
                        spatial_profile['interp_degree_interpolated_profile']
                else:  # use default
                    interp_degree_interpolated_profile = 1
            spatial_profile = spatial_profile_type
            bkgrd_prof = None
        else:
            raise ValueError('``spatial_profile`` must either be string or dictionary.')

        img = image.to_masked_array()
        variance = image.uncertainty.array
        mask = img.mask
        unit = image.unit

        # If the trace is not flat, shift the rows in each column
        # so the image is aligned along the trace:
        if not isinstance(trace_object, FlatTrace):
            img = _align_along_trace(img, trace_object.trace)

        if self.spatial_profile == 'gaussian':
            # fit profile to average (mean) profile along crossdisp axis
            fit_ext_kernel = self._fit_gaussian_spatial_profile(img,
                                                                DISP_AXIS,
                                                                CROSSDISP_AXIS,
                                                                mask,
                                                                bkgrd_prof)

            # this is just creating an array of the trace to shift the mean
            # when iterating over each wavelength. this needs to be fixed in the
            # future to actually account for the trace shape in a non-flat trace
            # (or possibly omitted all togehter as it might be redundant if
            # _align_along_trace is correcting this already)
            if isinstance(trace_object, FlatTrace):
                mean_init_guess = trace_object.trace
            else:
                mean_init_guess = np.broadcast_to(
                    img.shape[CROSSDISP_AXIS] // 2, img.shape[DISP_AXIS]
                )

        else:  # interpolated_profile
            # for now, bkgrd_prof must be None because a compound model can't
            # be created with a interpolator + model. i think there is a way
            # around this, but will follow up later
            if bkgrd_prof is not None:
                raise ValueError('When `spatial_profile`is `interpolated_profile`,'
                                 '`bkgrd_prof` must be None. Background should'
                                 ' be fit and subtracted from `img` beforehand.')
            # make sure n_bins doesnt exceed the number of (for now) finite
            # columns. update this when masking is fixed.
            n_finite_cols = np.logical_or.reduce(mask, axis=crossdisp_axis)
            n_finite_cols = np.count_nonzero(n_finite_cols.astype(int) == 0)

            # determine interpolation degree from input and make tuple if int
            # this can also be moved to another method to parse the input
            # 'spatial_profile' arg, eventually
            if isinstance(interp_degree_interpolated_profile, int):
                kx = ky = interp_degree_interpolated_profile
            else:  # if input is tuple of ints

                if not isinstance(interp_degree_interpolated_profile, tuple):
                    raise ValueError("``interp_degree_interpolated_profile`` must be ",
                                     "an integer or tuple of integers.")
                if not all(isinstance(x, int) for x in interp_degree_interpolated_profile):
                    raise ValueError("``interp_degree_interpolated_profile`` must be ",
                                     "an integer or tuple of integers.")

                kx, ky = interp_degree_interpolated_profile

            if n_bins_interpolated_profile >= n_finite_cols:

                raise ValueError(f'`n_bins_interpolated_profile` ({n_bins_interpolated_profile}) '
                                 'must be less than the number of fully-finite '
                                 f'wavelength columns ({n_finite_cols}).')

            interp_spatial_prof = self._fit_self_spatial_profile(img,
                                                                 DISP_AXIS,
                                                                 CROSSDISP_AXIS,
                                                                 mask,
                                                                 n_bins_interpolated_profile,
                                                                 kx, ky)

            # add private attribute to save fit profile. should this be public?
            self._interp_spatial_prof = interp_spatial_prof

        col_mask = np.logical_or.reduce(mask, axis=CROSSDISP_AXIS)
        nonf_col = [np.nan] * img.shape[CROSSDISP_AXIS]

        # array of 'x' values for each wavelength for extraction
        nrows = img.shape[CROSSDISP_AXIS]
        xd_pixels = np.arange(nrows)

        kernel_vals = []
        norms = []
        for col_pix in range(img.shape[DISP_AXIS]):

            # for now, skip columns with any non-finite values
            # NOTE: fit and other kernel operations should support masking again
            # once a fix is in for renormalizing columns with non-finite values
            if col_mask[col_pix]:
                kernel_vals.append(nonf_col)
                norms.append(np.nan)
                continue

            if self.spatial_profile == 'gaussian':

                # set compound model's mean to column's matching trace value
                # again, this is probably not necessary
                if bkgrd_prof is not None:  # attr names will be diff. if not compound
                    fit_ext_kernel.mean_0 = mean_init_guess[col_pix]
                else:
                    fit_ext_kernel.mean = mean_init_guess[col_pix]

                # evaluate fit model (with shifted mean, based on trace)
                fitted_col = fit_ext_kernel(xd_pixels)

                # save result and normalization
                # this doesn't need to be in this loop, address later
                kernel_vals.append(fitted_col)

                norms.append(fit_ext_kernel.amplitude_0
                             * fit_ext_kernel.stddev_0 * np.sqrt(2*np.pi))

            else:  # interpolated_profile
                fitted_col = interp_spatial_prof(col_pix, xd_pixels)
                kernel_vals.append(fitted_col)
                norms.append(trapezoid(fitted_col, dx=1)[0])

        # transform fit-specific information
        kernel_vals = np.vstack(kernel_vals).T
        norms = np.array(norms)

        # calculate kernel normalization
        g_x = np.sum(kernel_vals**2 / variance, axis=CROSSDISP_AXIS)

        # sum by column weights
        weighted_img = np.divide(img * kernel_vals, variance)
        result = np.sum(weighted_img, axis=CROSSDISP_AXIS) / g_x

        # multiply kernel normalization into the extracted signal
        extraction = result * norms

        # convert the extraction to a Spectrum1D object
        return Spectrum1D(extraction * unit)


def _align_along_trace(img, trace_array):
    """
    Given an arbitrary trace ``trace_array`` (an np.ndarray), roll
    all columns of ``nddata`` to shift the NDData's pixels nearest
    to the trace to the center of the spatial dimension of the
    NDData.
    """
    # TODO: this workflow does not support extraction for >2D spectra
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
