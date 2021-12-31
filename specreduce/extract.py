# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.nddata import StdDevUncertainty

from specreduce.core import SpecreduceOperation
from specutils import Spectrum1D

__all__ = ['BoxcarExtract']


class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

    Parameters
    ----------
    img : nddata-compatible image
        The input image
    trace_object :
        The trace of the spectrum to be extracted TODO: define
    apwidth : int
        The width of the extraction aperture in pixels
    skysep : int
        The spacing between the aperture and the sky regions
    skywidth : int
        The width of the sky regions in pixels
    skydeg : int
        The degree of the polynomial that's fit to the sky

    Returns
    -------
    spec : `~specutils.Spectrum1D`
        The extracted spectrum
    skyspec : `~specutils.Spectrum1D`
        The sky spectrum used in the extraction process
    """
    apwidth: int = 8
    skysep: int = 3
    skywidth = 7
    skydeg: int = 0

    def __call__(self, img, trace_object):
        self.last_trace = trace_object
        self.last_img = img

        if self.apwidth < 1:
            raise ValueError('apwidth must be >= 1')
        if self.skysep < 1:
            raise ValueError('skysep must be >= 1')
        if self.skywidth < 1:
            raise ValueError('skywidth must be >= 1')

        trace_line = trace_object.trace

        onedspec = np.zeros_like(trace_line)
        skysubflux = np.zeros_like(trace_line)
        fluxerr = np.zeros_like(trace_line)

        for i in range(0, len(trace_line)):
            # first do the aperture flux
            # juuuust in case the trace gets too close to an edge
            widthup = self.apwidth / 2.
            widthdn = self.apwidth / 2.
            if (trace_line[i] + widthup > img.shape[0]):
                widthup = img.shape[0] - trace_line[i] - 1.
            if (trace_line[i] - widthdn < 0):
                widthdn = trace_line[i] - 1.

            # extract from box around the trace line
            low_end = trace_line[i] - widthdn
            high_end = trace_line[i] + widthdn

            self._extract_from_box(img, i, low_end, high_end, onedspec)

            # now do the sky fit
            # Note that we are not including fractional pixels, since we are doing
            # a polynomial fit over the sky values.
            j1 = self._find_nearest_int(trace_line[i] - self.apwidth/2. -
                                        self.skysep - self.skywidth)
            j2 = self._find_nearest_int(trace_line[i] - self.apwidth/2. - self.skysep)
            sky_y_1 = np.arange(j1, j2)

            j1 = self._find_nearest_int(trace_line[i] + self.apwidth/2. + self.skysep)
            j2 = self._find_nearest_int(trace_line[i] + self.apwidth/2. +
                                        self.skysep + self.skywidth)
            sky_y_2 = np.arange(j1, j2)

            sky_y = np.append(sky_y_1, sky_y_2)

            # sky can't be outside image
            np_indices = np.indices(img[::, i].shape)
            sky_y = np.intersect1d(sky_y, np_indices)

            sky_flux = img[sky_y, i]
            if (self.skydeg > 0):
                # fit a polynomial to the sky in this column
                pfit = np.polyfit(sky_y, sky_flux, self.skydeg)
                # define the aperture in this column
                ap = np.arange(
                    self._find_nearest_int(trace_line[i] - self.apwidth/2.),
                    self._find_nearest_int(trace_line[i] + self.apwidth/2.)
                )
                # evaluate the polynomial across the aperture, and sum
                skysubflux[i] = np.nansum(np.polyval(pfit, ap))
            elif (self.skydeg == 0):
                skysubflux[i] = np.nanmean(sky_flux) * self.apwidth

            # finally, compute the error in this pixel
            sigma_bkg = np.nanstd(sky_flux)  # stddev in the background data
            n_bkg = np.float(len(sky_y))  # number of bkgd pixels
            n_ap = self.apwidth  # number of aperture pixels

            # based on aperture phot err description by F. Masci, Caltech:
            # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
            fluxerr[i] = np.sqrt(
                np.nansum(onedspec[i] - skysubflux[i]) + (n_ap + n_ap**22 / n_bkg) * (sigma_bkg**2)
            )

        img_unit = u.DN
        if hasattr(img, 'unit'):
            img_unit = img.unit

        spec = Spectrum1D(
            spectral_axis=np.arange(len(onedspec)) * u.pixel,
            flux=onedspec * img_unit,
            uncertainty=StdDevUncertainty(fluxerr)
        )
        skyspec = Spectrum1D(
            spectral_axis=np.arange(len(onedspec)) * u.pixel,
            flux=skysubflux * img_unit
        )

        return spec, skyspec

    def _extract_from_box(self, image, wave_index, low_end, high_end, extracted_result):

        # compute nearest integer endpoints defining an internal interval,
        # and fractional pixel areas that remain outside this interval.
        # (taken from the HST STIS pipeline code:
        # https://github.com/spacetelescope/hstcal/blob/master/pkg/stis/calstis/cs6/x1dspec.c)
        #
        # This assumes that the pixel coordinates represent the center of the pixel.
        # E.g. pixel at y=15.0 covers the image from y=14.5 to y=15.5

        # nearest integer endpoints
        j1 = self._find_nearest_int(low_end)
        j2 = self._find_nearest_int(high_end)

        # fractional pixel areas at the end points
        s1 = 0.5 - (low_end - j1)
        s2 = 0.5 + high_end - j2

        # add up the total flux around the trace_line
        extracted_result[wave_index] = np.nansum(image[j1 + 1:j2, wave_index])
        extracted_result[wave_index] += np.nansum(image[j1, wave_index]) * s1
        extracted_result[wave_index] += np.nansum(image[j2, wave_index]) * s2

    def _find_nearest_int(self, end_point):
        if (end_point % 1) < 0.5:
            return int(end_point)
        else:
            return int(end_point + 1)

    def get_checkplot(self):
        trace_line = self.last_trace.line

        fig = plt.figure()
        plt.imshow(self.last_img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(self.last_img, (5, 98)))

        plt.plot(np.arange(len(trace_line)), trace_line, c='C0')
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line + self.apwidth,
            trace_line - self.apwidth,
            color='C0',
            alpha=0.5
        )
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line + self.apwidth + self.skysep,
            trace_line + self.apwidth + self.skysep + self.skywidth,
            color='C1',
            alpha=0.5
        )
        plt.fill_between(
            np.arange(len(trace_line)),
            trace_line - self.apwidth - self.skysep,
            trace_line - self.apwidth - self.skysep - self.skywidth,
            color='C1',
            alpha=0.5
        )
        plt.ylim(
            np.min(
                trace_line - (self.apwidth + self.skysep + self.skywidth) * 2
            ),
            np.max(
                trace_line + (self.apwidth + self.skysep + self.skywidth) * 2
            )
        )

        return fig
