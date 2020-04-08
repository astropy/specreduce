# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDPROC functions
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy import

from specreduce.core import SpecreduceOperation
from specutils import Spectrum1D


__all__ = ['extract']


class BoxcarExtract(SpecreduceOperation):
    """
    Does a standard boxcar extraction

    Parameters
    ----------
    img : nddata-compatible image
        The input image
    trace_object
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
    spec : `specutis.Spectrum1D`
        The extracted spectrum
    skyspec : `specutis.Spectrum1D`
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

        trace_line = trace_object.line

        onedspec = np.zeros_like(trace_line)
        skysubflux = np.zeros_like(trace_line)
        fluxerr = np.zeros_like(trace_line)

        for i in range(0,len(trace_line)):
            #-- first do the aperture flux
            # juuuust in case the trace gets too close to an edge
            widthup = self. apwidth
            widthdn = self. apwidth
            if (trace_line[i]+widthup > img.shape[0]):
                widthup = img.shape[0]-trace_line[i] - 1
            if (trace_line[i]-widthdn < 0):
                widthdn = trace_line[i] - 1

            # simply add up the total flux around the trace_line +/- width
            onedspec[i] = np.nansum(img[int(trace_line[i]-widthdn):int(trace_line[i]+widthup+1), i])

            #-- now do the sky fit
            itrace_line = int(trace_line[i])
            y = np.append(np.arange(itrace_line-self. apwidth-self.skysep-self.skywidth, itrace_line-self. apwidth-self.skysep),
                          np.arange(itrace_line+self. apwidth+self.skysep+1, itrace_line+self. apwidth+self.skysep+self.skywidth+1))

            z = img[y,i]
            if (self.skydeg>0):
                # fit a polynomial to the sky in this column
                pfit = np.polyfit(y,z,self.skydeg)
                # define the aperture in this column
                ap = np.arange(trace_line[i]-self. apwidth, trace_line[i]+self. apwidth+1)
                # evaluate the polynomial across the aperture, and sum
                skysubflux[i] = np.nansum(np.polyval(pfit, ap))
            elif (self.skydeg==0):
                skysubflux[i] = np.nanmean(z)*(self. apwidth*2.0 + 1)

            #-- finally, compute the error in this pixel
            sigB = np.nanstd(z) # stddev in the background data
            N_B = np.float(len(y)) # number of bkgd pixels
            N_A = self. apwidth * 2. + 1 # number of aperture pixels

            # based on aperture phot err description by F. Masci, Caltech:
            # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
            fluxerr[i] = np.sqrt(np.nansum((onedspec[i]-skysubflux[i])) +
                                 (N_A + N_A**2. / N_B) * (sigB**2.))

        spec = Spectrum1D(spectral_axis=np.arange(len(onedspec))*u.pixel,
                          flux=onedspec * img.unit,
                          uncertainty=StdDevUncertainty(fluxerr))
        skyspec = Spectrum1D(spectral_axis=np.arange(len(onedspec))*u.pixel,
                             flux=skysubflux * img.unit)

        return spec, skyspec

    def get_checkplot(self):
        trace_line = self.last_trace.line

        fig = plt.figure()
        plt.imshow(self.last_img, origin='lower', aspect='auto', cmap=plt.cm.Greys_r)
        plt.clim(np.percentile(self.last_img, (5, 98)))

        plt.plot(np.arange(len(trace_line)), trace_line, c='C0')
        plt.fill_between(np.arange(len(trace_line)), trace_line + self. apwidth, trace_line-self. apwidth, color='C0', alpha=0.5)
        plt.fill_between(np.arange(len(trace_line)), trace_line + self. apwidth + self.skysep, trace_line + self. apwidth + self.skysep + self.skywidth, color='C1', alpha=0.5)
        plt.fill_between(np.arange(len(trace_line)), trace_line - self. apwidth - self.skysep, trace_line - self. apwidth - self.skysep - self.skywidth, color='C1', alpha=0.5)
        plt.ylim(np.min(trace_line - (self. apwidth + self.skysep + self.skywidth)*2), np.max(trace_line + (self. apwidth + self.skysep + self.skywidth)*2))
        
        return fig
