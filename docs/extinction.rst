Atmospheric Extinction
======================

Introduction
------------

The Earth's atmosphere is highly chromatic in its transmission of light. The wavelength-dependence
is dominated by scattering in the optical (320-700 nm) and molecular features in the near-infrared
and infrared.

Supported Optical Extinction Models
-----------------------------------

`specreduce` offers support for average optical extinction models for a set of observatories:

.. csv-table::
    :header:  "Model Name", "Observatory", "Lat", "Lon", "Elevation (m)", "Ref"

    "ctio", "Cerro Tololo Int'l Observatory", "-30.165", "70.815", "2215", "1"
    "kpno", "Kitt Peak Nat'l Observatory", "31.963", "111.6", "2120", "2"
    "lapalma", "Roque de Los Muchachos Observatory", "28.764", "17.895", "2396", "3"
    "mko", "Mauna Kea Int'l Observatory", "19.828", "155.48", "4160", "4"
    "mtham", "Lick Observatory, Mt. Hamilton Station", "37.341", "121.643", "1290", "5"
    "paranal", "European Southern Obs., Paranal Station", "-24.627", "70.405", "2635", "6"
    "apo", "Apache Point Observatory", "32.780", "105.82", "2788", "7"


1. The CTIO extinction curve was originally distributed with IRAF and comes from the work of
Stone & Baldwin (1983 MN 204, 347) plus Baldwin & Stone (1984 MN 206,
241).  The first of these papers lists the points from 3200-8370A while
the second extended the flux calibration from 6056 to 10870A but the
derived extinction curve was not given in the paper.  The IRAF table
follows SB83 out to 6436, the redder points presumably come from BS84
with averages used in the overlap region. More recent CTIO extinction
curves are shown as Figures in Hamuy et al (92, PASP 104, 533 ; 94 PASP
106, 566).

2. The KPNO extinction table was originally distributed with IRAF. The ultimate provenance of this data is unclear,
but it has been used as-is in this form for over 30 years.

3. Extinction table for Roque de Los Muchachos Observatory, La Palma.
Described in https://www.ing.iac.es/Astronomy/observing/manuals/ps/tech_notes/tn031.pdf.

4. Extinction table for Mauna Kea Observatory constructed from
https://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/extinction.
Newer data is available from https://www.aanda.org/articles/aa/pdf/2013/01/aa19834-12.pdf.

5. Extinction table for Lick Observatory on Mt. Hamilton constructed from
https://mthamilton.ucolick.org/techdocs/standards/lick_mean_extinct.html.

6. Extinction table for ESO-Paranal taken from http://www.eso.org/sci/facilities/eelt/science/drm/tech_data/data/atm_scat/paranal.dat.
See also https://www.aanda.org/articles/aa/pdf/2011/03/aa15537-10.pdf for updated version.

7. Extinction table for Apache Point Observatory. Based on the extinction table used for SDSS and
available at https://www.apo.nmsu.edu/arc35m/Instruments/DIS/ (https://www.apo.nmsu.edu/arc35m/Instruments/DIS/images/apoextinct.dat).
