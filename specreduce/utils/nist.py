#!/usr/bin/env python 
# -*- coding: utf8 -*-
"""
    File which contains a method that can be used to retrieve data from
    the National Institute of Standards and Technology (NIST) Atomic Spectra
    Database.
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup, Tag


__author__ = 'Bruno Quint'


def get_nist_data(element, blue_limit, red_limit):
    """
    Retrieve data from NIST using a GET request and parsing the results into
    two NumPy arrays: one for the wavelengths and other for the intensities.

    Parameters
    ----------
        element : str
            A string that matches a element (e.g.: "Th", "Cu", "Na").

        blue_limit : float
            Blue wavelength in Angstrom.

        red_limit : float
            Red wavelength in Angstrom.

    Returns
    -------
        wavelengths : numpy.ndarray
            A 1D array with the wavelengths.

        intensities : numpy.ndarray
            A 1D array with the corresponding intensities.
    """

    URL = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl'
    payload = {
        'spectra': element,
        'limits_type': 0,
        'low_w': blue_limit,
        'upp_w': red_limit,
        'unit': 0,
        'submit': 'Retrieve Data',
        'de': 0,
        'format': 0,
        'line_out': 0,
        'en_unit': 0,
        'output': 0,
        'bibrefs': 1,
        'page_size': 15,
        'show_obs_wl': 1,
        'show_calc_wl': 1,
        'unc_out': 1,
        'order_out': 0,
        'max_low_enrg': '',
        'show_av': 2,
        'max_upp_enrg': '',
        'tsb_value': 0,
        'min_str': '',
        'A_out': 0,
        'intens_out': 'on',
        'max_str': '',
        'allowed_out': 1,
        'forbid_out': 1,
        'min_accur': '',
        'min_intens': '',
        'conf_out': 'on',
        'term_out': 'on',
        'enrg_out': 'on',
        'J_out': 'on',
    }

    r = requests.get(URL, params=payload)
    soup = BeautifulSoup(r.content, 'lxml')

    _ = [s.extract() for s in soup.find_all('table', attrs={'width': ['75%', '100%']})]
    _ = [s.extract() for s in soup(['script', 'span'])]

    table = soup.find('table', attrs={'rules': 'groups'})
    header, body = table.find_all('tbody')
    header.name = 'theader'
    _ = [s.extract() for s in table('theader')]

    _df = pd.read_html(str(table))[0]
    _df = _df[[0, 1, 5]]
    _df.columns = ['Element', 'Wavelength', 'Intensity']

    _df['Wavelength'] = _df['Wavelength'].str.replace(" ", "")
    _df['Wavelength'] = _df['Wavelength'].astype(float)

    _df = _df[_df['Intensity'].apply(lambda x: str(x).isnumeric())]

    _df = _df.dropna()

    wavelengths = np.array(_df['Wavelength'], dtype=float)
    intensities = np.array(_df['Intensity'], dtype=float)

    return wavelengths, intensities
