from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib import parse
import sys
import pandas as pd


class NistCrawler(HTMLParser):

    def error(self, message):
        return message

    def __init__(self):
        HTMLParser.__init__(self)
        self._first_table_tag = False
        self._in_table = False
        self._new_row = False
        self._new_field = False
        self._row = []
        self._field = str('')
        # full list of elements commented at the end of this file.
        self.elements = {'Ar': 'Argon',
                         'Cu': 'Copper',
                         'He': 'Helium',
                         'Fe': 'Iron',
                         'Hg': 'Mercury',
                         'Ne': 'Neon',
                         'Th': 'Thorium',
                         'Xe': 'Xenon'}
        self.df = pd.DataFrame(columns=['intensity',
                                        'vacuum_wavelength',
                                        'spectrum',
                                        'reference'])

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and not self._first_table_tag:
            self._first_table_tag = True
        elif tag == 'table' and self._first_table_tag:
            self._in_table = True
        if tag == 'tr':
            self._new_row = True
            self._row = []
        if tag == 'td':
            self._new_field = True
            self._field = str('')

    def handle_endtag(self, tag):
        if tag == 'table' and self._in_table:
            self._in_table = False
            # if not self.df.empty:
            #     print(self.df)
        if tag == 'tr':
            self._new_row = False
            if self._row != []:
                self.df.loc[len(self.df)] = self._row
                self._row = []
        if tag == 'td':
            self._new_field = False
            if self._field != '' and len(self._row) < 4:
                try:
                    self._row.append(float(self._field))
                except ValueError as value_error:
                    self._row.append(self._field)
                self._field = str('')

    def handle_data(self, data):
        if self._in_table:
            data = data.replace(u'\xa0', '')
            data = data.replace('\r\n', '')
            if data != '\r\n':
                # print(type(data))
                if self._field == str(''):
                    self._field = data
                elif str(data) != ' ' and str(data) != '':
                    self._field = "{:s} {:s}".format(self._field, str(data))

    def get_data(self, url):
        try:
            response = urlopen(url)
            if 'text/html' in response.getheader('Content-Type'):
                htmlBytes = response.read()
                htmlString = htmlBytes.decode('utf-8')
                self.feed(htmlString)
                return self.df
        except HTTPError as error:
            pass

    def get_strong_lines(self, lamp_name):
            data_frame_list = []
            base_url = 'https://physics.nist.gov/PhysRefData/' \
                       'Handbook/Tables/{:s}table2.htm'
            for element in [lamp_name[i:i+2] for i in range(0,
                                                            len(lamp_name),
                                                            2)]:
                try:
                    name = self.elements[element.title()]
                    element_url = base_url.format(name.lower())
                    df = self.get_data(url=element_url)
                    data_frame_list.append(df)
                except KeyError as error:
                    print("Element {:s} unknown".format(error))

            if len(data_frame_list) > 1:
                linelist_df = pd.concat(data_frame_list)
                linelist_df = linelist_df.sort_values('vacuum_wavelength')
                linelist_df = linelist_df.reset_index(drop=True)
                return linelist_df
            elif len(data_frame_list) == 1:
                linelist_df = data_frame_list[0]
                return linelist_df
            else:
                raise Exception("No data was recovered")


if __name__ == '__main__':
    nist_crawler = NistCrawler()
    nist_data = nist_crawler.get_strong_lines('Ne')
    print(nist_data)

# Full elements list
# elements_dict = {'Ac': 'Actinium',
#                  'Al': 'Aluminum',
#                  'Am': 'Americium',
#                  'Sb': 'Antimony',
#                  'Ar': 'Argon',
#                  'As': 'Arsenic',
#                  'At': 'Astatine',
#                  'Ba': 'Barium',
#                  'Bk': 'Berkelium',
#                  'Be': 'Beryllium',
#                  'Bi': 'Bismuth',
#                  'B': 'Boron',
#                  'Br': 'Bromine',
#                  'Cd': 'Cadmium',
#                  'Ca': 'Calcium',
#                  'Cf': 'Californium',
#                  'C"': 'Carbon',
#                  'Ce': 'Cerium',
#                  'Cs': 'Cesium',
#                  'Cl': 'Chlorine',
#                  'Cr': 'Chromium',
#                  'Co': 'Cobalt',
#                  'Cu': 'Copper',
#                  'Cm': 'Curium',
#                  'Dy': 'Dysprosium',
#                  'Es': 'Einsteinium',
#                  'Er': 'Erbium',
#                  'Eu': 'Europium',
#                  'F': 'Fluorine',
#                  'Fr': 'Francium',
#                  'Gd': 'Gadolinium',
#                  'Ga': 'Gallium',
#                  'Ge': 'Germanium',
#                  'Au': 'Gold',
#                  'Hf': 'Hafnium',
#                  'He': 'Helium',
#                  'Ho': 'Holmium',
#                  'H': 'Hydrogen',
#                  'In': 'Indium',
#                  'I': 'Iodine',
#                  'Ir': 'Iridium',
#                  'Fe': 'Iron',
#                  'Kr': 'Krypton',
#                  'La': 'Lanthanum',
#                  'Pb': 'Lead',
#                  'Li': 'Lithium',
#                  'Lu': 'Lutetium',
#                  'Mg': 'Magnesium',
#                  'Mn': 'Manganese',
#                  'Hg': 'Mercury',
#                  'Mo': 'Molybdenum',
#                  'Nd': 'Neodymium',
#                  'Ne': 'Neon',
#                  'Np': 'Neptunium',
#                  'Ni': 'Nickel',
#                  'Nb': 'Niobium',
#                  'N': 'Nitrogen',
#                  'Os': 'Osmium',
#                  'O': 'Oxygen',
#                  'Pd': 'Palladium',
#                  'P': 'Phosphorus',
#                  'Pt': 'Platinum',
#                  'Pu': 'Plutonium',
#                  'Po': 'Polonium',
#                  'K': 'Potassium',
#                  'Pr': 'Praseodymium',
#                  'Pm': 'Promethium',
#                  'Pa': 'Protactinium',
#                  'Ra': 'Radium',
#                  'Rn': 'Radon',
#                  'Re': 'Rhenium',
#                  'Rh': 'Rhodium',
#                  'Rb': 'Rubidium',
#                  'Ru': 'Ruthenium',
#                  'Sm': 'Samarium',
#                  'Sc': 'Scandium',
#                  'Se': 'Selenium',
#                  'Si': 'Silicon',
#                  'Ag': 'Silver',
#                  'Na': 'Sodium',
#                  'Sr': 'Strontium',
#                  'S': 'Sulfur',
#                  'Ta': 'Tantalum',
#                  'Tc': 'Technetium',
#                  'Te': 'Tellurium',
#                  'Tb': 'Terbium',
#                  'TI': 'Thallium',
#                  'Th': 'Thorium',
#                  'Tm': 'Thulium',
#                  'Sn': 'Tin',
#                  'Ti': 'Titanium',
#                  'W': 'Tungsten',
#                  'U': 'Uranium',
#                  'V': 'Vanadium',
#                  'Xe': 'Xenon',
#                  'Yb': 'Ytterbium',
#                  'Y': 'Yttrium',
#                  'Zn': 'Zinc',
#                  'Zr': 'Zirconium'}
