import pandas as pd
import os

nist_data_directory = 'specreduce/data/line_lists/NIST'
file_basename = 'nist_air_strong_lines_{:s}.csv'

def get_nist_string_lines(lamp_name):
    assert len(lamp_name) % 2 == 0
    data_frame_list = []
    for element in [lamp_name[i:i+2] for i in range(0,
                                                    len(lamp_name),
                                                    2)]:
        file_name = os.path.join(nist_data_directory,
                                 file_basename.format(element.title()))
        print(file_name)
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name, names=['intensity',
                                               'vacuum_wavelength',
                                               'spectrum',
                                               'reference'])
            data_frame_list.append(df)
        else:
            raise Exception('File not found')
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
    print(os.getcwd())
    df = get_nist_string_lines('HgArNe')
    print(df)

