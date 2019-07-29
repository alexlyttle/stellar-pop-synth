# Author:   Alex Lyttle
# Date:     27-07-2019

from pandas import DataFrame, read_csv

class TrackData:
    """Structure containing DataFrames from a MESA track history output file.

    Reads a history file assuming the following default structure:
    - line 0: header names
    - line 1: header data
    - line 2: blank (ignored by pandas.read_csv)
    - line 3: main data names
    - line 4: main data values
    but may be altered via class methods TrackData.set_header_rows and 
    TrackData.set_data_rows.

    Parameters
    ----------
    file_name : str, optional
        File name to be read in. The default is 'LOGS/history.data'. 

    Attributes
    ----------
    file_name : str
        Path to the file from which the data is read.
    data : pandas.DataFrame
        The main data from the MESA history track.
    header : pandas.DataFrame
        The header data from the MESA history track.
    ***ADD SECTION ON INITIAL ATTRIBUTES***
    """

    header_line = 1
    data_line = 4

    @classmethod
    def set_header_rows(cls, line=1):
        cls.header_line = line
    
    @classmethod
    def set_data_rows(cls, line=4):
        cls.data_line = line

    def __init__(self, file_name='LOGS/history.data'):
        self.file_name = file_name
        self.data = DataFrame()
        self.header = DataFrame()
        self.initial_mass = None  
        self.initial_z = None
        self.read_data()  # Reads data in the file 'self.file_name'

    def __repr__(self):
        # ***ADD INITIAL_MASS AND INITIAL_Z ATTRIBUTES***
        return 'TrackData(file_name={0}, data={1}, header={2})'.format(
            self.file_name, self.data, self.header)

    def __str__(self):
        return ('A member of TrackData with attributes,\n\n' +
                'file_name:\n"{}"\n\n'.format(self.file_name) +
                'data:\n{}\n\n'.format(self.data) +
                'header:\n{}'.format(self.header)
                )

    def read_data(self):
        """Loads or updates track data into a series of pandas DataFrame 
        objects.

        This reads the data from the file name provided. Lots of error checks
        needed here!

        """
        self.header = read_csv(self.file_name, delim_whitespace=True,
                               header=TrackData.header_line, nrows=1)
        self.data = read_csv(self.file_name, delim_whitespace=True, 
                             header=TrackData.data_line)
        self.initial_mass = self.header.initial_mass[0]
        self.initial_z = self.header.initial_z[0]
        
    def crop_rc(self, evo_state, center_he4_upper=0.95, center_he4_lower=1e-4,
                center_c12_lower=0.05):
        """
        Trims a given track history DataFrame subject to He core-burning conditions to produce a track in the red clump.
        The initial he4 is at 0.95 - Z, initial c12 is at 0.05 and track ends with he4 at 1e-4.
        The upper limit is, central He < (0.95 - Z).
        """
        if self.data.empty:
            # If data is empty, an empty DataFrame is returned
            cropped_data = DataFrame()
            print('Note, empty input DataFrame provided - nothing to crop.')
        else:
            condition = (self.data['center_he4'] < 
                         center_he4_upper - self.initial_z) &\
                        (self.data['center_he4'] > center_he4_lower) &\
                        (self.data['center_c12'] > center_c12_lower)
            cropped_data = self.data.loc[condition]
            cropped_data = cropped_data.reset_index()
            # Note - could have this apply the crop to itself? This may need
            # an additional function for changing self.data to a new DataFrame

        return cropped_data


if __name__ == '__main__':
    track = TrackData(file_name='test-data/m1.40.track')
    print(track)
