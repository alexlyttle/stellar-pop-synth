# Author:   Alex Lyttle
# Date:     27-07-2019

from pandas import DataFrame, read_csv
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


class MesaData:
    """Structure containing DataFrames from a MESA output file.

    Reads an output file assuming the following default structure:
    - line 0: header names
    - line 1: header data
    - line 2: blank (ignored by pandas.read_csv)
    - line 3: main data names
    - line 4: main data values
    but may be altered via class methods MesaData.set_header_rows and 
    MesaData.set_data_rows.

    Attributes
    ----------
    file_name : str
        Path to the file from which the data is read.
    data : pandas.DataFrame
        The main data from the MESA history track.
    header : pandas.DataFrame
        The header data from the MESA history track.
    """

    header_line = 1
    data_line = 4

    @classmethod
    def set_header_rows(cls, line=1):
        cls.header_line = line
    
    @classmethod
    def set_data_rows(cls, line=4):
        cls.data_line = line
    
    def __init__(self, file_name='LOGS/history.data', read_file=True,
                 data=None, header=None):
        self.file_name = file_name
        self.data = data
        self.header = header
        if read_file:
            self.read_file()

    def __repr__(self):
        return 'MesaData(file_name={0}, data={1}, header={2})'.format(
            self.file_name, self.data, self.header)

    def __str__(self):
        return ('A member of MesaData with attributes,\n\n' +
                'file_name:\n"{}"\n\n'.format(self.file_name) +
                'data:\n{}\n\n'.format(self.data) +
                'header:\n{}'.format(self.header)
                )
    
    def __getattr__(self, attr):
        if isinstance(self.data, (dict, DataFrame)) and \
            attr in self.data.keys():
            return self.data[attr]
        if isinstance(self.header, (dict, DataFrame)) and \
            attr in self.header.keys():
            return self.header[attr]
        else:
            raise AttributeError(attr)

    def read_file(self):
        """Loads or updates track data into a series of pandas DataFrame 
        objects.

        This reads the data from the file name provided. Lots of error checks
        needed here!
        """
        # This is quite ugly but works for now.
        self.header = read_csv(self.file_name, delim_whitespace=True,
                               header=TrackData.header_line,
                               nrows=1).to_dict(orient='index')[0]
        self.data = read_csv(self.file_name, delim_whitespace=True, 
                             header=TrackData.data_line)


class TrackData(MesaData):
    """Structure containing DataFrames from a MESA track history output file.

    Attributes
    ----------
    file_name : str
        Path to the file from which the data is read.
    data : pandas.DataFrame
        The main data from the MESA history track.
    header : pandas.DataFrame
        The header data from the MESA history track.
    """

    def __repr__(self):
        return 'TrackData(file_name={0}, data={1}, header={2})'.format(
                self.file_name, self.data, self.header)

    def __str__(self):
        return ('A member of TrackData with attributes,\n\n' +
                'file_name:\n"{}"\n\n'.format(self.file_name) +
                'data:\n{}\n\n'.format(self.data) +
                'header:\n{}'.format(self.header))

    def crop_rc(self, center_he4_upper=0.95, center_he4_lower=1e-4,
                center_c12_lower=0.05):
        """Trims a given TrackData DataFrame subject to core helium-burning 
        conditions to produce a track in the red clump (or horizontal branch).
        
        The start of the core helium-burning phase is defined by either an
        upper limit on central helium - default is (0.95 - initial_z) - or
        a lower limit on central carbon - default is 0.05. These are chosen
        so as to bypass the helium flash.
        
        The end of the core helium-burning phase is defined by a lower limit on
        central helium - default is 1e-4.

        Attributes
        ----------
        center_he4_upper : float, optional
            The upper limit on the central helium fraction minus the
            initial metallicity.
        center_he4_lower : float, optional
            The lower limit on the central helium fraction.
        center_c12_lower : float, optional
            The lower limit on the central carbon fraction.
        """
        if self.data.empty:
            # If data is empty, the original track object is returned
            cropped_track = self
            print('Note, empty input data provided - nothing to crop.')
        else:
            condition = (self.data['center_he4'] < 
                         center_he4_upper - self.initial_z) &\
                        (self.data['center_he4'] > center_he4_lower) &\
                        (self.data['center_c12'] > center_c12_lower)
            cropped_data = self.data.loc[condition]
            cropped_data = cropped_data.reset_index(drop=True)
            cropped_track = self  # Copy self into new, cropped track.
            cropped_track.data = cropped_data
            # Isn't this better as a static method? Or just make the change
            # to the current object?
        
        return cropped_track

    def plot(self, x='effective_T', y='luminosity', invert_x=True):
        fig, ax = plt.subplots()
        ax.loglog(self.data[x], self.data[y])
        if invert_x:
            ax.invert_xaxis()
        plt.xlabel(x)
        plt.ylabel(y)
        fig.show()


class ProfileData(MesaData):
    """Structure containing DataFrames from a MESA profile output file. TBC...
    """
    def __repr__(self):
        return 'ProfileData(file_name={0}, data={1}, header={2})'.format(
            self.file_name, self.data, self.header)

    def __str__(self):
        return ('A member of ProfileData with attributes,\n\n' +
                'file_name:\n"{}"\n\n'.format(self.file_name) +
                'data:\n{}\n\n'.format(self.data) +
                'header:\n{}'.format(self.header)
                )


class ProfileIndex(MesaData):
    """Reads the MESA profile index file.
    """

    index_column_names = ['model_number', 'priority', 'profile_number']
    index_row_start = 1

    @classmethod
    def set_index_column_names(cls, names):
        cls.index_column_names = names

    @classmethod
    def set_index_row_start(cls, row):
        cls.index_row_start = row
    
    def read_file(self):
        self.header =  'test header - TBC'  # Soon will read header line
        self.data = read_csv(self.file_name, delim_whitespace=True,
                             names=ProfileIndex.index_column_names,
                             skiprows=ProfileIndex.index_row_start)


class MesaLog:
    """Reads the MESA LOGS directory and creates instances of TrackData and 
    ProfileData where appropriate.
    """

    def __init__(self, log_path='LOGS',
                 profile_index_file='profiles.index',
                 profile_prefix='profile', profile_suffix='data',
                 history_file='history.data', read_track=False,
                 read_all_profiles=False):
        self.log_path = log_path
        self.profile_index_file = '/'.join((self.log_path, profile_index_file))
        self.profile_prefix = '/'.join((self.log_path, profile_prefix))
        self.profile_suffix = profile_suffix
        self.history_file = '/'.join((self.log_path, history_file))

        self.track = None
        self.profile_index = None
        self.profile = dict()

        self.read_log_dir(read_track=read_track,
                          read_all_profiles=read_all_profiles)
    
    def read_log_dir(self, read_track=False, read_all_profiles=False):
        """Reads the LOGS directory.
        """
        self.track = TrackData(self.history_file, read_file=read_track)
        self.profile_index = ProfileIndex(self.profile_index_file)
        
        for i, m in enumerate(
            self.profile_index.data[ProfileIndex.index_column_names[0]]):
            # For each model number
            self.profile[str(m)] = ProfileData(
                file_name='{0}{1}.{2}'.format(self.profile_prefix, i+1,
                                               self.profile_suffix),
                read_file=read_all_profiles)
    
    def read_track(self):
        self.track.read_file()
    
    def read_profile(self, model_number):
        self.profile[str(model_number)].read_file()


class MesaGrid:
    """Creates a grid of track data from a given directory path or specified
    list of file paths. Methods to access tracks with a given initial condition
    MesaGrid.search_grid() and sort the grid by mass/metallicity etc. will
    be added. To be compatible with another element of this MESA companion
    which allows for the running of grids.
    """

    def __init__(self, grid_path=None):
        self.grid_path = grid_path