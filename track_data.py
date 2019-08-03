# Author:   Alex Lyttle
# Date:     27-07-2019

from pandas import DataFrame, read_csv


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
            self.read_data()

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
        if attr in self.data.columns:
            return self.data[attr]
        elif attr in self.header:  # THIS IS NOT IDEAL: REVISE
            return self.header[attr]
        else:
            raise AttributeError(attr)

    def read_data(self):
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
    initial_mass : float
        The initial mass of the stellar track.
    initial_z : float
        The initial metallicity of the stellar track
    """

    # def __init__(self, file_name='LOGS/history.data', read_file=True,
    #              data=None, header=None, initial_mass=None, initial_z=None):

    #     MesaData.__init__(self, file_name=file_name, read_file=read_file,
    #                       data=data, header=header)
    #     self.set_initial_conditions(initial_mass=initial_mass,
    #                                 initial_z=initial_z)

    def __repr__(self):
        return ('TrackData(file_name={0}, data={1}, header={2}, '.format(
                self.file_name, self.data, self.header) +
                'initial_mass={0}, initial_z={1}'.format(
                self.initial_mass, self.initial_z))

    def __str__(self):
        return ('A member of TrackData with attributes,\n\n' +
                'file_name:\n"{}"\n\n'.format(self.file_name) +
                'data:\n{}\n\n'.format(self.data) +
                'header:\n{}'.format(self.header) +
                'initial_mass:\n{}'.format(self.initial_mass) +
                'initial_z:\n{}'.format(self.initial_z)
                )
    
    # def set_initial_conditions(self, initial_mass=None, initial_z=None):
    #     """Sets initial conditions of the stellar track to what is
    #     provided in the header data (default) unless
    #     provided explicitly in the keyword arguements.

    #     Attributes
    #     ----------
    #     initial_mass : float, optional
    #         Initial mass of the evolutionary track if not available in
    #         TrackData.header.
    #     initial_z : float, optional    
    #         Initial metallicity of the evolutionary track if not available in
    #         TrackData.header.
    #     """
    #     self.initial_mass = initial_mass if initial_mass else \
    #         self.header.initial_mass[0]
    #     self.initial_z = initial_z if initial_z else self.header.initial_z[0]

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



class MesaLogs:
    """Reads the MESA LOGS directory and creates instances of TrackData and 
    ProfileData where appropriate.
    """

    def __init__(self, logs_path='LOGS'):
        self.logs_path = logs_path


class MesaGrid:
    """Creates a grid of track data from a given directory path or specified
    list of file paths. Methods to access tracks with a given initial condition
    MesaGrid.search_grid() and sort the grid by mass/metallicity etc. will
    be added.
    """

    def __init__(self, grid_path=None):
        self.grid_path = grid_path