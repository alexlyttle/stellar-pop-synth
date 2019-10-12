# Author:   Alex Lyttle
# Date:     27-07-2019
# Description: Classes to interpret MESA output files with various useful
# methods for typical data usage. Initially, this is designed to make my
# Masters project work more useable and transferable. The long term goal is to
# develop a python package to be renamed "mesa-companion" to aid the use of and
# intepretation of MESA. Some elements of this have been borrowed from
# py_mesa_reader, written by by Bill Wolf.

import os, sys
import numpy as np
import xarray as xr
from pandas import DataFrame, read_csv
from matplotlib import pyplot as plt


class MesaData:
    """
    Structure containing data from a MESA output file.

    Reads an output file assuming the following default structure:
    - line 0: header names
    - line 1: header data
    - line 2: blank (ignored by pandas.read_csv)
    - line 3: main data names
    - line 4: main data values
    but may be altered via class methods MesaData.set_header_rows and 
    MesaData.set_data_rows.
    
    Parameters
    ----------
    file_name : str, optional
        Path to the file from which the MESA data is read (default is
        None).
    read_file : bool, optional
        Option to read the data in the history file (default is None).
    data : , optional
        The main data from the MESA history file (default is None).
    header : , optional
        The header data from the MESA history file (default is None).
    
    Attributes
    ----------
    file_name : str
        Where the MESA file is stored.
    data
        Where the main data from the MESA file is stored.
    header
        Where the header data from the MESA file is stored.

    Methods
    -------
    read_file()
        Reads the contents of the MESA output file associated with the object.
    """

    header_line = 1
    data_line = 4

    @classmethod
    def set_header_rows(cls, line=1):
        cls.header_line = line
    
    @classmethod
    def set_data_rows(cls, line=4):
        cls.data_line = line
    
    def __init__(self, file_name=None, read_file=None,
                 data=None, header=None):
        self.file_name = str(file_name)
        self.data = data
        self.header = header
        if read_file:
            try:
                self.read_file()
            except FileNotFoundError as fnf_err:
                print(fnf_err)
                print('File not found, read_file() not executed. ' +
                'Please set file_name attribute to a valid file name and ' +
                'run read_file() manually.')


    def __repr__(self):
        return 'MesaData(file_name={0}, header={1}, data={2})'.format(
            self.file_name, self.header, self.data)

    def __str__(self):
        return ('File name:\n"{}"\n\n'.format(self.file_name) +
                'Header:\n{}\n\n'.format(self.header) +
                'Data:\n{}'.format(self.data))
    
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
    """
    Structure containing DataFrames from a MESA track history output file.

    Parameters
    ----------
    file_name : str, optional
        Path to the file from which the MESA history data is read (default is
        'LOGS/history.data').
    read_file : bool, optional
        Option to read the data in the history file (default is True).
    data : , optional
        The main data from the MESA history file (default is None).
    header : , optional
        The header data from the MESA history file (default is None).
    
    Attributes
    ----------
    file_name : str
        Where the MESA history file is stored.
    data
        Where the main data from the MESA history file is stored.
    header
        Where the header data from the MESA history file is stored.
    
    Methods
    -------
    crop_rc(center_he4_upper=0.95, center_he4_lower=1e-4,
    center_c12_lower=0.05)
        Creates a new TrackData object cropped to the core helium-burning phase
        of the star's evolution.
    """

    def __init__(self, file_name='LOGS/history.data', read_file=True,
                 header=None, data=None):
        MesaData.__init__(self, file_name=file_name, read_file=read_file,
                          header=header, data=data)

    def __repr__(self):
        return 'TrackData(file_name={0}, header={1}, data={2})'.format(
            self.file_name, self.header, self.data)

    def crop_rc(self, center_he4_upper=0.95, center_he4_lower=1e-4,
                center_c12_lower=0.05):
        """
        Trims a given TrackData DataFrame subject to core helium-burning 
        conditions to produce a track in the red clump (or horizontal branch).
        
        The start of the core helium-burning phase is defined by either an
        upper limit on central helium - default is (0.95 - initial_z) - or
        a lower limit on central carbon - default is 0.05. These are chosen
        so as to bypass the helium flash.
        
        The end of the core helium-burning phase is defined by a lower limit on
        central helium - default is 1e-4.

        Parameters
        ----------
        center_he4_upper : float
            The upper limit on the central helium fraction minus the
            initial metallicity.
        center_he4_lower : float
            The lower limit on the central helium fraction.
        center_c12_lower : float
            The lower limit on the central carbon fraction.
        
        Returns
        -------
        cropped_track : TrackData
            A TrackData object containing the cropped track.
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
    """
    Structure containing DataFrames from a MESA profile output file. TBC...

    Parameters
    ----------
    file_name : str, optional
        Path to the file from which the MESA history data is read (default is
        'LOGS/profile1.data').
    read_file : bool, optional
        Option to read the data in the history file (default is True).
    data : , optional
        The main data from the MESA history file (default is None).
    header : , optional
        The header data from the MESA history file (default is None).
    
    Attributes
    ----------
    file_name : str
        Where the MESA profile file is stored.
    data
        Where the main data from the MESA profile file is stored.
    header
        Where the header data from the MESA profile file is stored.
    """

    def __init__(self, file_name='LOGS/profile1.data', read_file=True,
                 header=None, data=None):
        MesaData.__init__(self, file_name=file_name, read_file=read_file,
                          header=header, data=data)

    def __repr__(self):
        return 'ProfileData(file_name={0}, header={1}, data={2})'.format(
            self.file_name, self.header, self.data)


class ProfileIndex(MesaData):
    """
    Reads the MESA profile index file.

    Parameters
    ----------
    file_name : str, optional
        Path to the file from which the MESA history data is read (default is
        'LOGS/profiles.index').
    read_file : bool, optional
        Option to read the data in the index file (default is True).
    data : , optional
        The main data from the MESA index file (default is None).
    header : str, optional
        The header line from the MESA index file (default is None).
    
    Attributes
    ----------
    file_name : str
        Where the MESA profile file is stored.
    data
        Where the main data from the MESA index file is stored.
    header : str
        Where the header line from the MESA index file is stored.
    """

    index_column_names = ['model_number', 'priority', 'profile_number']
    index_row_start = 1

    @classmethod
    def set_index_column_names(cls, names):
        cls.index_column_names = names

    @classmethod
    def set_index_row_start(cls, row):
        cls.index_row_start = row
    
    def __init__(self, file_name='LOGS/profiles.index', read_file=True,
                 header=None, data=None):
        MesaData.__init__(self, file_name=file_name, read_file=read_file,
                          header=header, data=data)

    def __repr__(self):
        return 'ProfileIndex(file_name={0}, header={1}, data={2})'.format(
            self.file_name, self.header, self.data)

    def read_file(self):
        self.header =  'test header - TBC'  # Soon will read header line
        self.data = read_csv(self.file_name, delim_whitespace=True,
                             names=ProfileIndex.index_column_names,
                             skiprows=ProfileIndex.index_row_start)


class MesaLog:
    """
    Reads the MESA LOGS directory and creates instances of TrackData,
    ProfileIndex and ProfileData where appropriate.

    Parameters
    ----------
    log_path : str, optional
        Path to the MESA LOGS directory (default is 'LOGS').
    history_file : str, optional
        Filename of the MESA track history file (default is 'history.data').
    profile_index_file : str, optional
        Filename of the MESA profile index file (default is 'profiles.index').
    profile_prefix : str, optional
        Prefix for the MESA profile files, not including the profile number
        (default is 'profile').
    profile_suffix : str, optional
        Suffix for the MESA profile files, also known as the file extension
        (default is 'data').
    read_track : bool, optional
        Option to automatically read the track data from the history file
        (default is None).
    read_all_profiles : bool, optional
        Option to automatically read all profile data in the directory, may
        take some time (default is None).

    Attributes
    ----------
    log_path : str
        Where the path to the MESA log directory is stored.
    history_file : str
        Where the filename of the MESA track history file is stored.
    profile_index_file : str
        Where the filename of the MESA profile index file is stored.
    profile_prefix : str
        Where the prefix for the MESA profile files is stored.
    profile_suffix : str
        Where the s uffix for the MESA profile files (or file extension) is 
        stored.
    """

    def __init__(self, log_path='LOGS', history_file='history.data',
                 profile_index_file='profiles.index',
                 profile_prefix='profile', profile_suffix='data',
                 read_track=None, read_all_profiles=None):
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


# MAKE MORE GENERAL, DO WITHOUT SEPARATE CONTROLS FOR MASS AND Z AND PUT IN
# ONE DICT
class Track:
    """
    Stores information about a MESA track. initial_conditions is a dict
    containing explicit initial conditions (over the MESA defaults and those
    specified in the project inlist) and the name is optional, otherwise will
    be autogenerated. Will have functionality to cross-check initial conditions
    with those in the MESA defaults and raise an error if not there.

    Parameters
    ----------
    initial_mass : float
        The initial mass of the star.
    initial_z : float
        The initial fractional metallicity of the star.
    initial_y : float, optional
        The initial fractional helium content of the star.
        If None (default) the helium fraction will be generated by the defualt 
        helium enrichmennt.
    name : str, optional
        The track name, to be used in the naming of the MESA output files.
        If None (default) the name will be generated to contain key information
        about the track.
    other_controls : dict, optional
        Any other initial conditions for the track must be placed here (default
        is None). The key should correspond to the MESA inlist option, 
        otherwise the option will not work.
    other_controls_name : dict, optional
        If you want any of other_controls to appear in the auto-generated
        name then they should be placed in a dictionary containing a key
        corresponding to that of other_controls and the value corresponding
        to its abbreviation to appear in the name.
    dydz : float, optional
        TO BE MADE A CLASS VARIABLE? The helium enrichment law used when 
        generating initial_y (equivilant to dy/dz). Default is 2.0 as used
        in MESA by default.
    primordial_y : float, optional
        TO BE MADE A CLASS VARIABLE? The primordial helium abundance fraction
        to be used when generating initial_y. Default is 0.24 as used by MESA.
    """

    def __init__(self, initial_mass, initial_z, initial_y=None, name=None,
                 other_controls=None, other_controls_names=None,
                 dydz=2.0, primordial_y=0.24):
        self.initial_mass = initial_mass
        self.initial_z = initial_z
        if initial_y:
            self.initial_y = initial_y
        else:
            self.set_initial_y(dydz, primordial_y)
        self.other_controls = other_controls
        self.other_controls_names = other_controls_names

        if name:
            self.name = name
        else:
            self.set_name()
        
    def set_name(self):
        """
        Sets the name of the track according to the following convension:
        m.<initial_mass>.z.<initial_z>.y.<initial_y>.other_abbrv.<other>
        where the elements in angle brackets correspond to the Track
        attributes (angle brackets not included).
        The initial mass is rounded to 2 decimal places. The initial
        metallicity and helium are also rounded to 5 decimal places for
        readability.
        """
        name = ['m.{0:.2f}.z.{1:.5f}.y.{2:.5f}'.format(self.initial_mass,
                                                       self.initial_z, 
                                                       self.initial_y)]  
        if self.other_controls_names:
            for i, j in self.other_controls_names.items():
                name.append(j)
                name.append(str(self.other_controls[i]))
        elif self.other_controls:
            for i, j in self.other_controls.items():
                name.append(i)
                name.append(str(j))
        
        self.name = '.'.join(name)
        print(self.name)

    def set_initial_y(self, dydz, primordial_y):
        """
        Sets the initial helium enrichment for a given enrichment law and 
        primordial helium abundance.

        Parameters
        ----------
        dydz : float, optional
            TO BE MADE A CLASS VARIABLE? The helium enrichment law used when 
            generating initial_y (equivilant to dy/dz). Default is 2.0 as used
            in MESA by default.
        primordial_y : float, optional
            TO BE MADE A CLASS VARIABLE? The primordial helium abundance fraction
            to be used when generating initial_y. Default is 0.24 as used by MESA.
        """
        self.initial_y = primordial_y + dydz*self.initial_z
    
    def write_inlist(self):
        """
        Writes the specified conditions of the track to the file
        'inlist_grid_vals'.
        """
        f = open('inlist_grid_vals', 'w')
        
        # Star Job -  currently empty
        f.write('&star_job\n')
        f.write('/\n')
        
        # Controls
        f.write('&controls\n')

        # Filenames currently done for use with a grid.
        f.write("\tstar_history_name = '%s.track'\n" %(self.name+'/'+self.name))
        f.write("\tprofiles_index_name = '%s.index'\n" %(self.name+'/'+self.name))
        f.write("\tprofile_data_prefix = '%s_n'\n" %(self.name+'/'+self.name))
        
        f.write('\tinitial_mass = {}\n'.format(self.initial_mass))
        f.write('\tinitial_z = {}\n'.format(self.initial_z))
        f.write('\tZbase = {}\n'.format(self.initial_z))
        f.write('\tinitial_y = {}\n'.format(self.initial_y))
        
        if self.other_controls:
            for i, j in self.other_controls.items():
                # For each other control, write its value on a new line.
                f.write('\t{0} = {1}\n'.format(i, j))

        f.write('/\n')
        f.close()

    def run(self, log_console_output=True):
        """
        Runs MESA for the given track.

        Parameters
        ----------
        log_console_output : bool, optional
            If True (default) the console output is logged in the MESA LOGS
            directory under <self.name>.out.
        """
        os.system('mkdir -p LOGS/'+self.name)
        self.write_inlist()
        
        if log_console_output:
            # Put console output into separate file
            os.system('./rn > LOGS/'+self.name+'.out')
        else:
            os.system('./rn')
        os.system('echo '+self.name+' >> track_finished')
        

class MesaGrid:
    """
    Creates a grid of track data from a given directory path or specified
    list of file paths. Methods to access tracks with a given initial condition
    MesaGrid.search_grid() and sort the grid by mass/metallicity etc. will
    be added. To be compatible with another element of this MESA companion
    which allows for the running of grids.

    Takes in dictionary of grid values and initialises a grid of Tracks with
    the given initial conditions.

    """

    def __init__(self, initial_mass_list, initial_z_list, initial_y_list=None, 
                 other_controls_list=None, other_controls_names=None,
                 path_to_grid='LOGS'):
        self.initial_mass_list = initial_mass_list  # iterable
        self.initial_z_list = initial_z_list  # iterable - not used??
        self.initial_y_list = initial_y_list  # iterable or None
        self.other_controls_list = other_controls_list  # dict of iterables
        self.other_controls_names = other_controls_names  # dict
        self.path_to_gid = path_to_grid  # str
        

        # CONSIDER MAKING CONTROLS_LIST A LIST NOT A DICT AND THEN MAKING
        # A SEPARATE LIST OF DICTIONARY NAMES.
        self.controls_names = ['initial_mass', 'initial_z']
        self.controls_list = [self.initial_mass_list, self.initial_z_list]
        # self.controls_list = {'initial_mass': self.initial_mass_list,
        # 'initial_z': self.initial_z_list}
        if self.initial_y_list:
            self.controls_names.append('initial_y')
            self.controls_list.append(self.initial_y_list)
            # self.controls_list['initial_y'] = initial_y_list
        if self.other_controls_list:
            self.controls_names.append(
                [key for key in self.other_controls_list.keys()]
                # Note that these are the controls names not the file ref names
            )
            self.controls_list.append(
                [val for val in self.other_controls_list.values()]
            )
            # self.controls_list.update(other_controls_list)
        
        # # Create the grid and a whole bunch of Track objects
        # grid_shape = []
        # # Change below so that it only creates a grid for values that change
        # grid_shape = [len(self.initial_mass_list), len(self.initial_z_list)]
        # if initial_y_list:
        #     grid_shape.append(len(initial_y_list))
        # if other_controls_list:
        #     for i in other_controls_list.values():
        #         grid_shape.append(len(i))
        #
        # self.grid_tracks = np.ndarray(grid_shape)

        # Alternatively - must be shorter way? Using xarray!
        grid_shape = [len(i) for i in self.controls_list]
        self.grid_tracks = xr.DataArray(np.ndarray(grid_shape, dtype=Track),
                                        coords=self.controls_list,
                                        dims=self.controls_names)
        n_controls = len(self.controls_list)
        n = 0
        c = np.meshgrid(*self.controls_list)  # grid of coord combinations
        c = np.array([i.flatten() for i in c])  # flatten each coord

        # axis 0 = each dimension
        # axis 1 = need to flatten along this
        # now you can iterate through c to create a track at each coord.
        # for i in range(np.size(c, axis=1)):
        #     # This may be quite slow, better to index with 
        #     self.grid_tracks.sel(dict(zip(self.controls_names, c[:, i]))) = Track(c[0, i], c[1, i], initial_y=None)



    def get_track(self, grid_axis, grid_value):
        return None