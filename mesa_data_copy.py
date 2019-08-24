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


class Track:
    """
    Stores information about a MESA track. initial_conditions is a dict
    containing explicit initial conditions (over the MESA defaults and those
    specified in the project inlist) and the name is optional, otherwise will
    be autogenerated. Will have functionality to cross-check initial conditions
    with those in the MESA defaults and raise an error if not there.

    Parameters
    ----------
    name : str, optional
        The track name, to be used in the naming of the MESA output files.
        If None (default) the name will be generated to contain key information
        about the track.
    controls : dict, optional
        Any other initial conditions for the track must be placed here (default
        is None). The key should correspond to the MESA inlist option, 
        otherwise the option will not work.
    controls_abbreviations : dict, optional
        If you want any of other_controls to appear in the auto-generated
        name then they should be placed in a dictionary containing a key
        corresponding to that of other_controls and the value corresponding
        to its abbreviation to appear in the name.
    calculate_initial_y : bool, optional
        Will calculate initial helium enrichment provided "initial_z" in the
        controls, using the optional parameters dydz and primordial_y. Defualt
        is false.
    dydz : float, optional
        TO BE MADE A CLASS VARIABLE? The helium enrichment law used when 
        generating initial_y (equivilant to dy/dz). Default is 2.0 as used
        in MESA by default.
    primordial_y : float, optional
        TO BE MADE A CLASS VARIABLE? The primordial helium abundance fraction
        to be used when generating initial_y. Default is 0.24 as used by MESA.
    initial_y_precision : int, optional
        The precision to which the initial_y value is calculated. Default is 5
        which corresponds to 5 places after the decimal point.
    name_precisions : dict, optional
        The precisions at which the controls quoted in the name of the Track
        are to be given too. Default is an empty dict,
        precision is automatic based on what is entered as the control.
    filenaming_method : str, optional
            Rather than specifying filename controls, one can automatically 
            set them via a method. 'mesa' uses MESA's default namescheme.
            'grid' uses the Track name to generate identifiable filenames.
            Default behaviour is 'mesa'. Set to None if manually providing
    """

    def __init__(self, name=None, controls=None, controls_abbreviations=None,
                 name_precisions=dict(), filenaming_method='mesa',
                #  log_path=None, history_file=None, profile_index_file=None,
                #  profile_prefix=None, profile_suffix=None,
                 calculate_initial_y=False, dydz=2.0, primordial_y=0.24,
                 initial_y_precision=5,
                 track_data=None, profile_index=None, profile=dict()):

        self.controls = controls
        self.controls_abbreviations = controls_abbreviations
        self.name_precisions = name_precisions

        if calculate_initial_y:
            # May want to take this out of the initialisation and have the 
            # user run the set_initial_y() function which will automatically
            # reset the name?
            self.set_initial_y(dydz, primordial_y, precision=initial_y_precision)
        
        self.reset_name(new_name=name)  # may be useful to specify each
        
        self.reset_filenames(method=filenaming_method)

        self.track_data = track_data
        self.profile_index = profile_index
        self.profile = profile

    def reset_filenames(
            self, method='mesa', log_path=None, history_file=None,
            profile_index_file=None, profile_prefix=None, profile_suffix=None
            ):
        if method is 'mesa':
            fns = ['LOGS', 'history.data', 'profiles.index', 'profile', 'data']
        elif method is 'grid':
            fns = [
                '/'.join([self.log_path, self.name]),
                '.'.join([self.name, 'track']), '.'.join([self.name, 'index']),
                '_'.join([self.name, 'n']), 'profile'
                ]
        elif not method:
            fns = [
                log_path, history_file, profile_index_file, profile_prefix,
                profile_suffix
                ]
        
        self.log_path, self.history_file, self.profile_index_file, \
            self.profile_prefix, self.profile_suffix = fns

    def read_log_dir(self, read_track=False, read_all_profiles=False):
        """Reads the LOGS directory.
        """
        self.track_data = TrackData(
            file_name='/'.join([self.log_path, self.history_file]),
            read_file=read_track
            )
        self.profile_index = ProfileIndex(
            file_name='/'.join([self.log_path, self.profile_index_file])
            )
        
        for i, m in enumerate(
            self.profile_index.data[ProfileIndex.index_column_names[0]]):
            # For each model number
            self.profile[str(m)] = ProfileData(
                file_name='{0}/{1}{2}.{3}'.format(self.log_path,
                                                  self.profile_prefix, i+1,
                                                  self.profile_suffix),
                read_file=read_all_profiles)
    
    def read_track(self):
        self.track_data.read_file()
    
    def read_profile(self, model_number):
        self.profile[str(model_number)].read_file()        
                
    def reset_name(self, new_name=None):
        """
        Sets the name of the track according to the following convension:
        control_abbreviation.<control_value> (repeats for each control)
        where the elements in angle brackets correspond to the Track control
        attributes (angle brackets not included).
        """
  
        name = []
        if self.controls_abbreviations:
            for i, j in self.controls_abbreviations.items():
                name.append(j)
                try:
                    # Tries to use precision name
                    # Could avoid repetition and put this try except in a 
                    # function.
                    name.append('{0:.{1}f}'.format(self.controls[i],
                                                   self.name_precisions[i]))
                except KeyError:
                    # If no precision for such key, just quotes directly.
                    # note this might fail if there is a key error for 'i'.
                    # so another try except catches this - IMPROVE CATCH EARLY
                    try:
                        name.append(str(self.controls[i]))
                    except KeyError as kerr:
                        print("Control abbreviations key '{0}' ".format(kerr) +
                              "could not be found in controls " +
                              "- abbreviation not added to name.")

        elif self.controls:
            # Else name Track using full list of controls - ADD WARNING here
            # if length becomes long.
            for i, j in self.controls.items():
                name.append(i)
                try:
                    name.append('{0:.{1}f}'.format(j, self.name_precisions[i]))
                except KeyError:
                    name.append(str(j))
        elif new_name:
            name.append(new_name)
        else:
            name.append('untitled_track')
        
        self.name = '.'.join(name)
        print(self.name)

    def set_initial_y(self, dydz, primordial_y, precision=5):
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
        precision : int, optional
            Number of decimal places to round initial_y calculation, default is
            5.
        """
        # HAVE THIS RESET THE NAME OF THE TRACK TO ACCOUNT FOR THIS
        try:
            self.controls['initial_y'] = round(
                primordial_y + dydz*self.controls['initial_z'], precision
                )
        except KeyError:
            print('Cannot set initial_y without initial_z. Please add a ' +
                'value for "initial_z" in the Track.controls.')
    
    def write_inlist(self):
        """
        Writes the specified conditions of the track to the file
        'inlist_grid_vals'.
        """
        f = open('inlist_grid_vals', 'w')
        
        # Star Job -  currently empty as not added this functionality
        f.write('&star_job\n')
        f.write('/\n')
        
        # Controls
        f.write('&controls\n')

        # Make the following optional in case such details are in another inlist
        f.write("\tlog_directory = '{0}'\n".format(self.log_path))
        f.write("\tstar_history_name = '{0}'\n".format(self.history_file))
        f.write("\tprofiles_index_name = '{0}'\n".format(self.profile_index_file))
        f.write("\tprofile_data_prefix = '{0}'\n".format(self.profile_prefix))
        f.write("\tprofile_data_suffix = '{0}'\n".format(self.profile_suffix))
        if self.controls:
            for i, j in self.controls.items():
                # For each other control, write its value on a new line.
                f.write('\t{0} = {1}\n'.format(i, j))

        f.write('/\n')
        f.close()

    def run(self, path='', log_console_output=True):
        """
        Runs MESA for the given track.

        Parameters
        ----------
        log_console_output : bool, optional
            If True (default) the console output is logged in the MESA LOGS
            directory under <self.name>.out.
        """
        # os.system('mkdir -p LOGS/'+self.name)
        self.write_inlist()
        
        if log_console_output:
            # Put console output into separate file
            os.system('.'+path+'/rn > LOGS/'+self.name+'.out')
        else:
            os.system('.'+path+'/rn')
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

    CURRENTLY DOESN'T WORK IF YOU HAVE COUPLED CONTROLS (E.G. ZBASE GOES WITH
    INITIAL_Z, OR OVERSHOOTING CONTROLS WHICH COUPLE)
    """

    def __init__(self, controls=dict(), controls_abbreviations=dict(),
                 path_to_grid='LOGS'):
        # Need to add parameters for every attribute so it is possible to
        # recreate from itself.
        self.controls = controls  # dict of iterables
        for i, j in self.controls.items():
            # Crudely turns each item into a list
            self.controls[i] = [j] if not isinstance(j, (list, tuple)) else j
        self.controls_abbreviations = controls_abbreviations  # dict
        self.path_to_gid = path_to_grid  # str

        self.controls_names = [key for key in self.controls.keys()]
            # Note that these are the controls names not the file ref names
        self.controls_list = [val for val in self.controls.values()]
        self.create_tracks()  # Will define grid tracks from controls

        # An array of all combinations of controls list
        self.coords = np.meshgrid(*self.controls_list)  # grid of coord combinations
        self.coords = np.array([i.flatten() for i in self.coords])  # flatten each coord

    def create_tracks(self):
        grid_shape = [np.size(i) for i in self.controls_list]
        self.grid_tracks = xr.DataArray(np.ndarray(grid_shape, dtype=Track),
                                        coords=self.controls_list,
                                        dims=self.controls_names)

        for i in range(np.size(self.coords, axis=1)):
            # For each combination of controls, create a track
            track_controls = dict(zip(self.controls_names, self.coords[:, i]))
            self.grid_tracks.loc[track_controls] = \
                Track(controls=track_controls,
                      controls_abbreviations=self.controls_abbreviations,
                      filenaming_method='grid')

    def read_logs(self, args, kwargs):
        for i in range(np.size(self.coords, axis=1)):
            # For each combination of controls, read logs
            track_controls = dict(zip(self.controls_names, self.coords[:, i]))
            self.grid_tracks.loc[track_controls].read_log_dir(read_track=True) # need args!!!

    # def interpolate(self, coords, track_columns, method='linear')