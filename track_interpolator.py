# Author:   Alex Lyttle
# Date:     27-07-2019
# Description: Define a class which reads a list of TrackData objects and
# interpolates in age, mass and metallicity given certain tolerences.
# It could be in the form of a DataFrame with columns for age, mass and 
# metallicity, or it could be a multidimentionaly numpy array (whichever is 
# easier to sample from - probably the latter).

# Use scipy.interpolate.griddata with linear or cubic method where points is
# the set of ages, masses and metallicities available and values are the 
# numpy.arrays of column values for each one
import numpy as np
import pickle
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata
from tqdm import tqdm
from matplotlib import pyplot as plt

def norm_age(track_data):
    min_age = min(track_data['star_age'])
    max_age = max(track_data['star_age'])
    track_data['norm_age'] = (track_data.loc[:, 'star_age'] - min_age) /\
        (max_age - min_age)
    return track_data

def interpolate(feh, mass, col, tracks, n_feh=100, n_mass=400, n_age=400,
                method='linear'):
    """Interpolates an array of track data over mass and metallicity.

    Parameters
    ----------
    feh : list (float)
        List of metallicities - must be the same length as the first axis of
        tracks.
    mass : list (float)
        List of masses - must be 2D, with the first axis varying with
        metallicity and the second axis varying with mass.
    col : list (str)
        List of columns from the track data to be interpolated.
    n_feh : int, optional
        Size of the interpolated grid in metallicity.
    n_mass : int, optional
        Size of the interpolated grid in mass.
    n_age : int, optional
        Size of the interpolated grid in normalised age.
    method : str, optional
        Interpolation method - either 'linear' (default) or 'nearest'.
    """    
    n_col = len(col)
    n_dim = 3

    x = np.ndarray((0, n_dim))
    Y = np.ndarray((0, n_col))

    for i in tqdm(range(len(tracks))):
        # For each metallicity
        for j, tr in enumerate(tracks[i]):
            # For each mass
            len_age = len(tr['norm_age'])  # Length of normalised ages for tr
            xx = np.ndarray((len_age, n_dim))
            xx[:, 0] = np.tile(feh[i], len_age)
            xx[:, 1] = np.tile(mass[i][j], len_age)
            xx[:, 2] = tr['norm_age']
            x = np.append(x, xx, axis=0)
            Y = np.append(Y, tr[col], axis=0)

    feh_fine = np.linspace(np.min(feh), np.max(feh), n_feh)
    mass_fine = np.linspace(np.min(mass), np.max(mass), n_mass)
    age_fine = np.linspace(0, 1, n_age)
    grid_feh, grid_mass, grid_age = np.meshgrid(feh_fine, mass_fine, age_fine)
    
    print('Begin interpolation...')
    interpolated = griddata(x, Y, (grid_feh, grid_mass, grid_age), 
                            method=method, rescale=True)
    print('Done.')
    
    return interpolated

        
if __name__ == "__main__":
    
    # To test the interpolator, use a reduced grid of tracks and then plot an 
    # interpolated track against a MESA generated track - qunatify the
    # difference with a metric of some kind (e.g. diff in lum with mass)
    rc = pickle.load(open('test-data/rc_grid.pkl', 'rb'))
    params = pickle.load(open('test-data/grid_params.pkl', 'rb'))
    feh = params['Fe_H']
    mass = params['M'][0]  # Test with overshoot = 0 first
    tracks = rc[0]

    # Compute normalised ages
    for t in tqdm(tracks):
        # For each Fe_H
        for tr in t:
            # For each mass
            tr = norm_age(tr)
    
    col = ['effective_T', 'luminosity', 'radius']
    interp = interpolate(feh, mass, col, tracks)
    # Structure of interp is [mass, feh, age, columns] for some reason
    plt.figure()
    for i in range(20):
        plt.loglog(interp[20*i, 0, :, 0], interp[20*i, 0, :, 1], alpha=0.3)
    plt.gca().invert_xaxis()
