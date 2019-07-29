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

class TrackInterpolator:
    """Contains methods for interpolating a grid of TrackData objects.

    Need a grid of TrackData objects 
    """
    
    def __init__(self):
        return None
    
    def norm_age(self, track_data):
        min_age = min(track_data['star_age'])
        max_age = max(track_data['star_age'])
        track_data['norm_age'] = (track_data.loc[:, 'star_age'] - min_age) /\
            (max_age - min_age)
    
    def  sort_grid(self, tracks, masses, metallicities):
        x = [[[]]]
        sorted_tracks = [[[]]]
        for i in range(len(metallicities)):
            for j in masses:
                sorted_tracks.append(tracks[i][j])