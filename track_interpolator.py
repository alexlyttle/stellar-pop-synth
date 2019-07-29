# Author:   Alex Lyttle
# Date:     27-07-2019
# Description: Define a class which reads a list of TrackData objects and
# interpolates in age, mass and metallicity given certain tolerences.
# It could be in the form of a DataFrame with columns for age, mass and 
# metallicity, or it could be a multidimentionaly numpy array (whichever is 
# easier to sample from - probably the latter). It 