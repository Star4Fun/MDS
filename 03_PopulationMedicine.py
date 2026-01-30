import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from collections import deque

# For reproducibility
np.random.seed(1000)

bc_data_path = '../wdbc.data'  # path to the input data file.

bc_data_columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                   'compactness_mean', 'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
                   'perimeter_se', 'area_se', 'smoothness_se',
                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                   'radius_worst', 'texture_worst', 'perimeter_worst',
                   'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                   'symmetry_worst', 'fractal_dimension_worst']


#######################################################################################################################
# Function db_scan(dataExceptLabels, labels):
# Function performs Density-based spatial clustering on provided data (from scratch, not using sklearn)
#
# Input arguments:
#   - dataframe dataExceptLabels: pandas DataFrame of attributes except labels
#   - Series labels: pandas Series of labels
#
# Output:
#   - no. of estimated clusters using DBSCAN
#   - no. of noise points using DBSCAN
#   - remove the noise points from data and labels
#   - shows the plot of data with noise points
#   - shows the plot of data without noise points
#   - we return None here since we don't want to further progress on the output
#######################################################################################################################

def db_scan(dataExceptLabels, labels, eps=0.3, minPts=10):
    # TODO Step 1: Scale data first and convert to numpy use the StandardScaler again

    # TODO Step 2: Initialize some variables you might need to store information. Follow the pseudo-code.

    # TODO Step 3: Write the get_neighbours helper function. You can do this before or after implementing the main code for the DBSCAN algorithm.

    # TODO Step 3: Write the expand_clusters helper function. You can do this before ot after implementing the main code for the DBSCAN algorithm.

    # TODO Step 3: Implement the DBSCAN algorithm.


    # TODO Step 4: Lets analyse the results
    # TODO Step 4.1: Lets print the amount of clusters and noise data points first.


    # TODO Step 4.2: Lets create a new numpy array and for each data point we add the label information according to DBSCAN
    #  use (len(data), -1) to initialize all points as noise

    # TODO Step 4.3: Lets plot all data point (noise once included)
    #  Use the following color code:
    #  unique_labels = set(labels)
    #  colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #  Color each datapoint according to its label and plot them. For noise you can use color black --> [0, 0, 0, 1]

    # TODO Step 4.4: Remove noise data points and plot only the remaining once. Use the same color coding.


    return None


if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv(bc_data_path, names=bc_data_columns).fillna(0.0)

    # Drop the  "id" as we don't need
    data.drop(["id"], axis=1, inplace=True)

    # count the lable values
    data["diagnosis"].value_counts()
    print(data["diagnosis"].value_counts())

    dataExceptLables = data.loc[:, ['radius_mean', 'fractal_dimension_mean']]
    diagonsis = pd.Series(data['diagnosis'].replace(['M', "B"], [0, 1]))

    true_labels = array(diagonsis)
    dataExceptLables.head()

    print(dataExceptLables.info())
    db_scan(dataExceptLables, true_labels)