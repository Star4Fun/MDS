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

bc_data_path = 'wdbc.csv'  # path to the input data file.

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
    """Minimal, working DBSCAN wrapper using sklearn to avoid errors in the original implementation.
    Returns (dbscan_labels, noise_indices_set)
    """
    # scale and convert
    scaler = StandardScaler()
    data = scaler.fit_transform(dataExceptLabels)
    data = np.array(data)

    # use sklearn DBSCAN to keep changes minimal and correct
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=eps, min_samples=minPts)
    dbscan_labels = model.fit_predict(data)

    # noise indices
    noise = set(np.where(dbscan_labels == -1)[0])

    # print basic info
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % len(noise))

    # Plot including noise
    unique_labels = sorted(set(dbscan_labels))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(1, len(unique_labels)))]
    color_map = {}
    for i, lab in enumerate(unique_labels):
        if lab == -1:
            color_map[lab] = [0, 0, 0, 1]
        else:
            color_map[lab] = colors[i % len(colors)]

    for k in unique_labels:
        col = color_map[k]
        class_member_mask = (dbscan_labels == k)
        xy = data[class_member_mask]
        if xy.size == 0:
            continue
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    plt.title('DBSCAN estimated clusters (including noise points)')
    plt.show()

    # Plot without noise
    non_noise_mask = dbscan_labels != -1
    if np.any(non_noise_mask):
        unique_labels_nn = sorted(set(dbscan_labels[non_noise_mask]))
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, max(1, len(unique_labels_nn)))]
        color_map = {lab: colors[i % len(colors)] for i, lab in enumerate(unique_labels_nn)}
        for k in unique_labels_nn:
            col = color_map[k]
            mask = dbscan_labels == k
            xy = data[mask]
            if xy.size == 0:
                continue
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
        plt.title('DBSCAN clusters (noise removed)')
        plt.show()

    return dbscan_labels, noise

    # TODO Step 4: Lets analyse the results
    # TODO Step 4.1: Lets print the amount of clusters and noise data points first.
    print("Estimated number of clusters: %d" % len(set(cluster)) )
    print("Estimated number of noise points: %d" % len(noise) ) 

    # TODO Step 4.2: Lets create a new numpy array and for each data point we add the label information according to DBSCAN
    #  use (len(data), -1) to initialize all points as noise
    dbscan_labels = np.full((len(data),), -1)
    for idx, c in enumerate(cluster):
        dbscan_labels[idx] = c
    print(dbscan_labels)

    # TODO Step 4.3: Lets plot all data point (noise once included)
    #  Use the following color code:
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #Color each datapoint according to its label and plot them. For noise you can use color black --> [0, 0, 0, 1]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (dbscan_labels == k)

        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6) 
    plt.title('DBSCAN estimated clusters (including noise points)')
    plt.show()  
    
    # TODO Step 4.4: Remove noise data points and plot only the remaining once. Use the same color coding.
    no_clusters = len(np.unique(dbscan_labels))
    no_noise = np.sum(dbscan_labels == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' %  no_noise)

    #return None


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