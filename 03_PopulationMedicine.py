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

bc_data_path = '../wdbc.csv'  # path to the input data file.

bc_data_columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                   'compactness_mean', 'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
                   'perimeter_se', 'area_se', 'smoothness_se',
                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                   'radius_worst', 'texture_worst', 'perimeter_worst',
                   'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                   'symmetry_worst', 'fractal_dimension_worst']

def get_neighbours(data, point_idx, eps):
        neighbours = set()
        #point = data[point_idx]
        for q_idx in range(len(data)):
            q = data(q_idx)
            dist = np.linalg.norm(point - q)
            if dist <= eps:
                neighbours.add(q_idx)
        return neighbours

def expand_cluster(data, point_idx, neighbours, eps, minPts, visited):
        cluster = {point_idx}
        point = data[point_idx]
        
        for q in range(len(neighbours)):
            if q not in visited:
                visited.add(q)
                q_point = data[q]
                q_neighbours = get_neighbours(data, q_point, eps)
                if len(q_neighbours) >= minPts:
                    neighbours.update(q_neighbours)
            if q not in cluster:
                cluster.add(q)
        return cluster

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
    scaler = StandardScaler()
    data = scaler.fit_transform(dataExceptLabels.to_numpy())

    # TODO Step 2: Initialize some variables you might need to store information. Follow the pseudo-code.
    clusters = []
    visited = set()
    noise = set()

    # TODO Step 3: Write the get_neighbours helper function. You can do this before or after implementing the main code for the DBSCAN algorithm.
    
    # TODO Step 3: Write the expand_clusters helper function. You can do this before ot after implementing the main code for the DBSCAN algorithm.

    # TODO Step 3: Implement the DBSCAN algorithm.
    for i in range(len(data)):
        if i not in visited:
            visited.add(i)
            p = data[i]
            neighbours = get_neighbours(data, p, eps)
            if len(neighbours) < minPts:
                noise.add(i)
            else:
                new_cluster = expand_cluster(data, i, neighbours, eps, minPts, visited)
                clusters.append(new_cluster)

    # TODO Step 4: Lets analyse the results
    # TODO Step 4.1: Lets print the amount of clusters and noise data points first.
    n_clusters = len(clusters)
    n_noise = len(noise)
    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')

    # TODO Step 4.2: Lets create a new numpy array and for each data point we add the label information according to DBSCAN
    #  use (len(data), -1) to initialize all points as noise
    dbscan_labels = np.full((len(dataExceptLabels),), -1)
    for q_idx in range(len(clusters)):
        if q_idx not in noise:
            for cluster_idx, cluster in enumerate(clusters):
                if q_idx in cluster:
                    dbscan_labels[q_idx] = cluster_idx

    # TODO Step 4.3: Lets plot all data point (noise once included)
    #  Use the following color code:
    #  unique_labels = set(labels)
    #  colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #  Color each datapoint according to its label and plot them. For noise you can use color black --> [0, 0, 0, 1]
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # plt.figure(figsize=(10, 7))
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         col = [0, 0, 0, 1]  # Black color for noise
    #     class_member_mask = (dbscan_labels == k)
    #     xy = data.values[class_member_mask]
    #     plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise')
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black color for noise
        class_mask = (dbscan_labels == k)
        xy = data[class_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}' if k != -1 else 'Noise')

    plt.title('DBSCAN with Noise')
    plt.show()

    # TODO Step 4.4: Remove noise data points and plot only the remaining once. Use the same color coding.
    no_noise_mask = dbscan_labels != -1
    data_no_noise = data[no_noise_mask]
    labels_no_noise = dbscan_labels[no_noise_mask]

    unique_labels_no_noise = set(labels_no_noise)
    colors_no_noise = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels_no_noise))]
    
    plt.figure()
    for k, col in zip(unique_labels_no_noise, colors_no_noise):
        class_mask = (labels_no_noise == k)
        xy = data_no_noise[class_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}')
        
    plt.title('DBSCAN without Noise')
    # plt.xlabel('radius_mean')
    # plt.ylabel('fractal_dimension_mean')
    # plt.legend()
    plt.show()

    return clusters, noise


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