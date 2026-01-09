import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.cm as cm


#######################################################################################################################
# Function: plot_random_data(n_samples, centers)
# Description:
#     This function generates and plots random data points using a Gaussian blob distribution. The data is visualized
#     as a scatter plot with each cluster distinguished by a color.
#
# Input arguments:
#     - int n_samples: Number of samples to generate (default is 2000).
#     - list centers: List of center points for each cluster. Each sublist specifies the x and y coordinates of
#                     the cluster center. Default value: [[-2, -1], [4, 4], [1, 1]].
#
# Output:
#     - None (the function does not return any value, but it plots the data)
#######################################################################################################################

def plot_random_data(n_samples=2000, centers=[[-2, -1], [4, 4], [1, 1]]):
    # Set the random seed to ensure reproducibility of the results
    np.random.seed(0)

    # TODO Step 1: Generate synthetic data using 'make_blobs'
    # X: Array of shape [n_samples, n_features] representing the feature matrix
    # y: Array of shape [n_samples,] representing the cluster labels for each sample
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2, random_state=0)

    # TODO Step 2: Create a scatter plot of the generated data points
    # X[:, 0] represents the x-coordinates, X[:, 1] represents the y-coordinates
    # 'c=y' colors the points based on their cluster labels
    # 'alpha=0.5' sets the transparency of the points to 50%
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)

    # TODO Step 3: Set the title of the plot, assign Labels to the axis and display the plot
    plt.title('Plot Random Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X, y  # Return generated data for reuse

#######################################################################################################################
# Function: BSAS(data, max_clusters=3, threshold=1.0)
# Description:
#     This function implements the Basic Sequential Algorithmic Scheme (BSAS) to cluster the input data.
#     The algorithm sequentially assigns data points to existing clusters or creates new clusters based on
#     a user-defined distance threshold and maximum number of clusters.
#
# Input arguments:
#     - np.array data: The feature matrix (shape [n_samples, n_features]) containing the data points to cluster.
#     - int max_clusters: The maximum number of clusters allowed (default is 3).
#     - float threshold: The maximum allowable distance for a point to be assigned to an existing cluster (default is 1.0).
#
# Output:
#     - np.array bsas_cluster_centers: Coordinates of the cluster centers.
#     - np.array bsas_labels: Cluster labels for each data point.
#######################################################################################################################

def BSAS(data, max_clusters=3, threshold=1.0):
    # TODO Step 1: Initialize the first cluster with the first data point. Think about what else you need.
    data = np.array(data)
    n_samples = data.shape[0]
    if n_samples == 0:
        return np.empty((0, data.shape[1])), np.empty((0,), dtype=int)
    
    #initialize
    centers = [data[0].astype(float).copy()]
    counts = [1]
    labels = -1 * np.ones(n_samples, dtype=int) 
    labels[0] = 0
    n_clusters = 1

    # TODO Step 2: Iterate through all other data points to assign them to clusters
    for i in range(1, data.shape[0]):
        x = data[i].astype(float)

        # TODO Step 2a: Calculate distance from the current point to each cluster center
        #  HINT: You can use np.linalg.norm to get all distances and np.argmin to get the closest cluster.
        arr_centers = np.vstack(centers)
        distances = np.linalg.norm(arr_centers - x, axis=1)
        closest_cluster = np.argmin(distances)
        min_distance = distances[closest_cluster]

        # TODO Step 2b: Check if the point should create a new cluster
        #  If yes, create a new cluster center; otherwise, assign the point to the closest cluster.
        #  Update the cluster center position after assignment.
        if min_distance > threshold and n_clusters < max_clusters:
            centers.append(x)
            counts.append(1)
            labels[i] = n_clusters
            n_clusters += 1
        else:
            labels[i] = closest_cluster
            counts[closest_cluster] += 1
            # Update the cluster center
            centers[closest_cluster] = centers[closest_cluster] + (x - centers[closest_cluster]) / counts[closest_cluster]

    # TODO Step 3: Plot the clustered data points and cluster centers
    #  Titles, x and y labels are always nice to have. Display the plot.

    bsas_cluster_centers = np.vstack(centers)
    bsas_labels = labels.copy()

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=bsas_labels, alpha=0.5)
    plt.scatter(bsas_cluster_centers[:, 0], bsas_cluster_centers[:, 1], c='k', marker='x', s=100, linewidths=2)
    plt.title(f'BSAS (k_max={max_clusters}, threshold={threshold})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    #plt.tight_layout()
    plt.show()

    # TODO Step 4: Return cluster centers and labels as numpy arrays.
    return bsas_cluster_centers, bsas_labels


#######################################################################################################################
# Function: k_means(nb_clusters, data, generate_plot=True)
# Description:
#     This function applies the K-Means clustering algorithm on the input data and visualizes the results using a
#     2D scatter plot. Each cluster is displayed in a different color, and the cluster centroids are marked
#     as distinct points.
#
# Input arguments:
#     - int nb_clusters: The number of clusters to create (k value for K-Means clustering).
#     - np.array data: The feature matrix (shape [n_samples, n_features]) to be clustered.
#     - boolean generate_plot: If True, generates a plot of the clustering results (default is True).
#
# Output:
#     - k_means_cluster_centers: Coordinates of the cluster centers.
#     - k_means_labels: Labels for each point indicating the cluster it belongs to.
#######################################################################################################################

def k_means(nb_clusters, data, generate_plot=True):
    # TODO Step 1: Initialize the KMeans object using k-means++ for centroid initialization.
    #  Use n_clusters=nb_clusters and n_init=12.
    kmeans = KMeans(n_clusters=nb_clusters, init='k-means++', n_init=12)

    # TODO Step 2: Fit the KMeans model to the input data and extract labels and cluster centers.
    kmeans.fit(data)
    k_means_labels = kmeans.labels_
    k_means_cluster_centers = kmeans.cluster_centers_

    # Generate a plot only if generate_plot=True
    if generate_plot:  # Uncomment to enable plotting
        # TODO Step 3: Create a scatter plot with cluster centers and data points.
        #  Assign colors to points based on their labels using plt.cm.Spectral or similar colormap.
        #  We recommend using the following color map.
        colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels)) + 1))[:-1]

        plt.figure()
        for k, col in zip(range(nb_clusters), colors):
            plt.scatter(data[k_means_labels == k, 0], data[k_means_labels == k, 1], c=[col], alpha=0.5, label=f'cluster {k}')
            plt.scatter(k_means_cluster_centers[k, 0], k_means_cluster_centers[k, 1], c='k', marker='x', s=100, linewidths=2)
        plt.title(f'K-Means Clustering (k={nb_clusters})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Return the cluster centers and labels.
    return k_means_cluster_centers, k_means_labels


#######################################################################################################################
# Function: compute_elbow(max_clusters, data)
# Description:
#     This function computes the elbow plot to determine the optimal number of clusters (k) by calculating
#     the distortion for each k. The distortion is the average distance from data points to the nearest cluster
#     center.
#
# Input arguments:
#     - int max_clusters: The maximum number of clusters to evaluate.
#     - np.array data: The feature matrix (shape [n_samples, n_features]) to be clustered.
#
# Output:
#     - None (plots the elbow curve).
#######################################################################################################################

def compute_elbow(max_clusters, data):
    # TODO Step 1: Create an empty list to store distortions and define a range from 1 to max_clusters.
    distortions = []
    cluster_range = range(1, max_clusters + 1)

    # TODO Step 2: Loop over your range/list and run your previously created k_means function,
    #  with different values of n_clusters. Use the returned cluster_centers and the input data,
    #  to compute the distortion and store it in your List.
    #  Use cdist() with metric='euclidean' to compute the distance matrix.
    #  There is a way to compute the distortions in on step, where you might need np.min() and sum().
    for k in cluster_range:
        centers, labels = k_means(nb_clusters=k, data=data, generate_plot=False)
        distances = cdist(data, centers, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        distortion = np.sum(min_distances) / data.shape[0]
        distortions.append(distortion)

    # TODO Step 3: Plot the elbow curve with a clear title and axis labels.
    plt.figure()
    plt.plot(cluster_range, distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return distortions


#######################################################################################################################
# BONUS-TASK
# Function: silhouette_analysis(X, cluster_centers, labels, n_clusters=3)
# Description:
#     This function performs silhouette analysis for a clustering model using the provided feature matrix,
#     cluster centers, and cluster labels. It generates a silhouette plot and a cluster visualization plot
#     to analyze the quality of the clustering.
#
# Input arguments:
#     - np.array X: The feature matrix (shape [n_samples, n_features]) containing the data points.
#     - np.array cluster_centers: Coordinates of the cluster centers.
#     - np.array labels: Cluster labels for each data point.
#     - int n_clusters: The number of clusters (default is 3).
#     - str algorithm_name: Name of the clustering algorithm (default is 'Clustering').
#
# Output:
#     - None (plots the silhouette plot and the cluster visualization plot).
#######################################################################################################################
def silhouette_analysis(data, cluster_centers, labels, n_clusters=3, algorithm_name=None):
    data = np.asarray(data)
    labels = np.asarray(labels)

    # Step 1: Compute average silhouette score and print it.
    score = silhouette_score(data, labels)
    print(f"Average silhouette score ({algorithm_name}):", score)

    # Step 2: Compute silhouette scores for individual samples.
    sample_silhouette_values = silhouette_samples(data, labels)

    # Step 3: Create the silhouette plot.
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        if size_cluster_i == 0:
            continue

        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for {algorithm_name}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=score, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.linspace(-0.1, 1.0, 6))
    plt.tight_layout()
    plt.show()

    # Step 4: Visualize the clusters with their centers.
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(data[:, 0], data[:, 1], marker='o', s=30, lw=0, alpha=0.7, c=colors)
    if cluster_centers is not None and len(cluster_centers) > 0:
        centers_arr = np.asarray(cluster_centers)
        ax2.scatter(centers_arr[:, 0], centers_arr[:, 1], marker='x', c="k", s=100, linewidths=2)
    ax2.set_title(f"{algorithm_name} cluster visualization (avg silhouette = {score:.3f})")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    ##### TODO:Call your sub-tasks/methods here. Uncomment each Task step by step #####
    ### Task 1: Random Plot ###
    plot_random_data(n_samples=2000) # run random data with 2000 samples

    # Create some 2D Data which is needed for the following Algorithms as input.
    x1_axis_data = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
    x2_axis_data = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
    combined_data = np.array(list(zip(x1_axis_data, x2_axis_data))).reshape(len(x1_axis_data), 2)

    ### Task 2: Run your self implemented BSAS Algorithm ###
    bsas_cluster_centers, bsas_labels = BSAS(data=combined_data, max_clusters=3, threshold=1.0)

    ### Task 3: K-means ###
    k_means_cluster_centers, k_means_asigned_labels = k_means(nb_clusters=3, data=combined_data, generate_plot=True)

    ### Task 4: Run the Elbow function with k-Means from 0 until max_clusters for our 2D data ###
    compute_elbow(max_clusters=10, data=combined_data)  #Plot the elbow to determine the no. of clusters.

    ### Task 5 (BONUS): Run the silhouette analysis for k-Means with our 2D data ###
    # You can compute the scores on the results of your BSAB and the k_means.
    silhouette_analysis(data=combined_data, cluster_centers=k_means_cluster_centers, labels=k_means_asigned_labels, n_clusters=3, algorithm_name='K-Means')
    silhouette_analysis(data=combined_data, cluster_centers=bsas_cluster_centers, labels=bsas_labels, n_clusters=3, algorithm_name='BSAS')