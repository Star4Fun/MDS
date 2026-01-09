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
    plt.title('Plot Random')
    plt.show()

    return None  # This function does not return any value

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

    # TODO Step 2: Iterate through all other data points to assign them to clusters
    # for i in range(1, data.shape[0]):  # Uncomment to start processing
        # TODO Step 2a: Calculate distance from the current point to each cluster center
        #  HINT: You can use np.linalg.norm to get all distances and np.argmin to get the closest cluster.

        # TODO Step 2b: Check if the point should create a new cluster
        #  If yes, create a new cluster center; otherwise, assign the point to the closest cluster.
        #  Update the cluster center position after assignment.

    # TODO Step 3: Plot the clustered data points and cluster centers
    #  Titles, x and y labels are always nice to have. Display the plot.

    # TODO Step 4: Return cluster centers and labels as numpy arrays.
    return  # bsas_cluster_centers, bsas_labels


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

    # TODO Step 2: Fit the KMeans model to the input data and extract labels and cluster centers.

    # Generate a plot only if generate_plot=True
    # if generate_plot:  # Uncomment to enable plotting
        # TODO Step 3: Create a scatter plot with cluster centers and data points.
        #  Assign colors to points based on their labels using plt.cm.Spectral or similar colormap.
        #  We recommend using the following color map.
        #  colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels)) + 1))[:-1]

    # Return the cluster centers and labels.
    return  # k_means_cluster_centers, k_means_labels


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

    # TODO Step 2: Loop over your range/list and run your previously created k_means function,
    #  with different values of n_clusters. Use the returned cluster_centers and the input data,
    #  to compute the distortion and store it in your List.
    #  Use cdist() with metric='euclidean' to compute the distance matrix.
    #  There is a way to compute the distortions in on step, where you might need np.min() and sum().


    # TODO Step 3: Plot the elbow curve with a clear title and axis labels.
    return


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
#
# Output:
#     - None (plots the silhouette plot and the cluster visualization plot).
#######################################################################################################################
def silhouette_analysis(data, cluster_centers, labels, n_clusters=3):
    # This task is mainly about plotting the right things.
    # TODO Step 1: Compute the average silhouette score and print it.

    # TODO Step 2: Compute silhouette scores for individual samples.

    # TODO Step 3: Create the silhouette plot.

    # TODO Step 4: Visualize the clusters with their centers.
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
    BSAS(data=combined_data, max_clusters=3, threshold=1.0)

    ### Task 3: K-means ###
    #k_means_cluster_centers, k_means_asigned_labels = k_means(nb_clusters=3, data=combined_data, generate_plot=True)

    ### Task 4: Run the Elbow function with k-Means from 0 until max_clusters for our 2D data ###
    #compute_elbow(max_clusters=10, data=combined_data)  #Plot the elbow to determine the no. of clusters.

    ### Task 5 (BONUS): Run the silhouette analysis for k-Means with our 2D data ###
    # You can compute the scores on the results of your BSAB and the k_means.




    
    
    
    
    
    
   


   