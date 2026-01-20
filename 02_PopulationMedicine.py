import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Path and column names for dataset
bc_data_path = './wdbc.csv' # Path to the input data file.
bc_data_columns = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

#######################################################################################################################
# Function correlation_map(dataExceptLabels):
# Function generates a heatmap of correlation
#
# Input arguments:
#   - dataframe dataExceptLabels: pandas dataframe of all attributes except labels
# Output:
#   - None (displays a heatmap of correlation data)
#
# Description:
#   This function calculates the correlation matrix for the given dataframe and generates a heatmap using seaborn.
#   It visually represents the pairwise correlation between attributes.
#######################################################################################################################

def correlation_map(data):
    # TODO Step 1: Calculate the correlation matrix of the dataframe
    #  Hint: Use the `corr()` method on the dataframe.
    corr = data.corr()

    # TODO Step 2: Generate the heatmap plot for the correlation matrix
    #  Please rotate the labels so that they are readable.
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    return None

#######################################################################################################################
# Function find_no_of_clusters(data):
# Function generates a plot to visualize the optimal number of clusters in a dataset.
# Same as we have done in Exercise 1/Elbow but here we are using the functions from sklearn.
# If you want, you can import your Exercise 1 script here and call the compute_elbow function here.
# Instead of distortion, you can also compute the WCSS.
# WCSS: Represents the total squared distance to the centroid. <-- Faster to implement
# Distortion: Represents the average squared distance to the centroid (normalized)
#
# Input arguments:
#   - dataframe data: pandas dataframe of all attributes except labels
# Output:
#   - None (displays a plot of Within-Cluster Sum of Squares (WCSS) for different values of K)
#
# Description:
#   This function applies the K-means algorithm for a range of cluster numbers (K) to calculate WCSS.
#   It then plots WCSS against K to help determine the optimal number of clusters using the elbow method.
#######################################################################################################################

def find_noof_clusters(data):
    # TODO Step 1: Initialize a list to store Within-Cluster Sum of Squares (WCSS) for each value of K
    #  Hint: Use an empty list to store the WCSS values.
    wcss = []

    # TODO Step 2: Loop through K (Number of Clusters) values from 1 to 14
    #  Always run KMeans on your data and save the WCSS in your list.
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # TODO Step 3: Plot your results. The Plot is what you have done in Exercise 1/Elbow,
    #  so you can copy most of it.
    number_of_clusters = range(1, 15)
    plt.figure()
    plt.plot(number_of_clusters, wcss, marker='o')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    #plt.grid(True)
    plt.show()

    return None

#######################################################################################################################
# Function compute_results_Kmeans(data, labels):
# Function computes the number of correct and incorrect classifications after K-means clustering by comparing the
# clustered data labels with the original labels. Additionally, it visualizes the results as a confusion matrix.
#
# Input arguments:
#   - dataframe data: pandas dataframe containing attributes like 'radius_mean' and 'fractal_dimension_mean'.
#   - pandas series labels: pandas series containing the original labels (e.g., 'diagnosis').
# Output:
#   - Prints the total number of incorrect classifications.
#   - Prints the total number of correct classifications.
#   - Prints the percentage of correct classifications.
#   - Displays a confusion matrix plot.
#
# Description:
#   This function scales the input features, applies K-means clustering, and compares the predicted cluster labels
#   with the true labels ('diagnosis') from the dataset to compute accuracy. It also plots a confusion matrix.
#######################################################################################################################
def compute_results_Kmeans(data, labels):
    # TODO Step 1: Initialize a scaler for standardizing the data
    scaler = StandardScaler()

    # TODO Step 2: Initialize K-means clustering with 2 clusters (assuming two classes: 'M' and 'B') --> try out different random states 1 and 4, and check the differences in the results.
    kmeans = KMeans(n_clusters=2, random_state=1)

    # TODO Step 3: Create a pipeline using `make_pipeline()`.
    pipeline = make_pipeline(scaler, kmeans)

    # TODO Step 4: Fit the pipeline to the input data and use it to predict the
    #  labels assigned by the clustering afterwards.
    cluster_labels = pipeline.fit_predict(data)

    # TODO Step 5: Map the original labels ('diagnosis') to numerical values: 'M' -> 0, 'B' -> 1
    #  Hint: Use the `replace()` method.
    true_labels = labels.replace({'M': 0, 'B': 1}).values
    predicted_labels = cluster_labels

    # TODO Step 6: Calculate the confusion matrix with the true_labels and the predicted_labels
    cm = confusion_matrix(true_labels, predicted_labels)

    # TODO Step 7: Plot the confusion matrix
    plt.figure()
    sns.heatmap(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # TODO Step 8: Calculate and print the correct and incorrect predictions
    total_correct = np.trace(cm)
    total_predictions = np.sum(cm)
    total_incorrect = total_predictions - total_correct
    accuracy_percentage = (total_correct / total_predictions) * 100
    print(f"Total Incorrect Classifications: {total_incorrect}")
    print(f"Total Correct Classifications: {total_correct}")
    print(f"Percentage of Correct Classifications: {accuracy_percentage:.2f}%")

    return None

#######################################################################################################################
# Function compute_dendrogram(data):
# Function plots the hierarchical clustering of the provided data using a dendrogram.
#
# Input arguments:
#   - dataframe or array data: pandas dataframe or NumPy array containing attributes for hierarchical clustering.
#
# Output:
#   - A dendrogram plot showing the hierarchical clustering of the data.
#
# Description:
#   This function uses the `linkage` method from `scipy.cluster.hierarchy` to compute hierarchical clustering based on
#   the Ward's method and visualizes it using a dendrogram. It allows us to visualize the hierarchy and relationships
#   between clusters.
#
# Hint:
#   - Use the `linkage` and `dendrogram` functions from `scipy.cluster.hierarchy`.
#######################################################################################################################

def compute_dendrogram(data):
    # TODO Step 1: Compute the linkage matrix using Ward's method
    #  Hint: Use the `linkage()` function with `method='ward'`.
    linkage_matrix = linkage(data, method='ward')

    # TODO Step 2: Create a dendrogram to visualize the hierarchical clustering
    #  Hint: Use the `dendrogram()` function from scipy.
    plt.figure()
    dendrogram(linkage_matrix)

    # TODO Step 3: Add axis labels for clarity and plot the dendrogram
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

    return None

#######################################################################################################################
# Function compute_results_Agglomerative(data, labels):
# Function performs Agglomerative Clustering on the provided data and compares the results with the ground truth.
#
# Input arguments:
#   - dataframe or array data: pandas dataframe or NumPy array containing attributes for clustering.
#   - pandas series labels: Original labels (e.g., 'diagnosis') as a pandas series.
#
# Output:
#   - Prints the total number of incorrect classifications.
#   - Prints the total number of correct classifications.
#   - Prints the percentage of correct classifications.
#   - Shows a confusion matrix plot.
#   - Shows a scatter plot of 'radius_mean' vs 'fractal_dimension_mean' after clustering.
#
# Description:
#   This function applies Agglomerative Clustering to the provided data, standardizes it using a pipeline,
#   and compares the clustering results to the ground truth using a confusion matrix.
#######################################################################################################################

def compute_results_Agglomerative(data, labels):
    # TODO Step 1: Initialize a scaler for standardizing the data
    #  Hint: Standardize the data to have mean 0 and standard deviation 1.
    scaler = StandardScaler()

    # TODO Step 2: Initialize Agglomerative Clustering with 2 clusters and Ward's linkage
    agglomerative = AgglomerativeClustering(n_clusters=2, linkage='ward')

    # TODO Step 3: Create a pipeline again using `make_pipeline()`.
    pipeline = make_pipeline(scaler, agglomerative)

    # TODO Step 4: Fit the pipeline to the input data and predict the cluster-assigned labels.
    #  Hint: Use `fit_predict()` instead of `predict()`.
    cluster_labels = pipeline.fit_predict(data)

    # TODO Step 5: Map the original labels ('diagnosis') to numerical values: 'M' -> 0, 'B' -> 1
    true_labels = labels.replace({'M': 0, 'B': 1}).values
    predicted_labels = cluster_labels

    # TODO Step 6: Copy your code from KMeans to calculate the confusion matrix,
    #  plot it, and calculate and print the correct and incorrect predictions.
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure()
    sns.heatmap(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    total_correct = np.trace(cm)
    total_predictions = np.sum(cm)
    total_incorrect = total_predictions - total_correct
    accuracy_percentage = (total_correct / total_predictions) * 100
    print(f"Total Incorrect Classifications: {total_incorrect}")
    print(f"Total Correct Classifications: {total_correct}")
    print(f"Percentage of Correct Classifications: {accuracy_percentage:.2f}%")

    # Step 7: Visualize clustering results --> We did this for you!
    if 'radius_mean' in data.columns and 'fractal_dimension_mean' in data.columns:
        # Add cluster assignments to the dataframe
        data["cluster"] = cluster_labels

        # Plot clustering results
        plt.figure(figsize=(15, 10))
        plt.scatter(
            data["radius_mean"][data["cluster"] == 0],
            data["fractal_dimension_mean"][data["cluster"] == 0],
            color="red",
            label="Cluster 0"
        )
        plt.scatter(
            data["radius_mean"][data["cluster"] == 1],
            data["fractal_dimension_mean"][data["cluster"] == 1],
            color="blue",
            label="Cluster 1"
        )
        plt.xlabel("radius_mean")
        plt.ylabel("fractal_dimension_mean")
        plt.title("Scatter Plot of Clustering Results")
        plt.legend()
        plt.show()
    else:
        print("Scatter plot cannot be generated as 'radius_mean' and 'fractal_dimension_mean' are not in the data.")

    return None

if __name__ == '__main__':
    ##### TODO: Call your sub-tasks/methods here #####
    ### Data Preparation ### <-- We did this for you!

    data = pd.read_csv(bc_data_path, names=bc_data_columns).fillna(0.0)  # Load the dataset
    data.drop(["id"], axis=1, inplace=True)  # Drop the "id" column as we don't need it
    print("Label counts (diagnosis):")  # Print information about the occurrences of each label in the 'diagnosis' column
    labels = data["diagnosis"]  # Separate labels for evaluation task later on
    print(labels.value_counts())  # Print the label counts

    dataExceptLabels = data.drop(["diagnosis"], axis=1)  # Drop the label/diagnosis, since we want to predict it
    print("\nFeatures (dataExceptLabels):")  # Print information about the separated data
    print(dataExceptLabels.info())  # Print summary info for the features dataframe

    ### Task 1: Random plot ###
    #correlation_map(data=dataExceptLabels)

    ### Task 2: Find number of clusters ###
    #find_noof_clusters(data=dataExceptLabels)

    ### Task 3: Use K-Means to cluster data points and compare it with the ground truth ###
    # Task 3.1: Run it only with 2 features: ['radius_mean','fractal_dimension_mean']
    #data_two_features = data.loc[:, ['radius_mean', 'fractal_dimension_mean']]
    #compute_results_Kmeans(data=data_two_features, labels=labels)
    # Task 3.2: Run it with the whole data (all features)
    #compute_results_Kmeans(data=dataExceptLabels, labels=labels)

    ### Task 4: Compute a Dendrogram ###
    #compute_dendrogram(data=dataExceptLabels)

    ### Task 5: Run Agglomerative Clustering to cluster data points and compare it with the ground truth  ###
    compute_results_Agglomerative(data=dataExceptLabels, labels=labels)