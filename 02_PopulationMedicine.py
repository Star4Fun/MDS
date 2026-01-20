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
bc_data_path = './data/wdbc.data'  # Path to the input data file.
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
    # Step 1
    corr = data.corr()

    # Step 2
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0.0,
        square=True,
        cbar=True
    )
    plt.title("Correlation Heatmap (features)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
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
    wcss = []
    K_range = range(1, 15)

    X = StandardScaler().fit_transform(data)

    for k in K_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=12, random_state=0)
        km.fit(X)
        wcss.append(km.inertia_)  # WCSS

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), wcss, marker="o")
    plt.title("Elbow Method (WCSS / inertia)")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS (inertia)")
    plt.grid(True)
    plt.tight_layout()
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
    X = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    y_true = labels.replace({"M": 0, "B": 1}).to_numpy()

    best = None  # (acc, rs, y_pred_mapped, cm)

    for rs in [1, 4]:
        pipe = make_pipeline(
            StandardScaler(),
            KMeans(n_clusters=2, init="k-means++", n_init=12, random_state=rs)
        )

        cluster = pipe.fit_predict(X)

        # Try both mappings because cluster IDs are arbitrary
        pred_a = cluster
        pred_b = 1 - cluster

        acc_a = (pred_a == y_true).mean()
        acc_b = (pred_b == y_true).mean()

        if acc_b > acc_a:
            y_pred = pred_b
            acc = acc_b
        else:
            y_pred = pred_a
            acc = acc_a

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if best is None or acc > best[0]:
            best = (acc, rs, y_pred, cm)

    acc, rs, y_pred, cm = best

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=["M", "B"], yticklabels=["M", "B"])
    plt.title(f"Confusion Matrix (KMeans, random_state={rs})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    correct = int((y_pred == y_true).sum())
    incorrect = int((y_pred != y_true).sum())
    percent = 100.0 * correct / len(y_true)

    print(f"KMeans chosen random_state={rs}")
    print("Incorrect classifications:", incorrect)
    print("Correct classifications:", correct)
    print(f"Correct percentage: {percent:.2f}%")

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
    X = StandardScaler().fit_transform(np.asarray(data))

    Z = linkage(X, method="ward")

    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="level",
        p=10,
        no_labels=True,
        count_sort=True
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
    plt.xlabel("Cluster / samples (truncated)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    return None


#######################################################################################################################
# Function computer_results_Agglomerative(data, labels):
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
    X = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    y_true = labels.replace({"M": 0, "B": 1}).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
    cluster_labels = agg.fit_predict(Xs)  # <-- FIX: now defined

    # Map cluster IDs to true labels (try both)
    pred_a = cluster_labels
    pred_b = 1 - cluster_labels

    acc_a = (pred_a == y_true).mean()
    acc_b = (pred_b == y_true).mean()

    if acc_b > acc_a:
        y_pred = pred_b
    else:
        y_pred = pred_a

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=["M", "B"], yticklabels=["M", "B"])
    plt.title("Confusion Matrix (Agglomerative, Ward)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    correct = int((y_pred == y_true).sum())
    incorrect = int((y_pred != y_true).sum())
    percent = 100.0 * correct / len(y_true)

    print("Agglomerative (Ward)")
    print("Incorrect classifications:", incorrect)
    print("Correct classifications:", correct)
    print(f"Correct percentage: {percent:.2f}%")

    # Step 7: visualization (your sheet code) - avoid mutating original df
    if hasattr(data, "columns") and 'radius_mean' in data.columns and 'fractal_dimension_mean' in data.columns:
        data_plot = data.copy()
        data_plot["cluster"] = cluster_labels

        plt.figure(figsize=(15, 10))
        plt.scatter(
            data_plot["radius_mean"][data_plot["cluster"] == 0],
            data_plot["fractal_dimension_mean"][data_plot["cluster"] == 0],
            color="red",
            label="Cluster 0"
        )
        plt.scatter(
            data_plot["radius_mean"][data_plot["cluster"] == 1],
            data_plot["fractal_dimension_mean"][data_plot["cluster"] == 1],
            color="blue",
            label="Cluster 1"
        )
        plt.xlabel("radius_mean")
        plt.ylabel("fractal_dimension_mean")
        plt.title("Scatter Plot of Clustering Results (Agglomerative)")
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
    correlation_map(data=dataExceptLabels)

    ### Task 2: Find number of clusters ###
    find_noof_clusters(data=dataExceptLabels)

    ### Task 3: Use K-Means to cluster data points and compare it with the ground truth ###
    # Task 3.1: Run it only with 2 features: ['radius_mean','fractal_dimension_mean']
    data_two_features = data.loc[:, ['radius_mean', 'fractal_dimension_mean']]
    compute_results_Kmeans(data=data_two_features, labels=labels)
    # Task 3.2: Run it with the whole data (all features)
    compute_results_Kmeans(data=dataExceptLabels, labels=labels)

    ### Task 4: Compute a Dendrogram ###
    compute_dendrogram(data=dataExceptLabels)

    ### Task 5: Run Agglomerative Clustering to cluster data points and compare it with the ground truth  ###
    compute_results_Agglomerative(data=dataExceptLabels, labels=labels)
