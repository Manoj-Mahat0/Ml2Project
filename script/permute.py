# ### Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import heapq
import itertools
from sklearn.model_selection import train_test_split

# ### Set File Paths and Constants
RED_WINE_PATH = '../data/winequality-red.csv'  # Path to the red wine quality dataset
WHITE_WINE_PATH = '../data/winequality-white.csv'  # Path to the white wine quality dataset
TARGET_VARIABLE = 'quality'  # Target variable name
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol',
    # 'quality', 'wine_type'
] # Features for clustering

# ### Load the Datasets
red_wine_data = pd.read_csv(RED_WINE_PATH, sep=';')
white_wine_data = pd.read_csv(WHITE_WINE_PATH, sep=';')

# Add a column to indicate the wine type
red_wine_data['wine_type'] = 'red'
white_wine_data['wine_type'] = 'white'

# Concatenate the datasets
data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# ### Data Cleaning
# Check for duplicates and drop if any
data.drop_duplicates(inplace=True)


def compute_cluster_options(df, cluster_col, target_col):
    """
    Compute cluster options based on the DataFrame.

    df: DataFrame containing the clusters and true labels
    cluster_col: column name for cluster labels
    target_col: column name for true labels

    Returns a dictionary where keys are cluster labels and values are lists of possible true labels.
    """
    cluster_options = {}

    # Group by cluster labels and collect unique target labels
    for cluster, group in df.groupby(cluster_col):
        # print(cluster, group)
        # unique_labels = group[target_col].unique()
        unique_labels, unique_counts = np.unique(group[target_col], return_counts=True)
        cluster_options[int(cluster)] = sorted(
            list(zip(unique_labels, unique_counts)), key=lambda x: -x[1]
        )

    return cluster_options

def label_permute_compare(y_labels, clusters, cluster_options):
    """
    y_labels: labels dataframe object
    clusters: clustering label prediction output
    cluster_options: (cluster) => list( label, count in cluster ) as seen in training data
    Returns the best permuted label order and its corresponding accuracy.
    Example output: ({3: 0, 4: 1, 1: 2, 2: 3}, 0.74)
    """
    y_np_labels = y_labels.to_numpy().flatten()

    # Get unique true labels and their counts
    true_labels, true_counts = np.unique(y_np_labels, return_counts=True)
    f_map = {
        value : count
        for (value, count) in zip(true_labels, true_counts)
    }

    # Get unique predicted labels and their counts
    pred_labels, pred_counts = np.unique(clusters, return_counts=True)
    cluster_counts = {
        value : count
        for (value, count) in zip(pred_labels, pred_counts)
    }
    # print(cluster_options)
    assert sum(pred_counts) == sum(true_counts)

    # Generate permutations of the predicted labels
    best_accuracy = 0
    best_mapping = {}

    # Greedy matching algorithm,
    for (cluster, label_counts) in cluster_options.items():
        best_mapping[cluster] = label_counts[0][0]
        best_accuracy += label_counts[0][1]

    best_accuracy /= len(y_np_labels)
    # print(f_map, best_mapping, best_accuracy)
    return best_mapping, best_accuracy


# ### Define the WineQualityClustering Class
class WineQualityClustering:
    def __init__(self, n_clusters, features, target,
            metric='euclidean', linkage='ward'):
        self.features = features
        self.target = target
        # self.model = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=linkage)
        self.model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.mapping = {}

    def fit(self, df):
        """ Fit the clustering model. """
        df['cluster'] = self.model.fit_predict( df[self.features])
        # print(df['cluster'].isna().sum())

        cluster_options = compute_cluster_options(df, 'cluster', self.target)
        # print(cluster_options)

        best_mapping, best_accuracy = label_permute_compare(df[self.target], df['cluster'], cluster_options)
        self.mapping = best_mapping
        # print(best_accuracy)

        return df['cluster'].map(self.mapping)

    def predict(self, df):
        clusters = pd.Series( self.model.predict( df[self.features] ) )
        return clusters.map(self.mapping)

    def evaluate(self, df):
        """ Evaluate the clustering performance using confusion matrix and classification report. """
        y_true = df[self.target]
        y_pred = self.predict(df)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        pct_error = (mae / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else None  # Avoid division by zero
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Return results as an object (dictionary)
        return {
            'accuracy': accuracy,
            'mae': mae,
            'pct_error': pct_error,
            'conf_matrix': conf_matrix
        }

    def display(self, df):
        evaluation_results = w.evaluate(df)

        # Print evaluation statistics
        print("Evaluation Results:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Mean Absolute Error: {evaluation_results['mae']:.4f}")
        print(f"Percentage Error: {evaluation_results['pct_error']:.2f}%")

        # Display confusion matrix
        conf_matrix = evaluation_results['conf_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(df[self.target]),
                    yticklabels=np.unique(df[self.target]))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Quality')
        plt.ylabel('Actual Quality')
        plt.show()

# ### Prepare Data for the Clustering Algorithm
# Split the dataset into training and testing sets
X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)

# Print the shape of the training and testing sets
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

n_clusters = 500
w = WineQualityClustering(n_clusters, FEATURES, TARGET_VARIABLE)
# C = pd.Series( w.model.fit_predict(X_train[w.features]) )
# print(C.unique())
# print(C.isnull().sum())

# print( w.fit(X_train) )
# print( X_train[TARGET_VARIABLE] )
# print( X_train['cluster'] )
# print( w.mapping )
# print( X_train[TARGET_VARIABLE] )
print( accuracy_score( w.fit(X_train), X_train[TARGET_VARIABLE] ) )
