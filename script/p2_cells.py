import json

def convert_cells_to_ipynb(cells, filename="output_notebook.ipynb"):
    """
    Converts a list of code and markdown cells into an .ipynb file.

    Args:
        cells (list): List of cells where each cell is a dictionary containing "cell_type" and "source" keys.
        filename (str): The name of the output .ipynb file.

    Returns:
        None
    """

    # Define the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Write to .ipynb file
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=4)
    print(f"Notebook saved as {filename}")

cells = [
    # Cell 1: Import Libraries
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "e0a1c779",
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
            "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.linear_model import LinearRegression\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "import joblib\n"
        ]
    },

    # Cell 2: Markdown - Project Overview and Setup
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Wine Quality Prediction and Clustering Analysis\n",
            "This project aims to analyze and predict the quality of wines based on various physicochemical properties. The dataset consists of red and white wine samples, with several features such as acidity, alcohol content, sugar levels, and pH, along with a target variable indicating the quality of each wine. The goal is to predict wine quality and identify patterns using clustering techniques.\n",
            "\n",
            "### Type of Learning/Algorithm\n",
            "This project employs unsupervised learning for clustering analysis and supervised learning for regression tasks. The following algorithms are used in this analysis:\n",
            "- **Clustering (Unsupervised Learning):** K-Means, Agglomerative Clustering\n",
            "- **Regression (Supervised Learning):** Linear Regression\n",
            "\n",
            "### Type of Task\n",
            "The task associated with this dataset involves predicting wine quality using the following approaches:\n",
            "- **Clustering:** Unsupervised learning models are used to group wines into clusters, and then the clusters are mapped to quality labels based on their most frequent quality scores.\n",
            "- **Regression:** A supervised learning approach using Linear Regression to predict continuous wine quality scores.\n"
        ]
    },

    # Cell 3: Set File Paths and Constants
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "fa3108a2",
        "metadata": {},
        "outputs": [],
        "source": [
            "RED_WINE_PATH = '../data/winequality-red.csv'  # Path to the red wine quality dataset\n",
            "WHITE_WINE_PATH = '../data/winequality-white.csv'  # Path to the white wine quality dataset\n",
            "TARGET_VARIABLE = 'quality'  # Target variable name\n",
            "FEATURES = [\n",
            "    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',\n",
            "    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',\n",
            "    'pH', 'sulphates', 'alcohol',\n",
            "]"
        ]
    },

    # Cell 4: Load the Datasets
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "9b5198bb",
        "metadata": {},
        "outputs": [],
        "source": [
            "red_wine_data = pd.read_csv(RED_WINE_PATH, sep=';')\n",
            "white_wine_data = pd.read_csv(WHITE_WINE_PATH, sep=';')\n",
            "\n",
            "# Add a column to indicate the wine type\n",
            "red_wine_data['wine_type'] = 'red'\n",
            "white_wine_data['wine_type'] = 'white'\n",
            "\n",
            "# Concatenate the datasets\n",
            "data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)\n"
        ]
    },

    # Cell 5: Data Cleaning - Drop Duplicates
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "9d57b8b7",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check for duplicates and drop if any\n",
            "data.drop_duplicates(inplace=True)\n"
        ]
    },

    # Cell 6: Prepare Data for Clustering and Regression
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "ab358f06",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare Data for the Clustering Algorithm\n",
            "# Split the dataset into training and testing sets\n",
            "X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)\n",
            "\n",
            "# Apply standard scaling to features set\n",
            "scaler = StandardScaler()\n",
            "X_train.loc[:, FEATURES] = scaler.fit_transform(X_train.loc[:, FEATURES])\n",
            "X_test.loc[:, FEATURES] = scaler.transform(X_test.loc[:, FEATURES])\n",
            "\n",
            "# Print the shape of the training and testing sets\n",
            "print(f\"Training set shape: {X_train.shape}\")\n",
            "print(f\"Testing set shape: {X_test.shape}\")\n",
            "\n",
            "X_train.head()"
        ]
    },

    # Cell 7: Markdown - K-Means Clustering
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### K-Means Clustering\n",
            "We will first apply K-Means clustering to the data. K-Means is an unsupervised learning algorithm that will allow us to group wines into clusters. The number of clusters (k) will be determined using the **elbow method** based on the inertia score."
        ]
    },

    # Cell 8: K-Means Clustering - Elbow Method
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "92a39f99",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Elbow method to determine optimal k for K-Means clustering\n",
            "inertia = []\n",
            "k_range = range(1, 11)\n",
            "for k in k_range:\n",
            "    kmeans = KMeans(n_clusters=k, random_state=42)\n",
            "    kmeans.fit(X_train[FEATURES])\n",
            "    inertia.append(kmeans.inertia_)\n",
            "\n",
            "# Plot the inertia to visualize the elbow point\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.plot(k_range, inertia, marker='o')\n",
            "plt.title('Elbow Method for Optimal k')\n",
            "plt.xlabel('Number of Clusters (k)')\n",
            "plt.ylabel('Inertia')\n",
            "plt.show()"
        ]
    },

    # Cell 9: K-Means Clustering - Fit the Model
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "df7e61a3",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fit the K-Means model with the optimal number of clusters (e.g., k=4)\n",
            "kmeans = KMeans(n_clusters=4, random_state=42)\n",
            "kmeans.fit(X_train[FEATURES])\n",
            "X_train['cluster'] = kmeans.labels_\n",
            "\n",
            "# Visualize the clusters\n",
            "plt.figure(figsize=(8, 5))\n",
            "sns.scatterplot(x=X_train['alcohol'], y=X_train['fixed acidity'], hue=X_train['cluster'], palette='viridis')\n",
            "plt.title('K-Means Clusters')\n",
            "plt.show()"
        ]
    },

    # Cell 10: Markdown - Linear Regression for Quality Prediction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Linear Regression for Wine Quality Prediction\n",
            "Now, let's use linear regression to predict wine quality based on the features. Linear regression is a supervised learning algorithm that models the relationship between the target variable (wine quality) and the features."
        ]
    },

    # Cell 11: Linear Regression - Fit the Model
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "9f1dbb70",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize and train the Linear Regression model\n",
            "lr = LinearRegression()\n",
            "lr.fit(X_train[FEATURES], X_train[TARGET_VARIABLE])\n",
            "\n",
            "# Predict wine quality on the test set\n",
            "y_pred = lr.predict(X_test[FEATURES])\n",
            "\n",
            "# Evaluate the model\n",
            "mae = mean_absolute_error(X_test[TARGET_VARIABLE], y_pred)\n",
            "print(f'Mean Absolute Error: {mae:.2f}')\n",
            "print('Regression Coefficients:', lr.coef_)\n",
            "print('Intercept:', lr.intercept_)\n"
        ]
    },

    # Cell 12: Save the Trained Models
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "c30fdd12",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save the trained models for future use\n",
            "joblib.dump(kmeans, 'kmeans_model.pkl')\n",
            "joblib.dump(lr, 'linear_regression_model.pkl')\n"
        ]
    },
]

convert_cells_to_ipynb(cells, filename="/home/algorithmspath/ms_ds/intro_ml_2/ml2_project/script/project2.conv.ipynb")
