### CELL
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

### CELL
# Set file paths and constants
DATASET_PATH = 'path/to/your/dataset.csv'
TARGET_VARIABLE = 'Y'  # Target variable name
FEATURES = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
            'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
            'X20', 'X21', 'X22', 'X23']

### CELL
# Load the dataset
data = pd.read_csv(DATASET_PATH)

### CELL
# Display the first few rows of the dataset
data.head()

### CELL
# Data Description
print(f"Dataset contains {data.shape[0]} instances and {data.shape[1]} features.")
print(f"Features: {data.columns.tolist()}")
print(f"Missing values: \n{data.isnull().sum()}")

### CELL
# Data Cleaning
# Check for duplicates and drop if any
data.drop_duplicates(inplace=True)

# Describe the dataset
data.describe()

### CELL
# Exploratory Data Analysis (EDA)
# Visualize the distribution of the target variable
sns.countplot(x=TARGET_VARIABLE, data=data)
plt.title('Distribution of Default Payments')
plt.show()

### CELL
# Visualize correlations using a heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

### CELL
# Prepare the data for modeling
X = data[FEATURES]
y = data[TARGET_VARIABLE]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### CELL
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### CELL
# Fit the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

### CELL
# Make predictions
y_pred = model.predict(X_test_scaled)

### CELL
# Results and Analysis
# Classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

### CELL
# Discussion and Conclusion
print("The Decision Tree model has been fitted to the data. The classification report indicates the model's precision, recall, and F1-score.")
print("The confusion matrix visualizes the model's performance on the test set.")
print("Considerations for future work include exploring hyperparameter tuning and testing other algorithms for comparison.")

### CELL
# Save the model if needed (optional)
import joblib
joblib.dump(model, 'decision_tree_model.pkl')
