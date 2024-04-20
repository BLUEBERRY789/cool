import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the dataset
data = {
    'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
    'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
    'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
    'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
    'target': [0, 0, 0, 0, 1, 0, 0, 2, 0, 0]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Display the entire dataset
print("Original dataset:")
print(df.head())

# Extract features (X) and target (y)
X = df.drop('target', axis=1).values
y = df['target'].values

# Step 1: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Data Transformation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Display scaled training data
print("\nScaled training data:")
print(pd.DataFrame(data=X_train_scaled, columns=df.columns[:-1]).head())

# Step 4: Display scaled test data
print("\nScaled test data:")
print(pd.DataFrame(data=X_test_scaled, columns=df.columns[:-1]).head())
