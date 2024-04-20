import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('lgr2.csv')

# Display the first few rows of the dataset
print("Dataset:")
print(data.head())

# Separate features (X) and target variable (y)
X = data.drop(columns=['target_column'])
y = data['target_column']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the dataset
plt.scatter(X[y == 0]['feature1'], X[y == 0]['feature2'], color='red', label='Class 0')
plt.scatter(X[y == 1]['feature1'], X[y == 1]['feature2'], color='blue', label='Class 1')
plt.title('Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Plot the decision boundary
x_values = np.linspace(X['feature1'].min(), X['feature1'].max(), 100)
y_values = -(model.coef_[0][0] * x_values + model.intercept_[0]) / model.coef_[0][1]
plt.plot(x_values, y_values, color='green', linestyle='--', label='Decision Boundary')
plt.scatter(X[y == 0]['feature1'], X[y == 0]['feature2'], color='red', label='Class 0')
plt.scatter(X[y == 1]['feature1'], X[y == 1]['feature2'], color='blue', label='Class 1')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
