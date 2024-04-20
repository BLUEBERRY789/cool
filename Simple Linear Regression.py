import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data from CSV file
data = pd.read_csv('simple1.csv')

# Extract X and y from the dataset
X = data['X_column'].values.reshape(-1, 1)  # Assuming 'X_column' is the column containing the feature
y = data['y_column'].values.reshape(-1, 1)  # Assuming 'y_column' is the column containing the target variable

# Create a scatter plot of the data
plt.scatter(X, y, color='blue')
plt.title('Data from CSV File')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])  # New data points for prediction
y_pred = model.predict(X_new)

# Plot the linear regression line
plt.scatter(X, y, color='blue')
plt.plot(X_new, y_pred, color='red', linewidth=2)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Print the coefficients
print("Intercept:", model.intercept_[0])
print("Coefficient:", model.coef_[0][0])
