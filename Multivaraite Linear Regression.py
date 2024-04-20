import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

# Load the data from a CSV file
file_path = 'book1.csv'
df = pd.read_csv(file_path)

# Extract features (X) and target variable (y)
X = df[['X1', 'X2']].values
y = df['y'].values

# Train a multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict the target variable
y_pred = model.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Plot the data points and the plane predicted by the model
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data points')

# Adjust the meshgrid
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20), np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
y_plane = model.intercept_ + model.coef_[0] * x1 + model.coef_[1] * x2

# Plot the plane predicted by the model
ax.plot_surface(x1, x2, y_plane, alpha=0.5, color='red', label='Regression plane')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Multivariate Linear Regression')
ax.legend()

plt.show()
