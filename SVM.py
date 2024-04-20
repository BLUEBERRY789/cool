import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Read data from CSV file
recipes = pd.read_csv('svm1.csv')

# Extract features and labels
features = recipes[['Flour', 'Sugar']].to_numpy()
label = np.where(recipes['Type'] == 'Muffin', 0, 1)

# Train SVM model
model = svm.SVC(kernel='linear')
model.fit(features, label)

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_
yy_down = a * xx + (b[0, 1] - a * b[0, 0])
yy_up = a * xx + (b[-1, 1] - a * b[-1, 0])

# Plot the hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=label, cmap='Set1', s=70)
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(b[:, 0], b[:, 1], 'o', markerfacecolor='white', markeredgecolor='black', markersize=10)
plt.plot(xx, yy_down, 'k--', linewidth=1)
plt.plot(xx, yy_up, 'k--', linewidth=1)
plt.xlabel('Flour')
plt.ylabel('Sugar')
plt.title('SVM Classifier')
plt.show()

# Output the model's parameters
print("SVM Classifier Parameters:")
print(f"Support Vectors: {model.support_vectors_}")
print(f"Number of Support Vectors for Each Class: {model.n_support_}")
print(f"Intercept (b): {model.intercept_}")
print(f"Coefficients (w): {model.coef_}")
