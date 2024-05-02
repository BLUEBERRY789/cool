# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
data = np.random.randn(100, 2)  # 100 samples, 2 features
print(data)

# Initialize PCA and fit the data
pca = PCA(n_components=2)  # We will retain both components for visualization
pca.fit(data)

# Transform the data onto the new feature space
transformed_data = pca.transform(data)

# Plot original data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')

# Plot transformed data (after PCA)
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title('Transformed Data (after PCA)')

plt.show()



#LDA
# Import necessary libraries
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Generate some sample data with two classes
np.random.seed(0)
num_samples = 100
mean_class1 = [1, 1]
mean_class2 = [4, 4]
covariance = [[1, 0.5], [0.5, 1]]
class1_data = np.random.multivariate_normal(mean_class1, covariance, num_samples)
class2_data = np.random.multivariate_normal(mean_class2, covariance, num_samples)
data = np.concatenate((class1_data, class2_data))
labels = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))

# Initialize LDA and fit the data
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(data, labels)

# Transform the data onto the new feature space
transformed_data = lda.transform(data)

# Plot original data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2')
plt.title('Original Data')
plt.legend()

# Plot transformed data (after LDA)
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[labels == 0], np.zeros(num_samples), label='Class 1')
plt.scatter(transformed_data[labels == 1], np.zeros(num_samples), label='Class 2')
plt.title('Transformed Data (after LDA)')
plt.legend()

plt.show()




# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AdaBoost classifier
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train AdaBoost classifier
adaboost_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = adaboost_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


