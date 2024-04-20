import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
df = pd.read_csv('synthetic_data.csv')

# Split the data into features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base classifier
base_classifier = DecisionTreeClassifier()

# Number of base classifiers in the ensemble
num_classifiers = 10

# Create an ensemble of decision tree classifiers using bagging
ensemble_model = BaggingClassifier(base_classifier, n_estimators=num_classifiers)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
predictions = ensemble_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
