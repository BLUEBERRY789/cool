import pandas as pd

# Load dataset from CSV file
dataset_path = 'sample_dataset.csv'  # Provide the path to your CSV file
df = pd.read_csv(dataset_path)

# Separate features (X) and labels (y)
X = df.drop(columns=['label']).values
y = df['label'].values

# Splitting the data into training and testing sets (if needed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluating the performance of the classifier
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
