import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('dataset.csv')

# Convert categorical variables into numerical using one-hot encoding
data = pd.get_dummies(data)

# Split data into features and target
X = data.drop(columns=['target_yes', 'target_no'])  # Drop one of the target columns
y = data['target_yes']  # Use one of the target columns as target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
clf = DecisionTreeClassifier()

# Train decision tree classifier
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Now, let's classify a new sample
# Get the feature names used during training
feature_names = X_train.columns

# Create the new sample with correct feature names
new_sample = pd.DataFrame({'age_group_<25': [1], 'age_group_25-40': [0], 'age_group_>40': [0],
                           'gender_F': [0], 'gender_M': [1]}, columns=feature_names)
prediction = clf.predict(new_sample)
print("Predicted class for new sample:", prediction[0])
