import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load Heart Disease Data Set
data = pd.read_csv('heart_disease.csv')

# Display the data
print(data)

# Define the Bayesian Network structure
model = BayesianNetwork([('age', 'heartdisease'), ('sex', 'heartdisease'), ('cp', 'heartdisease'), 
                         ('restecg', 'heartdisease'), ('exang', 'heartdisease'), ('chol', 'heartdisease')])

# Train the model with Maximum Likelihood Estimator
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference using Variable Elimination
infer = VariableElimination(model)

# Predicting heart disease
print('\n\nPredicting for Heart Disease:')
query = infer.query(variables=['heartdisease'], evidence={'age': 73, 'sex': 0, 'cp': 1, 'restecg': 2, 'exang': 1})
print(query)

# Predicting cholesterol level given heart disease
print('\n\nPredicting for Cholesterol level:')
query = infer.query(variables=['chol'], evidence={'heartdisease': 1})
print(query)
