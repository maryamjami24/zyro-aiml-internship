import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

print("All AI/ML libraries imported successfully!")

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create the ML model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
prediction = model.predict([[6]])

print("Prediction for input 6:", prediction[0])
print("Basic ML model ran successfully!")