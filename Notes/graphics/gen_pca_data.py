import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set a random seed for reproducibility
np.random.seed(123)

# Generate 100 data points in R^2 with decent correlation
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n)  # Linear relationship with some noise

x_mean = sum(x)/n
y_mean = sum(y)/n

# Create a DataFrame with 0 mean
data = pd.DataFrame({
    'x': [z - x_mean for z in x], 
    'y': [z - y_mean for z in y]}
)

# Export the data to a CSV file
data.to_csv('pcadata.csv', index=False)

model = sm.OLS(data["y"], sm.add_constant(data["x"])) 
results = model.fit()
print(results.summary(slim=True))


X = np.array([data["x"], data["y"]])
C_X = (X @ X.T)/len(data)

E = np.linalg.eig(C_X)
P = - np.array([[0, 1], [1, 0]]) @ (E.eigenvectors).T

Y = P @ X

# Export the data to another CSV file
p_data = pd.DataFrame(Y.T)
p_data.to_csv('pcadata_rotated.csv', index=False)
