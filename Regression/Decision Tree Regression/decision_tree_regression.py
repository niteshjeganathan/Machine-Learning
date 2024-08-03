# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Decision Tree Regression model
# Feature Scaling is not required in Decision Tree Regression model since they work on splits, not equations
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting results
# Decision Tree Regression Model is more suitable for multi-feature datasets
print(regressor.predict([[6.5]]))

# Visualising Decision Tree Regression model with higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show();