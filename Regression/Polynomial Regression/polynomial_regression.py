# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Linear Regression Model
lRegressor = LinearRegression()
lRegressor.fit(X, y)

# Training Polinomial Regression Model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pRegressor = LinearRegression()
pRegressor.fit(X_poly, y)


# Visualising the Linear Regression Model results
plt.scatter(X, y, color = 'red')
plt.plot(X, lRegressor.predict(X), color = 'blue')
plt.title('Truth or Bluff ( Linear Regression Model )')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Model results
plt.scatter(X, y, color = 'red')
plt.plot(X, pRegressor.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression Model)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
print(lRegressor.predict([[6.5]]))
 
# Predicting a new result with Polynomial Regression
print(pRegressor.predict(poly_reg.fit_transform([[6.5]]))) 






