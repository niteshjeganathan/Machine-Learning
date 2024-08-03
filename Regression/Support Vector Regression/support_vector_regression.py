# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling
# Feature Scaling required since in SVR there are no coefficients to compensate
# Feature Scaling on the dependent variable is also required since they do not belong to the scale
scX = StandardScaler() 
X = scX.fit_transform(X)
scY = StandardScaler()
y = scY.fit_transform(y)

# Training SVR Model
regressor = SVR(kernel='rbf')
regressor.fit(X, y.ravel())

# Predicting a new result
print(scY.inverse_transform(regressor.predict(scX.transform([[6.5]])).reshape(-1, 1)))

# Visualising SVR Results
plt.scatter(scX.inverse_transform(X), scY.inverse_transform(y), color = 'red')
plt.plot(scX.inverse_transform(X), scY.inverse_transform(regressor.predict(X).reshape(-1, 1)))
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

