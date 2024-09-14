# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA

# Importing Dataset
dataset = pd.read_csv('winequality-red.csv', delimiter=';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
# There are no missing values in the dataset
# Assuming there are no significant outliers in the dataset

# Splitting Training and Test Data›
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardisation
ss1 = StandardScaler()
X_train = ss1.fit_transform(X_train)
X_test = ss1.transform(X_test)

ss2 = StandardScaler()
y_train = ss2.fit_transform(y_train)

# Feature Selection
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance: ", explained_variance)

# Linear Model Training 
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)
y_pred_scaled1 = linearRegressor.predict(X_test)
y_pred1 = ss2.inverse_transform(y_pred_scaled1)

# Support Vector Regressor Model Training
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.ravel())
y_pred_scaled2 = svr.predict(X_test)
y_pred_scaled2 = y_pred_scaled2.reshape(-1, 1)
y_pred2 = ss2.inverse_transform(y_pred_scaled2)

# Decision Tree Model Training
dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X_train, y_train.ravel())
y_pred_scaled3 = dt.predict(X_test)
y_pred_scaled3 = y_pred_scaled3.reshape(-1, 1)
y_pred3 = ss2.inverse_transform(y_pred_scaled3)

# Random Forest Model Training
rf = RandomForestRegressor(n_estimators = 15, random_state = 0)
rf.fit(X_train, y_train.ravel())
y_pred_scaled4 = rf.predict(X_test)
y_pred_scaled4 = y_pred_scaled4.reshape(-1, 1)
y_pred4 = ss2.inverse_transform(y_pred_scaled4)

# Accuracy Metrics
print("Mean Absolute Percentage Error (Linear Regression Model):         ", mean_absolute_percentage_error(y_test, y_pred1)*100)
print("Mean Absolute Percentage Error (Support Vector Regression Model): ", mean_absolute_percentage_error(y_test, y_pred2)*100)
print("Mean Absolute Percentage Error (Decision Tree Regression Model):  ", mean_absolute_percentage_error(y_test, y_pred3)*100)
print("Mean Absolute Percentage Error (Random Forest Regression Model):  ", mean_absolute_percentage_error(y_test, y_pred4)*100)