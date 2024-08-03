# Importing Library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Feature Scaling
sc = StandardScaler()   
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training KNN Model
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30, 87000]])))

# Predicting Test results
y_pred = classifier.predict(X_test)
print(np.concatenate([(y_pred.reshape(len(y_pred), 1)), (y_test.reshape(len(y_test), 1))], 1))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))





