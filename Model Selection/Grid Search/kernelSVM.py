# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Kernel SVM Model
classifier = SVC(kernel='rbf', random_state= 0)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30, 87000]])))

# Predicting test results
y_pred = classifier.predict(X_test)
print(np.concatenate([(y_pred.reshape(len(y_pred), 1)), (y_test.reshape(len(y_test), 1))], 1))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Applying K Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())

# Applying Grid Search
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']}, 
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters, 
                           scoring='accuracy', 
                           cv=10, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print("Best Accuracy: ", best_accuracy*100)
print("Best Parameters", best_params)