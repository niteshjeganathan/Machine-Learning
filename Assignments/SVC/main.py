# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training SVM Model
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting a new result
print("Prediction for (30, 87000): ", classifier.predict(sc.transform([[30, 87000]])))

# Predicting test results
y_pred = classifier.predict(X_test)
# print(np.concatenate([(y_pred.reshape(len(y_pred),1)), (y_test  .reshape(len(y_test), 1))], 1))

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()
print("Accuracy Scores: ", accuracy_score(y_test, y_pred))