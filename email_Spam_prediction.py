# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('emails.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
'''
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# Training the K-NN model on the Training set
print("\n***************KNN********************* ")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting a new result
#print("Predicted Salary for [45,22000]::",classifier.predict(sc.transform([[45,22000]])))

# Predicting the Test set results
y_pred_KNN = classifier.predict(X_test)
print(np.concatenate((y_pred_KNN.reshape(len(y_pred_KNN),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_KNN)
print(cm)
print ('Accuracy of KNN ::',accuracy_score(y_test, y_pred_KNN))

######SVM
# Training the SVM model on the Training set
print("\n***************SVM********************* ")

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_SVC = classifier.predict(X_test)
print(np.concatenate((y_pred_SVC.reshape(len(y_pred_SVC),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_SVC)
print(cm)
print ('Accuracy of KNN ::',accuracy_score(y_test, y_pred_SVC))

'''
***************KNN********************* 
[[1 0]
 [0 0]
 [0 0]
 ...
 [1 0]
 [0 0]
 [1 1]]
[[735 194]
 [ 23 341]]
Accuracy of KNN :: 0.8321732405259087

***************SVM********************* 
[[0 0]
 [0 0]
 [0 0]
 ...
 [0 0]
 [0 0]
 [1 1]]
[[890  39]
 [ 26 338]]
Accuracy of KNN :: 0.9497293116782676

'''