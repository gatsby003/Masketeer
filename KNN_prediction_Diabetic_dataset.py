# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score
cm = confusion_matrix(y_test, y_pred_KNN)
print(cm)
print ('Accuracy of KNN ::',accuracy_score(y_test, y_pred_KNN))

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_KNN)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#Precision Score = TP / (FP + TP)
precision=cm[0][0]/(cm[0][1]+cm[0][0])
print('Precision: %.3f' % precision)
print('Precision using standard function: %.3f' % precision_score(y_test, y_pred_KNN))


#Recall Score = TP / (FN + TP)
recall=cm[0][0]/(cm[1][0]+cm[0][0])
print('Recall Score: %.3f' % recall)
print('Recall using standard function: %.3f' % recall_score(y_test, y_pred_KNN))
