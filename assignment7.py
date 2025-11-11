import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state = 42)

# let us train svm with three different kernels
linear_svc = svm.SVC(kernel='linear')
polynomial_svc = svm.SVC(kernel='poly', degree=3)
rbf_svc = svm.SVC(kernel='rbf')

# now let us try fitting the data to each
linear_svc.fit(X_train, y_train)
polynomial_svc.fit(X_train, y_train)
rbf_svc.fit(X_train, y_train)

linear_predict = linear_svc.predict(X_test)
polynomial_predict = polynomial_svc.predict(X_test)
rbf_predict = rbf_svc.predict(X_test)

print('Accuracy of the linear model', accuracy_score(y_test, linear_predict))
print('Accuracy of the polynomial model', accuracy_score(y_test, polynomial_predict))
print('Accuracy of the rbf model', accuracy_score(y_test, rbf_predict))

print('Precision of the linear model', precision_score(y_test, linear_predict, average='macro'))
print('Precision of the polynomial model', precision_score(y_test, polynomial_predict, average='macro'))
print('Precision of the rbf model', precision_score(y_test, rbf_predict, average='macro'))

print('Recall of the linear model', recall_score(y_test, linear_predict, average='macro'))
print('Recall of the polynomial model', recall_score(y_test, polynomial_predict, average='macro'))
print('Recall of the rbf model', recall_score(y_test, rbf_predict, average='macro'))

print('F1 Score of the linear model', f1_score(y_test, linear_predict, average='macro'))
print('F1 Score of the polynomial model', f1_score(y_test, polynomial_predict, average='macro'))
print('F1 Score of the rbf model', f1_score(y_test, rbf_predict, average='macro'))

print('Confusion Matrix of the linear model', confusion_matrix(y_test, linear_predict))
print('Confusion Matrix of the polynomial model', confusion_matrix(y_test, polynomial_predict))
print('Confusion Matrix of the rbf model', confusion_matrix(y_test, rbf_predict))

# first let us train a rbf kernel without feature scaling
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state = 42) 
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, y_train)
rbf_predict_train = rbf_svc.predict(X_train)
rbf_predict_test = rbf_svc.predict(X_test)
print('Train Accuracy of the rbf model for breast cancer dataset without feature scaling', accuracy_score(y_train, rbf_predict_train))
print('Test Accuracy of the rbf model for breast cancer dataset without feature scaling', accuracy_score(y_test, rbf_predict_test))

# now let us perform some feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state = 42) 
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, y_train)
rbf_predict_train = rbf_svc.predict(X_train)
rbf_predict_test = rbf_svc.predict(X_test)
print('Train Accuracy of the rbf model for breast cancer dataset after feature scaling', accuracy_score(y_train, rbf_predict_train))
print('Test Accuracy of the rbf model for breast cancer dataset after feature scaling', accuracy_score(y_test, rbf_predict_test))

