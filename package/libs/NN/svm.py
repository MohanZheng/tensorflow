import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=125)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=2053)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=841)
import numpy as np


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1]]))



from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
''' 
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X[:20])
print(y[:20])
'''
X = np.array([
 [6.4, 3.2, 5.3, 2.3], [5.5, 3.5, 1.3, 0.2], [7.2, 3.2, 6., 1.8], [5., 3.4, 1.6, 0.4], [7.2, 3.6, 6.1, 2.5],
 [5., 3.5, 1.3, 0.3], [6.3, 2.7, 4.9, 1.8], [7.6, 3.,  6.6, 2.1], [5.4, 3.7, 1.5, 0.2], [5.5, 2.4, 3.7, 1.],
 [5., 3.2, 1.2, 0.2], [5.2, 2.7, 3.9, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.3, 5.7, 2.1], [6.1, 2.6, 5.6, 1.4],
 [4.9, 3.,  1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [6.7, 3.1, 4.4, 1.4], [5.1, 3.8, 1.9, 0.4], [5.,  2.,  3.5, 1.],
 [5., 3.4, 1.6, 0.4], [7.2, 3.6, 6.1, 2.5], [5., 3.5, 1.3, 0.3], [6.3, 2.7, 4.9, 1.8], [7.6, 3., 6.6, 2.1],
 [5.4, 3.7, 1.5, 0.2], [5.5, 2.4, 3.7, 1.], [5., 3.2, 1.2, 0.2], [5.2, 2.7, 3.9, 1.4], [7.1, 3.5, 3.1, 3.5],
 [5.4, 3.2, 5.3, 2.3], [3.5, 3.4, 1.3, 0.2], [8.2, 3.2, 6., 1.8], [2., 3.4, 1.6, 0.4], [9.2, 3.6, 6.1, 2.5],
 [4., 3.5, 1.3, 0.3], [4.3, 2.7, 4.9, 1.8], [7.3, 3., 6.6, 2.1], [2.4, 3.7, 1.5, 0.2], [4.5, 2.4, 3.7, 1.]])
y = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,  0, 1, 1, 1, 0,
              1, 0, 1, 1, 1])
X = X.tolist()
y = y.tolist()
#split the data to  7:3
X_train, X_test, y_train, y_test = ts(X, y, test_size=0.3)
# select different type of kernel function and compare the score

# 调参
from sklearn.model_selection import GridSearchCV
import pandas as pd
svr = svm.SVC(random_state=1)
#parameters = {'kernel': ('rbf', 'linear', 'poly', 'sigmoid'), 'degree': [1, 2, 3, 4, 5], 'C': [1, 3, 5, 7, 10],
#              'decision_function_shape': ['ovr', 'ovo'],
#              'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
parameters = {'kernel': ('rbf', 'linear', 'poly', 'sigmoid'), 'C': [1, 3, 5, 7, 10]}
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv', 'w') as f:
    cv_result.to_csv(f)

print(clf.best_estimator_)
print(clf.best_params_)



''' 
clf_linear = svm.SVC(kernel='rbf', C=1, random_state=1)
clf_linear.fit(X_train, y_train)
score_linear = clf_linear.score(X_test, y_test)
print("The score of sigmoid is : %f" % score_linear)
'''

''' 
import report
from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 30
parameters = {'kernel': ('rbf', 'linear', 'poly', 'sigmoid'), 'degree': [1, 2, 3, 4, 5], 'C': [1, 3, 5, 7, 10], 
'decision_function_shape': ['ovr', 'ovo']}
clf_x = svm.SVC()
random_search = RandomizedSearchCV(clf_x, param_distributions=parameters, n_iter=n_iter_search)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
#report(random_search.cv_results_)

clf_poly = svm.SVC(kernel='sigmoid', degree=4, C=10, decision_function_shape='ovo')
clf_poly.fit(X_train, y_train)
score_poly = clf_poly.score(X_test, y_test)
print("The score of rbf is : %f" % score_poly)

'''


