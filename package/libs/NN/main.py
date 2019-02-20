import warnings
warnings.filterwarnings("ignore")

import scipy.io as scio
offline_data = scio.loadmat('offline_data_random.mat')
online_data = scio.loadmat('online_data.mat')
offline_location, offline_rss = offline_data['offline_location'], offline_data['offline_rss']
offline_location
trace, rss = online_data['trace'][0:1000, :], online_data['rss'][0:1000, :]
del offline_data
del online_data

import numpy as np
# 定位准确度定义
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

# SVM分类
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

''' 
svr = svm.SVR()
parameters = {'kernel': ('rbf', 'linear'), 'C': [1, 5, 10]}
#, 'gamma': [0, 0.1, 0.2], 'degree': [1, 2, 3, 4]

clf_x = GridSearchCV(svr, parameters)
clf_y = GridSearchCV(svr, parameters)

clf_x.fit(offline_rss, offline_location[:, 0])
clf_y.fit(offline_rss, offline_location[:, 1])
print(clf_x.best_estimator_)
print(clf_y.best_estimator_)
'''



# 随机收索
import report
import re
n_iter_search = 20
parameters = {'multi_class': ['ovr', 'ovo'], 'kernel': ('rbf', 'poly'), 'degree': [1, 2, 3], 'C': [1, 5, 10]}
clf_x = svm.SVC()
random_search = RandomizedSearchCV(clf_x, param_distributions=parameters,
                                   n_iter=n_iter_search)
random_search.fit(offline_rss, offline_location[:, 0])
report(random_search.cv_results_)

''' 
x = clf_x.predict(rss)
y = clf_y.predict(rss)
predictions = np.column_stack((x, y))
acc = accuracy(predictions, trace)
print("支持向量机accuracy: ", acc/100, "m")
acc_of_each_point = np.sqrt(np.sum((predictions - trace)**2, 1))
acc_max = np.max(acc_of_each_point)
acc_mean = np.mean(acc_of_each_point)
print("支持向量机误差最大值: ", acc_max/100, '误差平均值：', acc_mean/100)
print('误差矩阵序列', acc_of_each_point)
'''

