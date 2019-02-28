#coding:utf-8
import numpy as np
from sklearn.mixture import GaussianMixture
#生成随机观测点，含有3个聚集核心
X = np.concatenate((np.random.randn(200, 1), 10 + np.random.randn(300, 1), 5 + np.random.randn(100, 1)))
clf = GaussianMixture(n_components=3)
gmm = clf.fit(X)
obs = [[1], [2], [10], [20]]
gmm.predict(obs)
probs = gmm.predict_proba(obs)
print(probs[:4].round(3))