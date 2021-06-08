import matplotlib.pyplot as plt
import numpy as np
from distutils.version import StrictVersion
import pandas as pd
import seaborn as sns
import sys
import warnings


plt.close('all')


warnings.filterwarnings('ignore')
#%matplotlib inline
#%pylab inline

#Populating the interactive namespace from numpy and matplotlib

n = 100
vmin = 0; vmax = 10
x1 =  np.random.uniform(vmin, vmax, n)
x2 =  np.random.uniform(vmin, vmax, n)
x3 =  np.random.uniform(vmin, vmax, n)


fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

ax.hist(x1, alpha=0.6, edgecolor='black', lw=1, bins=np.arange(0, 11, 1))

ax.set_xlabel('X1')
ax.set_ylabel('n samples')

# fig.savefig('ML_0512.png', bbox_inches='tight')
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, x3, c="b", marker="o")


#ax.plot_wireframe((3,4,4,3,3, 3,4,4,3,3, 3,3,4,4,4,4),
#                  (5,5,6,6,5, 5,5,6,6,5, 6,6,6,6,5,5),
#                  (2,2,2,2,2, 3,3,3,3,3, 3,2,2,3,3,2),
#                  color='red', rstride=1, cstride=1, alpha=0.9)

ax.set_xticks(np.arange(0, 11, 1) )
ax.set_yticks(np.arange(0, 11, 1) )
ax.set_zticks(np.arange(0, 11, 1) )
ax.grid(color='k', linestyle='-', linewidth=1, alpha=0.6)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

# fig.savefig('ML_0514.png', bbox_inches='tight')
plt.show()
# plt.clf()








##########

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_plot = np.linspace(-2, 2, 100)
n_samples = 200
X = np.random.uniform(-2, 2, n_samples)
y = X**3 + np.random.uniform(0, 2, n_samples)

print(X.shape, y.shape)
#plt.scatter(X, y, color='navy', s=30, marker='o', label="training points")

plt.xlabel('x')
plt.ylabel('y')

fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(X, y, color='navy', s=30, marker='o', label="training points")

poly_model = make_pipeline(PolynomialFeatures(3),
                           LinearRegression())
poly_model.fit(X[:, np.newaxis], y)
y_plot = poly_model.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, y_plot, lw=2, color="red")
plt.ylim(-12, 12)

plt.xlabel('x')
plt.ylabel('y')

# fig.savefig('ML_0509.png', bbox_inches='tight')
plt.show()
# plt.clf()


from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

X = iris.data
y = iris.target
print(X.shape, y.shape)

svc = SVC(kernel='linear', C=1E0)
print(X.shape, y.shape)

n_classes = 3
colors = 'byr'
CMAP = colors
plot_step = 0.01

# Add random noise
rns = np.random.RandomState(12)
#noise1 = rns.lognormal(mean=1, size=(len(X), 1))
noise2 = rns.uniform(0, 6, size=(len(X), 1))

#X_noise = np.hstack([X, noise1])
X_noise = np.hstack([X, noise2])

Y_feature_names = iris.feature_names
Y_target_names = iris.target_names
Y_feature_names = np.append(Y_feature_names, 'noise1')
#Y_feature_names = np.append(Y_feature_names, 'noise2')
Y_target_names = np.append(Y_target_names, 'noise1')
#Y_target_names = np.append(Y_target_names, 'noise2')


#fig = plt.figure(1, figsize=(9, 16))
fig = plt.figure(1, figsize=(16, 9))

BINS = []
BINS.append(np.arange(4, 8, 0.1))
BINS.append(np.arange(2, 5, 0.1))
BINS.append(np.arange(1, 7, 0.1))
BINS.append(np.arange(0, 3, 0.1))
BINS.append(np.arange(0, 6, 0.1))
#BINS.append(np.arange(0, 6, 0.1))
for fid in range(4):
#for fid in range(5):
    X = X_noise[:, fid]
    y = iris.target

    #plt.subplot(3, 2, fid + 1)
    plt.subplot(2, 2, fid + 1)

    plt.xlabel(Y_feature_names[fid])
    plt.ylabel('n examples')
    plt.axis("tight")

    for i, color in zip(range(n_classes), colors):
        idx = np.where(y == i)
        clf = svc.fit(X.reshape([150,1]), y)
        print(clf.score(X.reshape([150,1]), y))
        plt.hist(X[idx], alpha=0.6, color=color, edgecolor='black', lw=1, label=Y_target_names[i],
                 bins=BINS[fid])
        if fid==3:
            plt.legend(loc='upper right')

    plt.axis("tight")

plt.show()


from scipy.stats import pearsonr

Y_feature_names = iris.feature_names
#Y_target_names = iris.target_names
#Y_feature_names = np.append(Y_feature_names, 'noise1')
#Y_feature_names = np.append(Y_feature_names, 'noise2')
#Y_target_names = np.append(Y_target_names, 'noise1')
#Y_target_names = np.append(Y_target_names, 'noise2')


n_classes = 3
colors = 'byr'
CMAP = colors
plot_step = 0.01

#____________________________________________________________________
fig = plt.figure(5, figsize=(10, 5))

pos = [[6.2, 4.2], [4.5, 6.5], [7, 0.5], [3.5, 3], [3.5, 1], [5, 0.5]]

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    plt.subplot(2, 3, pairidx + 1)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    for i, color in zip(range(n_classes), colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, edgecolor='black', lw=2, label=iris.target_names[i],
                    cmap=CMAP)

    r = "r = " + str(round(pearsonr(X[:, 0], X[:, 1])[0], 3))
    plt.text(pos[pairidx][0], pos[pairidx][1], r)
    plt.axis("tight")
    plt.axis("tight")

plt.legend(loc='upper left')
plt.show()







from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X, y)
print(clf.feature_importances_)

selector = SelectFromModel(clf, threshold=0.02)
X_sel = selector.fit_transform(X, y)
print(selector.get_support())

svc = SVC(kernel='linear')
clf = svc.fit(X_sel, y)
print(clf.score(X_sel, y))
