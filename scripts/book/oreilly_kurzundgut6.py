import matplotlib.pyplot as plt
import numpy as np
from distutils.version import StrictVersion
import pandas as pd
import seaborn as sns
import sys
import warnings

plt.close('all')


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8, stratify=y)



from sklearn.svm import SVC
svc = SVC(kernel='rbf')
clf = svc.fit(X_train, y_train)

y_pred = clf.predict(X_test)



from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_norm)


sns.heatmap(cm_norm, annot=True, cmap="YlGnBu")
figure = plt.gcf()
ax = figure.add_subplot(111)
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')





####################################
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=3, target_names=iris.target_names))



#########################################



######### PRECISION RECALL


from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

colors = ['blue', 'orange', 'red']

iris = load_iris()

X = iris.data
y = iris.target

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

#random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 500 * n_features)]

#____________________________________________________________________
# training test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=19)


one_vs_rest = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma=100, probability=True, random_state=12))
y_score = one_vs_rest.fit(X_train, y_train).decision_function(X_test)
y_true, y_pred = y_test, one_vs_rest.predict(X_test)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

#____________________________________________________________________
# Plot Precision-Recall curve for each class
fig = plt.figure(1, figsize=(9, 7))
plt.clf()
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-recall curve of {0} (area = {1:0.3f})'
                   ''.format(iris.target_names[i], average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc="lower right")
plt.show()
# fig.savefig('ML_0604.png', bbox_inches='tight')
# plt.clf()





####################################
from sklearn.metrics import classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

svc = SVC(kernel='rbf', C=1, gamma=1)
clf = svc.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred, digits=3, target_names=iris.target_names))




######### ROC Curve

#######################################################
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from scipy import interp

colors = ['blue', 'orange', 'red']

iris = load_iris()

X = iris.data
y = iris.target

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

#random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 500 * n_features)]

#____________________________________________________________________
# training test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=19)

one_vs_rest = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma=100, probability=True, random_state=12))
y_score = one_vs_rest.fit(X_train, y_train).decision_function(X_test)
y_true, y_pred = y_test, one_vs_rest.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#____________________________________________________________________
# Plot Precision-Recall curve for each class
fig = plt.figure(6, figsize=(9, 7))
plt.clf()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of {0} (area = {1:0.3f})'
                   ''.format(iris.target_names[i], roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
plt.show()





##############################  METRIK FUER REGRESSION

import sklearn.datasets as data
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

boston = data.load_boston()
X = boston.data[:,5:6]
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=112)

lr = lm.LinearRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(median_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


y_train_pred = lr.predict(X_train)
print(mean_squared_error(y_train, y_train_pred))
print(mean_absolute_error(y_train, y_train_pred))
print(median_absolute_error(y_train, y_train_pred))
print(r2_score(y_train, y_train_pred))







################################### GAMMA optimieren


iris = load_iris()

X = iris.data
y = iris.target
svc = SVC(kernel='rbf', C=1, gamma=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=12)

fig = plt.figure(5, figsize=(9, 7))
ax = fig.add_subplot(111)

param_range = np.logspace(-3, 2, 200)
train_scores, val_scores = [], []
for g in param_range:
    svc = SVC(kernel='rbf', C=1, gamma=g)
    clf = svc.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    val_scores.append(clf.score(X_val, y_val))
best_param = param_range[np.argmax(val_scores)]

plt.plot(param_range, train_scores, label="training sample", color="b", lw=2)
plt.plot(param_range, val_scores, label="validation sample", color="r", lw=2)

ax.set_xscale( "log" )
plt.xlabel("gamma")
plt.ylabel("accuracy")
plt.axis("tight")
plt.legend(loc='center')

plt.show()

best_param = param_range[np.argmax(val_scores)]
print("Best gamma value:", best_param)
#Best gamma value: 0.0382749447852
svc = SVC(kernel='rbf', C=1, gamma=best_param)
clf = svc.fit(X_train, y_train)
print("Best train accuracy:", clf.score(X_train, y_train))
print("Best val accuracy:", clf.score(X_val, y_val))
print("Best test accuracy:", clf.score(X_test, y_test))










