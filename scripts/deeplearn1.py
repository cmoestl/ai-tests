import matplotlib.pyplot as plt
import numpy as np
from distutils.version import StrictVersion
import pandas as pd
import seaborn as sns
import sys
import warnings
import sklearn
import tensorflow as tf
import keras
import pandas as pd

plt.close('all')

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target


#one hot encoding
from keras.utils.np_utils import to_categorical
num_categories = 3
y = to_categorical(y, num_categories)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8, stratify=y)
colors = 'bwr'#['b','y','r']
CMAP = colors#plt.cm.rainbow
#print(iris.DESCR)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
pd.plotting.scatter_matrix(iris_df, c=iris.target, cmap=CMAP, edgecolor='black', figsize=(5, 5))






#ANN with 1 hidden layer

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

#input layer
inputs = Input(shape=(4, ))
#fully connected hidden layer
fc = Dense(3)(inputs)
#output
predictions = Dense(3, activation='softmax')(fc)

model = Model(input=inputs, output=predictions)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
              
              
print(np.round(model.predict(np.array([[ 5.1,  3.5,  1.4,  0.2]])),2))

#not trained so far


######################## now train


model.fit(X_train, y_train, epochs=500, validation_split=0.3)


print(np.round(model.predict(np.array([[ 5.1,  3.5,  1.4,  0.2]])),2))

print()
print()
print('Train loss/accuracy')
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(train_loss, train_accuracy)

print('Test loss/accuracy')
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(test_loss, test_accuracy)

sys.exit()








############################


from keras.models import Sequential

model = Sequential()

from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
              
model.compile(loss=keras.losses.categorical_crossentropy, 
          optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
          
sys.exit()     
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, y_train, epochs=5)#, batch_size=32)    

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)    


classes = model.predict(X_test, batch_size=128)  