import matplotlib.pyplot as plt
import numpy as np
from distutils.version import StrictVersion
import pandas as pd
import seaborn as sns
import sys
import warnings

plt.close('all')


warnings.filterwarnings('ignore')
import sklearn
print(sklearn.__version__)

assert StrictVersion(sklearn.__version__ ) >= StrictVersion('0.18.1')


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
#print(tf.__version__)

assert StrictVersion(tf.__version__) >= StrictVersion('1.1.0')

colors = 'bwr'#['b','y','r']
CMAP = colors#plt.cm.rainbow

import keras
print(keras.__version__)

assert StrictVersion(keras.__version__) >= StrictVersion('2.0.0')


import pandas as pd
print(pd.__version__)

assert StrictVersion(pd.__version__) >= StrictVersion('0.19.0')





############



from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
pd.plotting.scatter_matrix(iris_df, c=iris.target, cmap=CMAP, edgecolor='black', figsize=(5, 5))







from keras.layers import Input
inputs = Input(shape=(4, ))

from keras.layers import Dense
fc = Dense(3)(inputs)
from keras.models import Model
model = Model(input=inputs, output=fc)

model.summary()

sys.exit()


w0 = 3
w1 = -4
w2 = 2

def neuron_no_activation(x1, x2):
    sum = w0 + x1 * w1 + x2 * w2
    return sum
    
    
neuron_no_activation(5.1, 3.5)

def centerAxis(uses_negative=False):
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    if uses_negative:
        ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    
def np_step(X):
    return 0.5 * (np.sign(X) + 1)
    
    
    
x = np.arange(-10,10,0.01)
y = np_step(x)

centerAxis()
plt.plot(x, y, lw=3)



def np_sigmoid(X):
    return 1 / (1 + np.exp(X * -1))
    
    
x = np.arange(-10,10,0.01)
y = np_sigmoid(x)

centerAxis()
plt.plot(x,y,lw=3)



x = np.arange(-10,10,0.01)
y = np.tanh(x)

centerAxis()
plt.plot(x,y,lw=3)



########################


from keras.layers import Input
inputs = Input(shape=(4, ))
from keras.layers import Dense
fc = Dense(3)(inputs)
from keras.models import Model
model = Model(input=inputs, output=fc)

model.summary()



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.predict(np.array([[ 5.1,  3.5,  1.4,  0.2]]))


inputs = Input(shape=(4, ))
fc = Dense(3)(inputs)
predictions = Dense(3, activation='softmax')(fc)
model = Model(input=inputs, output=predictions)

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
              
model.predict(np.array([[ 5.1,  3.5,  1.4,  0.2]]))


#######################
X = np.array(iris.data)
y = np.array(iris.target)
X.shape, y.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


from keras.utils.np_utils import to_categorical

num_categories = 3

y = to_categorical(y, num_categories)



model.fit(X_train, y_train, epochs=500, validation_split=0.3)

model.predict(np.array([[ 5.1,  3.5,  1.4,  0.2]]))

print()
print()
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(train_loss, train_accuracy)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(test_loss, test_accuracy)