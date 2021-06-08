import matplotlib.pylab as plt
import numpy as np
from distutils.version import StrictVersion
import pandas as pd
import seaborn as sns
import sys


plt.close('all')



###### main

#!curl -O https://raw.githubusercontent.com/DJCordhose/ml-examples/master/datasets/Iris/iris_dirty.csv

df = pd.read_csv('iris_dirty.csv',
                 header=None,
                 encoding='iso-8859-15',
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])


df.head()

df.count()


s1 = pd.Series([10,20,30])

type(s1)
df1 = pd.DataFrame(
    {'one': pd.Series([10,20,30], index=['a', 'b', 'c']),
     'two': pd.Series([10,20,30], index=['a', 'b', 'c'])
    })


