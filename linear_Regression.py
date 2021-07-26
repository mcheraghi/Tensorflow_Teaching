import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   #to have pair plot


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)



# Get the data from the archive
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

   
   
# The unknon values

print(dataset.isna().sum())


# Convert the origin column into US, EUROPE and JAPAN columns
                    
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})              
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())              


# Split the data into train and test

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#ploting all the variable againseach other! That's just amazing!

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()


#To see the variables in 

a = train_dataset.describe().transpose()
print(a)
print(a[['mean', 'std']])



