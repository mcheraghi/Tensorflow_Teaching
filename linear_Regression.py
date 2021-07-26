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



# -----Get the data from the archive
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
dataset = dataset.dropna()


# -----Convert the origin column into US, EUROPE and JAPAN columns
                    
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})              
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())              


# -----Split the data into train and test

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# -----ploting all the variable againseach other! That's just amazing!

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')



#-----Split features from labels,

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')  #'MPG' is removed from train_features and added to train_labels
test_labels = test_features.pop('MPG')

print(train_features.head())
print(train_labels.head())


# -----Normalize

a = train_dataset.describe().transpose() #To see the variable statistics
print(a)
print(a[['mean', 'std']])

normalizer = preprocessing.Normalization(axis=-1)  #definition of normalization
normalizer.adapt(np.array(train_features))   #.adapt() it to the data:
print(normalizer.mean.numpy())


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())
  
  
# -----Predicting MPG out of horsepower variable

horsepower = np.array(train_features['Horsepower'])
print(horsepower)
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
]) # the model which is a normalizer plus a linear regression

print(horsepower_model.summary())

print(horsepower_model.predict(horsepower[:10])) # The model is not optimized yet


horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
    
    

history = horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())



def plot_loss(history):
  f = plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  return f

  
f = plot_loss(history) #Plot the convergance history




# -----Plot the data and fitted curve

test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  f = plt.figure()
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
 
print(horsepower_model.layers[1].weights)
f = plot_horsepower(x,y)



plt.show()



