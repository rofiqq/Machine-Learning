# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:06:31 2019

@author: saksake
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

def datasets() :
    # LOAD BOSTON HOUSING DATASET
    boston = load_boston()
    
    # MAKE FEATURE DICTIONARY
    all_features = {}
    for i in range(len(boston.feature_names)):
        key = str(boston.feature_names[i])
        all_features[key] = boston.data[:,i]
    
    # ADD TARGET DICTIONARY
    all_features['MEDV'] = boston.target
    return all_features

all_features = datasets() 
raw_dataset = pd.DataFrame(all_features)
dataset = raw_dataset.copy()

# Define 
use_feature_name = [
        'CRIM',      # per capita crime rate by town
        'ZN',        # proportion of residential land zoned for lots over 25,000 sq.ft.
        'INDUS',     # proportion of non-retail business acres per town
        'CHAS',      # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        'NOX',       # nitric oxides concentration (parts per 10 million)
        'RM',        # average number of rooms per dwelling
        'AGE',       # proportion of owner-occupied units built prior to 1940
        'DIS',       # weighted distances to five Boston employment centres
        'RAD',       # index of accessibility to radial highways
        'TAX',       # full-value property-tax rate per $10,000
        'PTRATIO',   # pupil-teacher ratio by town
        'B',         # 1000(Bk '0.63)^2 where Bk is the proportion of blacks by town
        'LSTAT',     # % lower status of the population
        'MEDV'       # Median value of owner-occupied homes in $1000's
        ]

name_columns_category = [
        'CHAS'       # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        ]

name_columns_bucket = []

name_columns_numeric = [
        'CRIM',     # per capita crime rate by town
        'ZN',       # proportion of residential land zoned for lots over 25,000 sq.ft.
        'INDUS',    # proportion of non-retail business acres per town
        'NOX',      # nitric oxides concentration (parts per 10 million)
        'RM',       # average number of rooms per dwelling
        'AGE',      # proportion of owner-occupied units built prior to 1940
        'DIS',      # weighted distances to five Boston employment centres
        'RAD',      # index of accessibility to radial highways
        'TAX',      # full-value property-tax rate per $10,000
        'PTRATIO',  # pupil-teacher ratio by town
        'B',        # 1000(Bk '0.63)^2 where Bk is the proportion of blacks by town
        'LSTAT',    # % lower status of the population
        ]

name_label = 'MEDV'

# Find unused colums
unused_columns_feature = [ x for x in list(dataset.columns) if x not in use_feature_name]

# Drop unused column
dataset = dataset.drop(unused_columns_feature, axis = 1)

# Set categorical Column
for cats_name in name_columns_category :
    cats = list(dataset[cats_name].unique())
    for i, cat in enumerate(cats):
        dataset[cat] = (dataset[cats_name] == cat)*1.0
    dataset.pop(cats_name)

# Set Bucket Column
for bucket_info in name_columns_bucket :
    bucket_ = pd.cut(dataset[bucket_info[0]], bucket_info[1])
    buckets = list(bucket_.unique())
    for bucket in buckets:
        dataset[bucket] = (bucket_ == bucket)*1.0
        
#sns_plot = sns.pairplot(dataset[name_columns_numeric],  diag_kind="kde")
#sns_plot.savefig("dataset.png")
#plt.close()


# Split Train and Test data    
train_dataset = dataset.sample(frac=0.7,random_state=1)
test_dataset = dataset.drop(train_dataset.index)

# Split features from labels
train_labels = train_dataset.pop(name_label)
test_labels = test_dataset.pop(name_label)

# Numeric Statistics
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# Normalize numeric data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

train_dataset = norm(train_dataset)
test_dataset = norm(test_dataset)


def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.01)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    n = 100
    to = epoch % n
        
    if epoch % n == 0: 
        txt = '\nEpoch - {:<5d}\n'.format(epoch+1)
        txt += 'Train MSE   :  {:8.3f}  |  Val MSE  :  {:8.3f}\n'.format(logs['mean_squared_error'], 
                             logs['val_mean_squared_error'])
        
        txt += 'Train MAE   :  {:8.3f}  |  Val MAE  :  {:8.3f}\n'.format(logs['mean_absolute_error'], 
                             logs['val_mean_absolute_error'])
        
        txt += 'Train Loss  :  {:8.3f}  |  Val Loss :  {:8.3f}'.format(logs['loss'], logs['val_loss'])
        
        print(txt)
    # ------------------------------------------------------------------
    to += 1
    maxblock = 46
    maxcount = n
    doneblock = int((float(to)/maxcount)*maxblock)
    sys.stdout.write('\r')
    sys.stdout.write('[{:}{:}]'.format('#'*doneblock,' '*(maxblock-doneblock)))
    if (maxblock-doneblock) != 0 :
        sys.stdout.flush()
    # ------------------------------------------------------------------

EPOCHS = 500

history = model.fit(
        train_dataset, train_labels,
        batch_size = 32,
        epochs=EPOCHS, validation_split = 0.2, verbose=0,
        callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_history(history):
  fig = plt.figure(figsize = (12, 6))
  ax1 = fig.add_subplot(121)
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Mean Abs Error')
  ax1.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  ax1.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  ax1.legend()
  
  ax2 = fig.add_subplot(122)
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Mean Square Error')
  ax2.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  ax2.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  ax2.legend()

plot_history(history)

from scipy import stats
test_predictions = model.predict(test_dataset).flatten()

slope, intercept, r_value, p_value, std_err = stats.linregress(test_labels, test_predictions)

def f(x):
    return slope*x+intercept

fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_subplot(121)
ax1.scatter(test_labels, test_predictions)
ax1.set_xlabel('True Values (log10 %s)' % (name_label))
ax1.set_ylabel('Predictions (log10 %s)' % (name_label))
ax1.axis('equal')
ax1.axis('square')
ax1.plot([0, max(test_labels)],[0,max(test_labels)], label = 'slope = 1.00')
ax1.plot([0, max(test_labels)],[f(0),f(max(test_labels))], label = 'slope = %.2f' % (slope))
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(range(len(test_labels)), test_labels)
ax2.plot(range(len(test_labels)), test_predictions)
ax2.set_ylabel('Values (%s)' % (name_label))
ax2.legend(['True Values', 'Predictions'])



