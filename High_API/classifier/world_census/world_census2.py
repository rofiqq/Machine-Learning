# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:16:36 2019

@author: saksake
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt

def onehot(y_label) :
    uniq = np.unique(y_label)
    new_labels = np.zeros((len(y_label), len(uniq)), dtype=int)
    for i in range(len(uniq)):
        idx = np.argwhere(y_label == uniq[i])
        new_labels[idx,i] = 1
    return new_labels



# Define 
use_feature_name = ['age',
                    'workclass',
                    'fnlwgt',
                    'education',
                    'education_num',
                    'marital_status',
                    'occupation',
                    'relationship',
                    'race',
                    'sex',
                    'capital_gain',
                    'capital_loss',
                    'hours_per_week',
                    'native_country',
                    'income']       

name_columns_category = [
                    'education',
                    'marital_status',
                    'native_country',
                    'occupation',
                    'relationship',
                    'race',
                    'workclass',
                    'sex']

name_columns_bucket = []

name_columns_numeric = ['age',
                    'fnlwgt',
                    'education_num',
                    'capital_gain',
                    'capital_loss',
                    'hours_per_week']

name_label ='income'

# Open data
filename = 'world_census.csv'
raw_dataset = pd.read_csv(filename,
                      na_values = ["?",''], comment='\t',
                      sep=";", skipinitialspace=True)
dataset = raw_dataset.copy()


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
    dataset.pop(bucket_info[0])
        
sns_plot = sns.pairplot(dataset[name_columns_numeric],  diag_kind="kde")
sns_plot.savefig("dataset.png")
plt.close()
# Edit Data
dataset['fnlwgt']  = np.log10(dataset['fnlwgt'])

dataset['capital_gain']  = np.log10(dataset['capital_gain'])
dataset['capital_gain'][np.isinf(dataset['capital_gain'])] = 0.

dataset['capital_loss']  = np.log10(dataset['capital_loss'])
dataset['capital_loss'][np.isinf(dataset['capital_loss'])] = 0.

dataset['hours_per_week']  = np.log10(dataset['hours_per_week'])
dataset['hours_per_week'][np.isinf(dataset['hours_per_week'])] = 0.

sns_plot = sns.pairplot(dataset[name_columns_numeric],  diag_kind="kde")
sns_plot.savefig("dataset_edit.png")
plt.close()

# Split Train and Test data    
train_dataset = dataset.sample(frac=0.7,random_state=1)
test_dataset = dataset.drop(train_dataset.index)

# Split features from labels
train_labels = train_dataset.pop(name_label)
test_labels = test_dataset.pop(name_label)

# One Hot Label
train_labels = onehot(train_labels)
test_labels = onehot(test_labels)

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
    tf.keras.layers.Dense(len(train_labels[0]), activation='softmax')
  ])

  optimizer = tf.keras.optimizers.Adam(lr=0.01)

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    n = 10
    to = epoch % n
        
    if epoch % n == 0: 
        txt = '\nEpoch - {:<5d}\n'.format(epoch+1)
        txt += 'Train Loss   :  {:8.3f}  |  Val Loss  :  {:8.3f}\n'.format(logs['loss'], 
                             logs['val_loss'])
        
        txt += 'Train Acc    :  {:8.3f}  |  Val Acc   :  {:8.3f}\n'.format(logs['acc'], 
                             logs['val_acc'])
        
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

EPOCHS = 100

history = model.fit(
        train_dataset, train_labels,
        batch_size = 32,
        epochs=EPOCHS, 
        validation_data=(test_dataset, test_labels),
        verbose=0,
        callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Evaluate the model
scores = model.evaluate(test_dataset, test_labels)
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

test_predictions = model.predict(test_dataset)

def plot_history(history):
  fig = plt.figure(figsize = (12, 6))
  ax1 = fig.add_subplot(121)
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.plot(hist['epoch'], hist['loss'], label='Train Loss')
  ax1.plot(hist['epoch'], hist['val_loss'],  label = 'Val Loss')
  ax1.legend()
  
  ax2 = fig.add_subplot(122)
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.plot(hist['epoch'], hist['acc'], label='Train Acc')
  ax2.plot(hist['epoch'], hist['val_acc'],  label = 'Val Acc')
  ax2.legend()
  return

plot_history(history)
