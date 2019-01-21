# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:16:36 2019

@author: saksake
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


class data():
    def square(ndat) :
        nd = int(ndat/4)
        data_ = np.random.uniform(0, 5, (nd,2))
        
        dat1 = np.vstack((data_, data_+[5.2,5.2]))
        v1 = np.linspace(1, 1, len(dat1)).reshape(len(dat1), 1)
        dat1 = np.hstack((dat1, v1))
        
        dat2 = np.vstack((data_+[5.2,0],data_+[0,5.2]))
        v2 = np.linspace(2, 2, len(dat2)).reshape(len(dat2), 1)
        dat2 = np.hstack((dat2, v2))
        
        
        data = np.vstack((dat1, dat2))
        x_feature = data[:,:2]
        y_label = data[:,2]
        return x_feature, y_label

    def spiral(ndat):
        nd = int(ndat/2)
        maxangle = 2.0*360
        tetha_deg = np.linspace(0.5, maxangle, nd)
        tetha_rad = np.deg2rad(tetha_deg)
        err = 0.02
        x0 = (tetha_deg**0.5) * np.cos(tetha_rad)
        x0 = x0 + x0 * np.random.uniform(-err, err, size = len(x0))
        
        y0 = (tetha_deg**0.5) * np.sin(tetha_rad)
        y0 = y0 + y0 * np.random.uniform(-err, err, size = len(y0))
        
        z0 = np.linspace(0,0, nd)
        
        x1 = -(tetha_deg**0.5) * np.cos(tetha_rad)
        x1 = x1 + x1 * np.random.uniform(-err, err, size = len(x0))
        
        y1 = -(tetha_deg**0.5) * np.sin(tetha_rad)
        y1 = y1 + y1 * np.random.uniform(-err, err, size = len(y0))
        
        z1 = np.linspace(1,1, nd)
        
        x = np.hstack((x0,x1))
        y = np.hstack((y0,y1))
        z = np.hstack((z0,z1))
        data = np.vstack((x, y, z)).T
        x_feature, y_label = data[:,:2], data[:,2]
        return x_feature, np.int64(y_label)

def onehot(y_label) :
    uniq = np.unique(y_label)
    new_labels = np.zeros((len(y_label), len(uniq)), dtype=int)
    for i in range(len(uniq)):
        idx = np.argwhere(y_label == uniq[i])
        new_labels[idx,i] = 1
    return new_labels


# Define 
use_feature_name = ['x', 'y', 'z'] 

name_columns_category = []

name_columns_bucket = []

name_columns_numeric = ['x', 'y']

name_label = 'z'

# Prepare Data
features, labels = data.spiral(1000)
all_features = {'x' : features[:,0],
                'y' : features[:,1],
                'z' : labels}

raw_dataset = pd.DataFrame(all_features)
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
    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(train_labels[0]), activation='softmax')
  ])

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    n = 100
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

EPOCHS = 500

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

##############################################################################
# Make Grid for mapping
###############################################################################
X_grid = []
ntest = 100
for i in np.linspace(min(train_dataset['x']), max(train_dataset['x']), ntest) :
    for j in np.linspace(min(train_dataset['y']), max(train_dataset['y']), ntest) :
        X_grid.append([i,j])
X_grid = np.asarray(X_grid)

grid_dataset = pd.DataFrame({'x':X_grid[:,0], 'y':X_grid[:,1]})

grid_predictions = model.predict(grid_dataset)
grid_pred = np.argmax(grid_predictions, axis = 1)

X1 = X_grid[:,0].reshape((ntest, ntest))
Y1 = X_grid[:,1].reshape((ntest, ntest))
Z1 = grid_pred.reshape((ntest, ntest))

idx1 = np.where(raw_dataset['z'] == 0)[0]
idx2 = np.where(raw_dataset['z'] == 1)[0]
plt.figure()
plt.plot(norm(raw_dataset).iloc[idx1]['x'], norm(raw_dataset).iloc[idx1]['y'],'o')
plt.plot(norm(raw_dataset).iloc[idx2]['x'], norm(raw_dataset).iloc[idx2]['y'],'o')
plt.contourf(X1, Y1, Z1)

