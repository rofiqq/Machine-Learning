# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 03:44:58 2019

@author: saksake
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_batch(x, y, batch_size=10) :
    arr = np.arange(len(x))
    np.random.shuffle(arr)
    i, high = 0, 0
    x_batch, y_batch = [], []
    while high < len(x):
        low = i*batch_size
        high = (i+1)*batch_size
        if high > len(x) :
            high = len(x)
        i += 1
        idxs = arr[low : high]
        x_batch.append(x[idxs])
        y_batch.append(y[idxs])
    return x_batch, y_batch


f = lambda x : np.sin(x)

n = 1000
x = np.linspace(0, 16, n).astype('float32')
y = f(x).astype('float32')
err = np.random.normal(0, 0.05, n)
yerr = y + err

x = x.reshape((len(x),1))
yerr = yerr.reshape((len(yerr),1))

#plt.plot(x,yerr,'o')

# Split Data Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, yerr, test_size = 0.3)

#plt.plot(X_train, y_train,'o')

# parameter
num_node_perlayer = [16, 32]
learning_rate = 0.1
EPOCH = 1200
batch_size = 64

# Make Model 
num_features = 1
num_labels = 1

# Input
X = tf.placeholder("float32", shape = [None, num_features])
Y = tf.placeholder("float32", shape = [None, num_labels])

# Dictionary of Weight and Bias
mod_weights = {
            'h1' : tf.Variable(np.random.normal(size = [num_features, num_node_perlayer[0]]).astype('float32')),
            'h2' : tf.Variable(np.random.normal(size = [num_node_perlayer[0], num_node_perlayer[1]]).astype('float32')),
            'out' : tf.Variable(np.random.normal(size = [num_node_perlayer[1], num_labels]).astype('float32'))
            }
mod_biases = {
            'b1' : tf.Variable(np.random.normal(size = [num_node_perlayer[0]]).astype('float32')),
            'b2' : tf.Variable(np.random.normal(size = [num_node_perlayer[1]]).astype('float32')),
            'out' : tf.Variable(np.random.normal(size = [num_labels]).astype('float32'))
            }

# Hidden layer1
layer_1 = tf.add(tf.matmul(X, mod_weights['h1']), mod_biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, mod_weights['h2']), mod_biases['b2'])
layer_2 = tf.nn.relu(layer_2)

# Output fully connected layer
y_est = tf.matmul(layer_2, mod_weights['out']) + mod_biases['out'] 


# Define a loss function 
loss = tf.losses.mean_squared_error(labels = Y,
                                    predictions = y_est)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)


import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title = 'Sinus Function', artist = 'Me')
writer = FFMpegWriter(fps=72, metadata = metadata)

fig = plt.figure(figsize = (12,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

epoches  = []
train_errors, test_errors = [], []
y_preds = []
# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    with writer.saving(fig, 'Sinus Function2a.mp4', dpi=100) :
        sess.run(init)
        to = 0
        for epoch in range(EPOCH):
            # ------------------------------------------------------------------
            to += 1
            maxblock = 46
            maxcount = EPOCH
            percent = 100*(to/float(EPOCH))
            doneblock = int((float(to)/maxcount)*maxblock)
            sys.stdout.write('\r')
            sys.stdout.write('[{:}{:}] {:6.2f} %'.format('#'*doneblock, ' '*(maxblock-doneblock), percent))
            if (maxblock-doneblock) != 0 :
                sys.stdout.flush()
            # ------------------------------------------------------------------
    
            x_batch, y_batch = get_batch(X_train, y_train, batch_size)
            for i in range(len(x_batch)):
                summary = sess.run(train_op, feed_dict={X: x_batch[i], Y: y_batch[i]})
            
            train_error = sess.run(loss, feed_dict={X: X_train, Y: y_train})
            test_error = sess.run(loss, feed_dict={X: X_test, Y: y_test})
            
            y_pred = sess.run(y_est, feed_dict={X: X_train, Y: y_train})
            y_preds.append(y_pred)
            epoches.append(epoch)
            train_errors.append(train_error)
            test_errors.append(test_error)
            
            minerr = np.argmin(train_errors)
            ybest = y_preds[minerr]
             
            ax1.clear()
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_xlim([-0.05*EPOCH, 1.05*EPOCH])
            ax1.semilogy(epoches, train_errors, 'r', label = 'Train Loss')
            ax1.semilogy(epoches, test_errors, 'b', label = 'Test Loss')
            ax1.set_title('Train Error = {:.3f}\nTest Error = {:.3f}'.format(train_error, test_error))
            ax1.legend()
            
            ax2.clear()
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.plot(X_train,  y_train, 'ok', label = 'True Value')
            ax2.set_ylim([-1.5, 1.5])
            idsort = np.argsort(X_train, axis = 0).reshape(-1)
            ax2.plot(X_train[idsort],  y_pred[idsort], '--b', lw = 3, label = 'Predicted')
            ax2.plot(X_train[idsort],  ybest[idsort], 'r', lw = 3, label = 'Best Predict')
            ax2.set_title('Best Fit Error = {:.3f}'.format(train_errors[minerr]))
            ax2.legend()
            
            writer.grab_frame()