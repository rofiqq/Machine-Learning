# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:36:46 2020

@author: saksake
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def sin(N, noise = False):
    # Make sinusoidal wave
    pi = np.degrees(np.pi)
    t = np.linspace(0, 1, N)
    fmax = 0.1
    freq = np.linspace(0, 1.5*fmax, 100)[1:]
    
    y = np.zeros(t.shape)
    for f in freq :
        phase = np.random.randint(0, 360)
        y +=  np.sin(2*pi*f*t + phase) 
    if noise :
        return t, y + 0.1*np.random.randn(N)
    else :
        return t, y 
        

N = 5000
time, series = sin(N, noise = False)

# Split Train - Test Data
train_percentage = 0.8
ntrain = int(len(series)*train_percentage)
series_train = series[:ntrain]
series_test = series[ntrain:]

# Standarisation
train_mean = series_train.mean()
train_std = series_train.std()
series_train = (series_train - train_mean)/train_std
series_test = (series_test - train_mean)/train_std



def create_dataset(series, n_past = 2, n_future = 1, n_step = 1) :
    features, labels = [], []
    for i in range(len(series) - (n_past + n_future - 1)):
        idin = np.arange(i, i+n_past, n_step)
        idout = np.arange(i+n_past, i+n_past+n_future, n_step)
        feature = np.expand_dims(series[idin], axis = 1)
        label = series[idout]
        
        features.append(feature)
        labels.append(label)
        
    return np.asarray(features), np.asarray(labels)


n_past = 20
n_future = 5
n_step = 1


print("Create Train Dataset")
Xtrain, Ytrain = create_dataset(series_train, n_past = n_past, 
                                n_future = n_future, n_step = n_step)

print("Create test Dataset")
Xtest, Ytest = create_dataset(series_test, n_past = n_past, 
                                n_future = n_future, n_step = n_step)


import tensorflow as tf
tf.keras.backend.clear_session()

# Make Model
num_epochs = 20
model = tf.keras.Sequential([
        tf.keras.layers.LSTM(8, return_sequences = False,
                             input_shape=(Xtrain.shape[1], Xtrain.shape[2])),
        tf.keras.layers.Dense(Ytrain.shape[1])
        ])
    
opt = tf.keras.optimizers.Adam(lr = 0.002)
model.compile(loss = 'mae', optimizer = opt)

    
history = model.fit(Xtrain, Ytrain,  validation_data = (Xtest, Ytest),
                    shuffle=False,
                    epochs=num_epochs,
                    verbose = 2)

# Predict Model
Ypredict = model.predict(Xtest)


##### Plot ######
fig = plt.figure()
ax = fig.add_subplot(121)
ax.semilogy(history.epoch, history.history['loss'], '-b', label = 'Train Loss')
ax.semilogy(history.epoch, history.history['val_loss'], '-r', label = 'Test Loss')
ax.legend();ax.grid()


ax1 = fig.add_subplot(122)
ax1.plot(time[:ntrain], series_train, label = 'Train Data')
ax1.plot(time[ntrain:], series_test, label = 'Test Data')
ax1.legend();ax1.grid()


fig = plt.figure()
ax2 = fig.add_subplot(121)
ax2.set_title('All Test Data')
for idx in range(0, len(Ytest), n_future) :
    ax2.plot(range(ntrain+n_past+idx, ntrain+n_past+n_future+idx, n_step), 
            Ytest[idx,:], '--x', c = 'k', ms = 5)
    ax2.plot(range(ntrain+n_past+idx, ntrain+n_past+n_future+idx, n_step), 
            Ypredict[idx,:], '--o', c = 'r',  ms = 5)
ax2.legend(['Output True', 'Output Predict'])
ax2.grid()


ax2 = fig.add_subplot(122)
ax2.set_title('First 3 Test Data')
n_pred = 0
for idx in range(0, len(Ytest), n_future) :
    ax2.plot(range(ntrain+n_past+idx, ntrain+n_past+n_future+idx, n_step), 
            Ytest[idx,:], '--x', ms = 5)
    ax2.plot(range(ntrain+n_past+idx, ntrain+n_past+n_future+idx, n_step), 
            Ypredict[idx,:], '--o',  ms = 5)
    n_pred += 1
    if n_pred == 4 :
        break
ax2.legend(['Output True', 'Output Predict'])
ax2.grid()