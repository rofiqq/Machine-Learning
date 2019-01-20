# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:06:31 2019

@author: saksake
"""

import numpy as np

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


def splitdict(feature_dict, train_portion, label_key) :
    train_feature, train_label = {}, {}
    key = list(feature_dict.keys())
    ndata = len(feature_dict[key[0]])
    train_n = int(ndata*train_portion)
    idxs = np.array(range(ndata))
    np.random.shuffle(idxs)
    
    train_idx = idxs[:train_n]
    test_idx = idxs[train_n:]
    for key in feature_dict :
        if key == label_key :
            train_label[key] = {}
            train_label[key] = np.array(feature_dict[key])[train_idx]
        else :
            train_feature[key] = {}
            train_feature[key] = np.array(feature_dict[key])[train_idx]       
            
    test_feature, test_label = {}, {}
    for key in feature_dict :
        if key == label_key :
            test_label[key] = {}
            test_label[key] = np.array(feature_dict[key])[test_idx]
        else :
            test_feature[key] = {}
            test_feature[key] = np.array(feature_dict[key])[test_idx]
                    
    return train_feature, train_label, test_feature, test_label


use_feature_name = ['x', 'y', 'z']       

name_columns_category = []

name_columns_bucket = []

name_columns_numeric = ['x', 'y']

label_key ='z'
train_portion = 0.6

# Prepare Data
features, labels = data.spiral(1000)
all_features = {'x' : features[:,0],
                'y' : features[:,1],
                'z' : labels}

for key in all_features:
    print("'{:}',".format(key))
    
    
# CHOOSE INTEREST FEATURES FROM ALL FEATURES
used_features = {}
for key in all_features:
    if key in use_feature_name :
        used_features[key] = all_features[key]


inp_train_feature, inp_train_label, inp_test_feature, inp_test_label = splitdict(feature_dict = used_features, 
                                                                 train_portion = train_portion, 
                                                                 label_key = label_key)


import tensorflow as tf

# MAKE INPUT FUNCTION
# TRAIN DATA
input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x = inp_train_feature, 
    y = inp_train_label[label_key], 
    shuffle=True, 
    batch_size=128, 
    num_epochs=None
)

# TEST DATA
input_fn_test = tf.estimator.inputs.numpy_input_fn(
    x = inp_test_feature, 
    y = inp_test_label[label_key], 
    shuffle=False, 
    batch_size=128, 
    num_epochs=1
)

# Define feature columns.
feature_columns_numeric, feature_columns_category, feature_columns_bucket = [], [], []
for key in inp_train_feature :
    # Define numeric feature columns.
    if key in name_columns_numeric :
        feature_columns_numeric.append(tf.feature_column.numeric_column(key))
        
    # Define categorycal feature columns.
    elif key in name_columns_category :
        uniq = (np.unique(inp_train_feature[key])).tolist()
        
        cat_column = tf.feature_column.categorical_column_with_vocabulary_list(key = key,
                                                                          vocabulary_list = uniq)
        
        embed_column = tf.feature_column.embedding_column(
                        categorical_column=cat_column,
                        dimension=len(uniq)
                        )
        feature_columns_category.append(embed_column)
        
    # Define bucket feature columns.
    elif key in name_columns_bucket :
        numeric_column = tf.feature_column.numeric_column(key)

        # make bucket boundaries
        arr = np.linspace(min(inp_train_feature[key]), max(inp_train_feature[key]), 1000)
        n_bucket = 3
        q = 1./(n_bucket+1.)
        boundaries = []
        for i in range(n_bucket):
            boundaries.append(int(np.quantile(arr, q*(i+1))))
            
        # Then, bucketize the numeric column on the years 1960, 1980, and 2000.
        bucketized_feature_column = tf.feature_column.bucketized_column(
            source_column = numeric_column,
            boundaries = boundaries)
        
        feature_columns_bucket.append(bucketized_feature_column)
        
feature_columns = feature_columns_numeric + feature_columns_category + feature_columns_bucket

# DEFINE ESTIMATOR
estimator= tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        
        # Two hidden layers
        hidden_units=[256, 128],
        
        optimizer='Adagrad', #'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'
        
        activation_fn=tf.nn.relu, # relu. tanh, sigmoid
        
        n_classes = len(np.unique(inp_train_label[label_key])),
        
        # Model directory
        model_dir = 'Geometri')

# TRAIN MODEL
estimator.train(input_fn=input_fn_train, steps=5000)

# EVALUATE MODEL
print('-------------------------------------')
evaluate = estimator.evaluate(input_fn = input_fn_test)
print('-------------------------------------')

# PREDICT
pred = list(estimator.predict(input_fn = input_fn_test))

# VISUALIZE TESTING DAN PREDICTED
y_prob = [x['probabilities'] for x in pred]
y_pred = np.asarray([np.argmax(x) for x in y_prob])

y_real = inp_test_label[label_key]

ntrue = len(np.where(y_pred == y_real)[0])

acc = ntrue/float(len(y_real))

print('Accuracy = {:}'.format(acc))

idx1 = np.where(used_features['z'] == 0)
idx2 = np.where(used_features['z'] == 1)
import matplotlib.pyplot as plt
plt.plot(used_features['x'][idx1], used_features['y'][idx1], 'o')
plt.plot(used_features['x'][idx2], used_features['y'][idx2], 'o')
plt.xticks([])
plt.yticks([])


###############################################################################
# Make Grid for mapping
###############################################################################
X_grid = []
ntest = 100
for i in np.linspace(min(used_features['x']), max(used_features['x']), ntest) :
    for j in np.linspace(min(used_features['y']), max(used_features['y']), ntest) :
        X_grid.append([i,j])
X_grid = np.asarray(X_grid)

grid_feature = {
        'x' : X_grid[:,0],
        'y' : X_grid[:,1]
        }

input_fn_grid = tf.estimator.inputs.numpy_input_fn(
    x = grid_feature,
    shuffle=False, 
    batch_size=32, 
    num_epochs=1
)
# Predict
pred_grid_desc = list(estimator.predict(input_fn = input_fn_grid))

grid_probalilies = np.asarray([list(x['probabilities']) for x in pred_grid_desc])
grid_pred = np.argmax(grid_probalilies, axis = 1)

X1 = X_grid[:,0].reshape((ntest, ntest))
Y1 = X_grid[:,1].reshape((ntest, ntest))
Z1 = grid_pred.reshape((ntest, ntest))

plt.contourf(X1, Y1, Z1)

