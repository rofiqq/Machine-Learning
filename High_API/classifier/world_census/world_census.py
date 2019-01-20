# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:06:31 2019

@author: saksake
"""

import numpy as np
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def datasets(filename) :
    # Open CSV File
    with open(filename,'r') as f :
        text = f.readlines()
        
    text[0] = re.sub(r'\n', '', text[0]) 
    feature_name = text[0].split(';')
    all_features = {}
    for i in range(len(text)-1):
        i += 1
        text[i] = re.sub(r'\n', '', text[i])
        txt = text[i].split(';')
        
        for j in range(len(txt)) :
            if feature_name[j] not in all_features :
                all_features[feature_name[j]] = []
                
            if txt[j] == '':
                txt[j] = '0'
                
            if txt[j] == '<=50K' :
                txt[j] = '0'
            elif txt[j] == '>50K' :
                txt[j] = '1'
            
            if is_number(txt[j]) :
                if txt[j].isnumeric():
                    txt[j] = int(txt[j])
                else :
                    txt[j] = float(txt[j])
                
            all_features[feature_name[j]].append(txt[j])
    return all_features


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

label_key ='income'
train_portion = 0.7

all_features = datasets(filename = 'world_census.csv')


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
        hidden_units=[512, 256],
        
        optimizer='Adagrad', #'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'
        
        activation_fn=tf.nn.relu, # relu. tanh, sigmoid
        
        n_classes = len(np.unique(inp_train_label[label_key])),
        
        # Model directory
        model_dir = 'WorldCensus')

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

