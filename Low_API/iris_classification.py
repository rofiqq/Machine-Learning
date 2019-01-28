from sklearn.datasets import load_iris
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_batch(x, y, batch_size=32) :
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


def onehot(y_label) :
    uniq = np.unique(y_label)
    new_labels = np.zeros((len(y_label), len(uniq)), dtype=int)
    for i in range(len(uniq)):
        idx = np.argwhere(y_label == uniq[i])
        new_labels[idx,i] = 1
    return new_labels

iris = load_iris()

data = iris.data
labels = iris.target
labels = onehot(labels)


# Split Data Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3)

# parameter
num_node_perlayer = [32, 64]
learning_rate = 0.01
EPOCH = 500
batch_size = 16

# Make Model 
num_features = len(X_train[0])
num_labels = len(y_train[0])

# Input
X = tf.placeholder("float32", shape = [None, num_features])
y = tf.placeholder("float32", shape = [None, num_labels])

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
ypredict = tf.argmax(y_est, axis=1)

# Define a loss function 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_est))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
epoches  = []
train_losses, test_losses = [], []
train_acc, test_acc = [], []

# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCH):
        x_batch, y_batch = get_batch(X_train, y_train, batch_size)
        for i in range(len(x_batch)):
            summary = sess.run(train_op, feed_dict={X: x_batch[i], y: y_batch[i]})
        
        train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train}))
        test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test}))
                
        train_loss = sess.run(loss, feed_dict={X: X_train, y: y_train})
        test_loss = sess.run(loss, feed_dict={X: X_test, y: y_test})
        
        epoches.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        
        title1 = 'Train Loss : {:.6f} Test Loss : {:.6f}'.format(train_loss, test_loss)
        title2 = 'Train Accuracy : {:.3f} Test Accuracy : {:.3f}'.format(train_accuracy, test_accuracy)
        
        ax1.clear()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_xlim([-0.05*EPOCH, 1.05*EPOCH])
        #ax1.set_ylim(bottom = 0)
        ax1.plot(epoches, train_losses, 'r', label = 'Train Loss')
        ax1.plot(epoches, test_losses, 'b', label = 'Test Loss')
        ax1.set_title(title1)
        ax1.legend()
        
        ax2.clear()
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlim([-0.05*EPOCH, 1.05*EPOCH])
        ax2.set_ylim(top = 1.1)
        ax2.plot(epoches, train_acc, 'r', label = 'Train Accuracy')
        ax2.plot(epoches, test_acc, 'b', label = 'Test Accuracy')
        ax2.set_title(title2)
        ax2.legend()
        
        plt.pause(0.001)
        
fig.savefig('iris.png')        
        