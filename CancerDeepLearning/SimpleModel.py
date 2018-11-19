import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import requests
import os.path
import csv
import pandas as pd
import sys
import os
ops.reset_default_graph()
sess = tf.Session()

y_vals = pd.read_table("CancerTypes_y.txt", sep="\t",header=None)
x_vals_rna = pd.read_table("RNAseq_processed.txt", sep="\t",header=0)
x_vals_cnv = pd.read_table("CNV_processed.txt", sep="\t",header=0)
x_vals_rna = x_vals_rna.ix[:,1:]
x_vals_cnv = x_vals_cnv.ix[:,1:]

# transpose
x_vals_rna  = x_vals_rna.transpose()
x_vals_cnv  = x_vals_cnv.transpose()

# convert pd dataframe to numpy array  
x_vals_cnv = x_vals_cnv.values
x_vals_rna = x_vals_rna.values
y_vals = y_vals.values
print(len(x_vals_cnv))
print(x_vals_cnv.shape)
print(x_vals_rna.shape)
print(y_vals.shape)

# Select training and testing sets
# Set seed for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)
# Split data into train/test = 80%/20%, note x_vals_rna and x_vals_cnv have the same length 
train_indices = np.random.choice(len(x_vals_rna), round(len(x_vals_rna)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals_rna))) - set(train_indices)))
# rna seq 
x_vals_rna_train = x_vals_rna[train_indices]
x_vals_rna_test = x_vals_rna[test_indices]
# cnv
x_vals_cnv_train = x_vals_cnv[train_indices]
x_vals_cnv_test = x_vals_cnv[test_indices]
# y vals
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


#1. Define Tensorflow computational graph for Logistic Model

# Initialize placeholders
nFeatures = x_vals_cnv_train.shape[1]
dimResponse = y_vals_train.shape[1]
# Define inputs for session.run 
x_data = tf.placeholder(shape=[None, nFeatures], dtype=tf.float32)# tensor with nFeature columns. Note None takes any value when computation takes place 
y_target = tf.placeholder(shape=[None, dimResponse], dtype=tf.float32)# tensor with 1 column
# Initialize variables for regression: "y=sigmoid(A×x+b)"
A = tf.Variable(tf.random_normal(shape=[nFeatures,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
# Declare model operations "y = A×x + b"
model_output = tf.add(tf.matmul(x_data, A), b)
# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y_target))
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.1**3)
train_step = my_opt.minimize(loss)

#1.1 Train Logistic Model for RNAseq data
# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)
# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output)) #model_output  y = 1 / (1 + exp(-x))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
#Declare batch size
batch_size = 25
# Initialize outputs 
loss_vec = []
train_acc = []
test_acc = []
# Run training loop 
for i in range(100): #on delta run range(5000)
    # get random batch from training set 
    rand_index = np.random.choice(len(x_vals_rna_train), size=batch_size)
    rand_x = x_vals_rna_train[rand_index]
    rand_y = y_vals_train[rand_index]
    # run train step 
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # get loss 
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    # get acc for training 
    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_rna_train, y_target: y_vals_train})
    train_acc.append(temp_acc_train)
    # get acc for testing 
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_rna_test, y_target: y_vals_test})
    test_acc.append(temp_acc_test)
    # print running stat 
    if (i+1)%100==0:
        print('Loss = ' + str(temp_loss))
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))


# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation  Logistic')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()


# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy Logistic')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()