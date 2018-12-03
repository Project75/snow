# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 13:07:17 2018

@author: 124578
"""

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

#Initialize logging
import logging
LOG_FILENAME = 'logfile.log'
logging.basicConfig(filename='logfile.log',format='%(asctime)s %(levelname)s %(message)s',level=logging.DEBUG,filemode='w')


logging.info('Starting Tensorflow Diabetes Predictive Analyzer')


df = pd.read_csv('diabetes.csv')
logging.info('Sample Training data')
logging.info(df.head(2))

X = df[
    ["Pregnancies", 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

Y = df[['Outcome']]

df['NotDiabetes'] = 1 - df['Outcome']
y = df[['Outcome', 'NotDiabetes']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#learning_rate variable

learningRate = tf.train.exponential_decay(learning_rate=0.01,global_step=1,decay_steps=X_train.shape[0],decay_rate=0.95,staircase=True)

# Neural Network Parameters

n_hidden_1 = 64
n_hidden_2 = 32
n_hidden_3 = 8
n_input = X_train.shape[1]
n_classes = y_train.shape[1]
dropout = 0.5
beta=0.01

#Training parameters
training_epochs = 50
batch_size = 32
display_step = 10

log_vars='{}:{}:{}:{}:{}:{}:{}:{}'.format(n_input,n_hidden_1,n_hidden_2,n_hidden_3,str(int(dropout)*100)+'%',n_classes,batch_size,beta)
logging.info('Configurable Parameters:n_input,n_hidden_1,n_hidden_2,n_hidden_3,dropout,n_classes,batch_size,regularization:'+log_vars)

# TensorFlow Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")
keep_prob = tf.placeholder(tf.float32)
#regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

# Store Layers weight and bias

weights = {
    #'h1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1),minval=0,maxval=0.005,dtype=tf.float32, seed=0)),   #maxval=0.005 , def =1
    'h1': tf.Variable(tf.random_uniform(shape=(n_input, n_hidden_1), dtype=tf.float32)),
    'h2': tf.Variable(tf.random_uniform(shape=(n_hidden_1, n_hidden_2),dtype=tf.float32)),
    'h3': tf.Variable(tf.random_uniform(shape=(n_hidden_2, n_hidden_3),dtype=tf.float32)),
    'out': tf.Variable(tf.random_uniform(shape=(n_hidden_3, n_classes), dtype=tf.float32))
}

biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1])),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2])),
    'b3': tf.Variable(tf.random_uniform([n_hidden_3])),
    'out1': tf.Variable(tf.random_uniform([n_classes]))
}

# Create Neural Network model with 3 hidden layers
def neural_network(x, weights, biases,keep_prob):
    # Hidden layer with relu activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with relu activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Output Layer with neurons =  number of output classes
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out1']

    return out_layer

def calculate_cost(nn_model,weights):
    # Defining loss fucntion
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_model, labels=y))
    # Loss function using L2 Regularization
    regL2 = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3']) +tf.nn.l2_loss(weights['out'])
#   Adding regularization to cost fn avoides overfitting of training data
    cost = tf.reduce_mean(cost + beta*regL2)       
    return cost

# Constructing model from given weights,bias and inputs
pred = neural_network(x, weights, biases,dropout)

#calculate cost fucntion for the model
cost=calculate_cost(pred,weights)
 
#Optimization adam- adaptive moment estimation.Optimizer should minimize the cost fucntion
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)


# Evaluate model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Saver for the model
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(1,training_epochs+1):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization operation (backprop) and cost operation(to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0 or epoch ==1:
            batch_cost,acc_train = sess.run([cost,accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            print("Step:", '%d' % (epoch), "avg cost=", "{:.4f}".format(avg_cost)," accuracy= ","{:.3f}".format(acc_train))
            debugstr= 'epoch: '+ str(epoch+1) + ' ,cost= '+ format(avg_cost)+" , accuracy= "+format(acc_train)
            logging.debug(debugstr)

    
    acc_eval = accuracy.eval({x: X_test, y: y_test, keep_prob: 1})
    print("Test Accuracy:", acc_eval)
    logging.info("Test Accuracy:"+ str(acc_eval))
    #saver.save(sess, "diabtest2-model2")
    #saver.export_meta_graph('diabtest2-model.meta')
    
#Export Meta Graph for calling via API

#meta_graph_def = tf.train.export_meta_graph(filename='diabtest2-model.meta')

print("Done. Please see log files for details")