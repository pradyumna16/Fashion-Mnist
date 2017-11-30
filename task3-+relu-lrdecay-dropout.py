
# coding: utf-8

# In[1]:

# all tensorflow api is accessible through this
import tensorflow as tf        
# Arrays and math exp
import numpy as np
import math
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data
mnist = input_data.read_data_sets('newinput', one_hot=True)


# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
starting_lr = 0.01
learning_rate = tf.train.exponential_decay(starting_lr, global_step, 100000, 0.96, staircase=True)


# In[2]:

# Weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# 2. Define the model
XX = tf.reshape(X, [-1, 784])

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

YLogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(YLogits)


# 3. Define the loss function  
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=YLogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

# 4. Define the accuracy 
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. Define an optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step = global_step)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):

    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y, pkeep: 0.75})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y, pkeep: 1.0})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        test_a.append(a)
        test_c.append(c)

    
    return (train_a, train_c, test_a, test_c)

# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []
train_c = []
test_a = []
test_c = []
    
training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

# 7. Plot and visualise the accuracy and loss

# accuracy training vs testing dataset
plt.plot(train_a)
plt.plot(test_a)
plt.grid(True)
plt.show()

print(test_a)

# loss training vs testing dataset
plt.plot(train_c)
plt.plot(test_c)
plt.grid(True)
plt.show()


# ## Accuracy of 77.39 %
# - dropout, learning decay, Adam optimizer
# 
# **Dropout** regularization helps in removing some nodes randomly or hidden units in each layer also removes the links between them. 
# pkeep is the probability that the given hidden unit can be kept. For ex: pkeep = 0.75, it means probability that a given hidden unit will be kept, i.e 0.25 probability of eliminating a unit. It helps in shrinking the weight and thus helps regularization from overfitting.
# 
# **Learning rate decay** - it is used to speed up the algorithm, we slowly reduce the learning rate over time. As learning rate becomes smaller the steps to take to the local minima will be smaller hence we tend to occilate in a tighter region in the minima. Thus during initial phase of learning when learning rate is bigger/ larger, the algorithm takes larger steps to come to the minima and slowly as we reduce the learning rate it take smaller steps to come closer to the minima and thus sticks around closely in a tighter region. 

# In[ ]:



