
# coding: utf-8

# In[1]:

# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data
fashion_mnist = input_data.read_data_sets('newinput', one_hot=True)

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])

# five layers and number of neurons
l1 = 200
l2 = 100
l3=  60
l4 = 30

W1 = tf.Variable(tf.truncated_normal([784, l1], stddev=0.1)) # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([l1]))
W2 = tf.Variable(tf.truncated_normal([l1, l2], stddev=0.1)) 
B2 = tf.Variable(tf.zeros([l2]))
W3 = tf.Variable(tf.truncated_normal([l2, l3], stddev=0.1)) 
B3 = tf.Variable(tf.zeros([l3]))
W4 = tf.Variable(tf.truncated_normal([l3, l4], stddev=0.1)) 
B4 = tf.Variable(tf.zeros([l4]))
W5 = tf.Variable(tf.truncated_normal([l4, 10], stddev=0.1)) 
B5 = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])

# 2. Define the model

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)

Ylogits = tf.matmul(Y4, W5) + B5

Y = tf.nn.softmax(Ylogits)


# 3. Define the loss function  

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# 4. Define the accuracy 
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={X: fashion_mnist.test.images, Y_: fashion_mnist.test.labels}))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):

   # print("\r", i)
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = fashion_mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: fashion_mnist.test.images, Y_: fashion_mnist.test.labels})
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


# ## Accuracy - 88.75 %
# **Softmax** helps in converting a K-dimentional real arbitrary valued vector to a K-dimentional vector of values in the range 0 to 1. It helps in finding the right output which has highest probability when there are more than 2 elements in the last layer. The summation of all the output values is 1. If the output layer has 2 classes the softmax layer becomes a logistic regression.

# In[ ]:



