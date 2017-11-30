
# coding: utf-8

# In[1]:

## Import libraries
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
import numpy as np

np.random.seed(0)


# In[2]:

## load_mnist data function is being implemented by zalando in git 
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# Load data
X_train, y_train = load_mnist('newinput', kind='train')
X_test, y_test = load_mnist('newinput', kind='t10k')



# In[3]:

# Prepare datasets
# This step contains reshaping of input. 

X_train = X_train.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

## Changing number to one-hot vector.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[10]:

drop_prob_1 = 0.25 

# creating model
clf = Sequential()

# define input_shape.
clf.add(InputLayer(input_shape=(1, 28, 28)))

## Adding convolution layer to model. 

## layer -1
clf.add(Conv2D(32, kernel_size=(4, 4),strides = (1,1),activation='relu',padding="same"))

##layer -2
clf.add(Conv2D(64, kernel_size=(4, 4),strides = (2,2),activation='relu',padding="same"))

##layer -3
clf.add(Conv2D(128, kernel_size=(4, 4),strides = (2,2),activation='relu',padding="same"))

# Flatten input data to a vector.
clf.add(Flatten())

# Fully-connected layers.
clf.add(Dense(512,activation='relu',bias_initializer=Constant(0.01), kernel_initializer='random_uniform' ))
clf.add(Dropout(drop_prob_1))

optimizer = Adam(lr=0.001, decay=0.00001)
# optimizer = SGD(lr=0.001, decay=0.00001)

# Add output layer, which contains ten numbers.
# Each number represents cloth type.
clf.add(Dense(10, activation='softmax'))

# compile model.
clf.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])


# In[11]:

print(clf.summary())


# In[12]:

clf.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=128, 
    validation_data=(X_test, y_test)
)


# In[13]:

clf.evaluate(X_test, y_test)


# ## 88.98 % accuracy 
# - Using 3 conv layers [stride =1 and rest of the layers with stride =2] and 1 FC layer with lr decay in Adam Optimizer and a dropout on FC layer.
# 
# ## 88.81 % accuracy
# - Using 3 conv layers [stride =1 and rest of the layers with stride =2] and 1 FC layer with lr decay in SGD and a dropout on FC layer.
# 
# Accuracy increased mostly because of introducing stride, drop out and lr decay in Adam  
# 

# ## Output structure of the conv layer based on given stride and padding depends on the following formula:
# 
# - Let (nxn) be the input, (fxf) be the filter p = padding and s = stride 
# - Then =>  [((n+2p-f)/s)+1 ] x [((n+2p-f)/s)+1 ] 
# 
# - So in a way 28x28 gets reduced to 14x14 with stride = 2 and in the next layer with addiiton stride = 2 it gets reduced to 7x7

# In[ ]:



