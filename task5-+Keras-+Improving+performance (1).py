
# coding: utf-8

# In[1]:

## Import libraries
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras import regularizers
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
# This step contains normalization and reshaping of input. 

X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

## Changing number to one-hot vector.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[4]:

drop_prob_1 = 0.25 
drop_prob_2 = 0.5

# creating model
clf = Sequential()

# define input_shape.
clf.add(InputLayer(input_shape=(1, 28, 28)))


# Normalize the activations of the previous layer at each batch.
clf.add(BatchNormalization())

## Adding convolution layer to model. 

## layer -1
clf.add(Conv2D(32, kernel_size=(4, 4),activation='relu',padding="same"))
# clf.add(MaxPool2D(padding='same')) # Add max pooling layer for 2D data.
# clf.add(Dropout(drop_prob_1))

##layer -2
clf.add(Conv2D(64, kernel_size=(4, 4),activation='relu',padding="same"))
clf.add(MaxPool2D(padding='same')) # Add max pooling layer 
# clf.add(Dropout(drop_prob_1))

##layer -3
clf.add(Conv2D(128, kernel_size=(4, 4),activation='relu',padding="same")) 
clf.add(MaxPool2D(padding='same')) # Add max pooling layer 
# clf.add(Dropout(drop_prob_1))

##layer -4
clf.add(Conv2D(128, kernel_size=(4, 4),activation='relu',padding="same")) 
clf.add(MaxPool2D(padding='same')) # max pooling
# clf.add(Dropout(drop_prob_1))

# Flatten input data to a vector.
clf.add(Flatten())
clf.add(Dropout(drop_prob_1))

# Fully-connected layers.
clf.add(Dense(512,activation='relu',bias_initializer=Constant(0.01), 
              kernel_initializer='random_uniform'))

clf.add(Dropout(drop_prob_2))

optimizer = Adam(lr=0.001, decay=0.00001)

# Add output layer, which contains ten numbers.
# Each number represents cloth type.
clf.add(Dense(10, activation='softmax'))

# compile model.
clf.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])


# In[5]:

print(clf.summary())


# In[6]:

clf.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=128, 
    validation_data=(X_test, y_test)
)


# In[7]:

clf.evaluate(X_test, y_test)


# ## Accuracy - 90.61 % 
# - Using 4 layer conv network, 1 FC layer, drop out of 0.25 and 0.5, maxpool layer, lr decay using Adam Optimizer

# In[ ]:



