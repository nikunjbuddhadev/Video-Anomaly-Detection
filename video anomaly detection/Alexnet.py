
import pandas as pd
import numpy as np
import cv2
import os
import scipy
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



#defining file path
DA = os.listdir("C:\datasets\data\project\daily activity")
fall = os.listdir("C:\datasets\data\project\FA cam1")
filepath="C:\datasets\data\project\daily activity"
filepath2="C:\datasets\data\project\FA cam1"



images = []
label = []
#loading daily activity images
for i in DA:
    image = scipy.misc.imread( os.path.join(filepath,i))
    images.append(image)
    label.append(0) #for daily activity images    


#loading fall activity images
for i in fall:
    image = scipy.misc.imread( os.path.join(filepath2,i))
    images.append(image)
    label.append(1) #for fall activity images
    

for i in range(0,118):
    images[i] = cv2.resize(np.array(images[i]),(224,224))
    

images=np.array(images)
label=np.array(label)
label = to_categorical(label)


#Splitting data into training and testing set
Xtrain, Xtest, ytrain, ytest = train_test_split(images, label, test_size=0.2)



# (3) Create a sequential model
model = Sequential()

model.add(keras.layers.InputLayer(input_shape = (224,224,3)) )


# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# data augumentation 
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

datagen.fit(Xtrain)

model.fit_generator(datagen.flow(Xtrain, ytrain, batch_size=10),
                    steps_per_epoch=len(Xtrain) // 10, epochs=6)




#prediction on test split .
scores = model.predict(Xtest,verbose=1)
score = pd.DataFrame(scores)
nn = score.apply(np.argmax,axis=1)
ytest = pd.DataFrame(ytest)
n = ytest.apply(np.argmax,axis=1)
error = pd.concat([n,nn],axis=1)
error['error'] = error[0] - error[1]
efficiency_CNN = error.ix[error['error']==0,:].shape[0]/error.shape[0]

