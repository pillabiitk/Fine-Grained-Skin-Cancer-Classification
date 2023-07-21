# Importing Required Libraries
import time
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Input, concatenate, UpSampling2D
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
#Define the model.
SIZE=32
num_classes = 3
input_shape=(SIZE,SIZE,3)

layer1=Input(shape=input_shape)

layer2=Conv2D(256, (3, 3), activation="relu")(layer1)
layer2_1=BatchNormalization()(layer2)
layer2_2=MaxPool2D(pool_size=(2, 2))(layer2_1)
layer2_3=Dropout(0.3)(layer2_2)


layer3=Conv2D(128, (3, 3), activation="relu")(layer2_3)
layer3_1=BatchNormalization()(layer3)
layer3_2=MaxPool2D(pool_size=(2, 2))(layer3_1)
layer3_3=Dropout(0.8)(layer3_2)


layer4=Conv2D(64, (3, 3), activation="relu")(layer3_3)
layer4_1=BatchNormalization()(layer4)
layer4_2=MaxPool2D(pool_size=(2, 2))(layer4_1)
layer4_3=Dropout(0.5)(layer4_2)


layera=Conv2D(64, (3, 3), activation="relu", padding='same')(layer4_3)
layera_1=BatchNormalization()(layera)
layera_2=MaxPool2D(pool_size=(2, 2), padding='same')(layera_1)
layera_3=Dropout(0.5)(layera_2)

layerb=Conv2D(64, (3, 3), activation="relu", padding='same')(layera_3)
layerb_1=BatchNormalization()(layerb)
layerb_2=MaxPool2D(pool_size=(2, 2),padding='same')(layerb_1)
layerb_3=Dropout(0.5)(layerb_2)

layerc=Conv2D(64, (3, 3), activation="relu", padding='same')(layerb_3)
layerc_1=BatchNormalization()(layerc)
layerc_2=MaxPool2D(pool_size=(2, 2), padding='same')(layerc_1)
layerc_3=Dropout(0.5)(layerc_2)
layerc_4=UpSampling2D(size = (2,2))(layerc_3)

#adding the both
layer_5=concatenate([layer4_3, layerc_4], axis=-1)
#adding early layer skip
layer_5_up=UpSampling2D(size = (15,15))(layer_5)
layer_6=concatenate([layer_5_up, layer2_1], axis=-1)


flaten_layer=Flatten()(layer_6)


layer5=Dense(32)(flaten_layer)


output=Dense(3, activation='softmax')(layer5)



fun_model=Model(inputs=layer1, outputs=output, name='Functional_API_Model')
fun_model.summary()
fun_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# from keras.layers import Input, concatenate 
# from keras.models import Model


  
# input_net = Input((32,32,3))
# ## Encoder starts
# conv1 = Conv2D(32, 3, strides=(2,2), activation = 'relu', padding = 'same')(input_net)
# conv2 = Conv2D(64, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv1)
# conv3 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv2)

# conv4 = Conv2D(128, 3, strides=(2,2), activation = 'relu', padding = 'same')(conv3)

# ## And now the decoder
# up1 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4))
# merge1 = concatenate([conv3,up1], axis = 3)
# up2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge1))
# merge2 = concatenate([conv2,up2], axis = 3)
# up3 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(merge2))
# merge3 = concatenate([conv1,up3], axis = 3)

# up4 = Conv2D(32, 3, padding = 'same')(UpSampling2D(size = (2,2))(merge3))

# output_net = Conv2D(3, 3, padding = 'same')(up4)

# model = Model(inputs = input_net, outputs = output_net)

# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
  