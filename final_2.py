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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------------------------------------------

skin_df = pd.read_csv(r'/home/stud1/Nitish/HAM1000/dataverse_files/HAM10000_metadata.csv')
# print(skin_df['dx'])
# -----------------------------------------------------------------------------------------------------------------

SIZE=32
# -----------------------------------------------------------------------------------------------------------------

# label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
# print(list(le.classes_))
# -----------------------------------------------------------------------------------------------------------------


skin_df['label'] = le.transform(skin_df["dx"]) 
# print(skin_df.sample(10))

print(skin_df['label'].value_counts())
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('/home/stud1/Nitish/HAM1000/dataverse_files/All_Image', '*.jpg'))} 
# image_path
# -----------------------------------------------------------------------------------------------------------------
#Define the path and add as a new column

start_time = time.time()

skin_df['path'] = skin_df['image_id'].map(image_path.get)
#Use the path to read images.
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

end_time = time.time()
print('Total Time taken in Data handelling:', end_time - start_time)

# -----------------------------------------------------------------------------------------------------------------
#Convert dataframe column of images into numpy array
X = np.asarray(skin_df['image'].tolist())
X = X/255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=skin_df['label']  #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem and it converted labels into one hot encoding matrix
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.20, random_state=42)
x_training, x_validation, y_training, y_validation = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
print(len(x_training), len(x_validation), len(x_test))

# Y_cat

# ------------------------------------------------FUNCTIONAL_API_MODEL-----------------------------------------------------------------
#Define the model.
# SIZE=32
num_classes = 7
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


flaten_layer=Flatten()(layer4_3)


layer5=Dense(32)(flaten_layer)


output=Dense(7, activation='softmax')(layer5)



fun_model=Model(inputs=layer1, outputs=output, name='Functional_API_Model')
fun_model.summary()
fun_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# --------------------------------------------------SAVING MODEL WITH CALLBACK---------------------------------------------------------------
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("/home/stud1/Nitish/Ppaer_2/saved_Model/checkpoint", monitor='loss', verbose=1,
    save_best_only=False, mode='auto', save_freq=1)
# -------------------------------------------------TRAINING THE MODEL----------------------------------------------------------------
#You can also use generator to use augmentation during training.

batch_size = 16 
epochs = 100

# history = model.fit(
#     x_train, y_train,
#     epochs=epochs,
#     batch_size = batch_size,
#     validation_data=(x_test, y_test),
#     verbose=2)

# score = model.evaluate(x_test, y_test)
# print('Test accuracy:', score[1])
start_time = time.time()

history=fun_model.fit(x_training, y_training,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation),
          callbacks=[checkpoint])

end_time = time.time()
print('Total Time taken in Running the model:', end_time - start_time)

# # -------------------------------------------------SAVING MODEL WITH SECOND METHOD----------------------------------------------------------------+
# Saving the model 
from genericpath import isfile
import os.path
if os.path.isfile('/home/stud1/Nitish/Ppaer_2/saved_Model/final_2.hdf5') is False:
    fun_model.save('/home/stud1/Nitish/Ppaer_2/saved_Model/final_2.hdf5')
# ------------------------------------------------LOADING THE SAVED MODEL-----------------------------------------------------------------+
# from tensorflow.keras.models import load_model
# fun_model=load_model('/home/stud1/Nitish/HAM1000/Code/Model_Saved/functional_skip_0.hdf5')

# ---------------------------------------------HISTORY PRINTING--------------------------------------------------------------------+
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# -------------------------------------------HISTORY ACCURACY ----------------------------------------------------------------------+
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# -----------------------------------------------------------------------------------------------------------------+
# Prediction on test data
y_pred = fun_model.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 
# y_pred_classes.size
# y_true.size
# ----------------------------------------------ACCURACY MATRIX PRINTING-------------------------------------------------------------------+
#Print confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
# -----------------------------------------------------------------------------------------------------------------+
#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
# -----------------------------------------------ACCURACY PRINTING------------------------------------------------------------------+
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true, y_pred_classes, target_names=['Class 0', 'Class 1', 'Class 2','Class 3','Class 4','Class 5','Class 6']))

