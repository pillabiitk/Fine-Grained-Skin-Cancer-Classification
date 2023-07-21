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

# skin_df = pd.read_csv(r'/home/stud1/Nitish/Ppaer_2/dataverse_files/HAM10000_metadata.csv')
# # print(skin_df['dx'])
# -----------------------------------------------------------------------------------------------------------------

SIZE=32
# -----------------------------------------------------------------------------------------------------------------
# # label encoding to numeric values from text
# le = LabelEncoder()
# le.fit(skin_df['dx'])
# LabelEncoder()
# # print(list(le.classes_))
# -----------------------------------------------------------------------------------------------------------------

# skin_df['label'] = le.transform(skin_df["dx"]) 
# # print(skin_df.sample(10))
# -----------------------------------------------------------------------------------------------------------------
# Distribution of data into various classes 
# print(skin_df['label'].value_counts())
from sklearn.utils import resample

# #Now time to read images based on image ID from the CSV file
# #This is the safest way to read images as it ensures the right image is read for the right ID
# # image_path = {os.path.splitext(os.path.basename(x))[0]: x
# #                      for x in glob(os.path.join('data/HAM10000/', '*', '*.jpg'))}
# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#                      for x in glob(os.path.join('/home/stud1/Nitish/Ppaer_2/dataverse_files/All_Image', '*.jpg'))} 
# # print(image_path)

# -----------------------------------------------------------------------------------------------------------------
# #Define the path and add as a new column

# start_time = time.time()

# skin_df['path'] = skin_df['image_id'].map(image_path.get)
# #Use the path to read images.
# skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

# end_time = time.time()
# print('Total Time taken in Data handelling:', end_time - start_time)


# # saving the dataframe
# skin_df.to_csv('/home/stud1/Nitish/Ppaer_2/dataverse_files/updated.csv')
# -----------------------------------------------------------------------------------------------------------------
# load the saved csv file
skin_df = pd.read_csv(r'/home/stud1/Nitish/Ppaer_2/dataverse_files/updated.csv')
# print(skin_df['dx'])
# print(skin_df.sample(10))
# -----------------------------------------------------------------------------------------------------------------
# n_samples = 5  # number of samples for plotting
# # Plotting
# fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
# for n_axs, (type_name, type_rows) in zip(m_axs, 
#                                          skin_df.sort_values(['dx']).groupby('dx')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')
# print('plotted')
# -----------------------------------------------------------------------------------------------------------------
#Convert dataframe column of images into numpy array
X = np.asarray(skin_df['image'].tolist())
print(type(X[1]))
# X[0] = X[0]/255 # Scale values to 0-1. You can also used standardscaler or other scaling methods.
# Y=skin_df['label']  #Assign label values to Y
# Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem and it converted labels into one hot encoding matrix
# print(Y_cat)
#Split to training and testing
# x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)
# Y_cat


