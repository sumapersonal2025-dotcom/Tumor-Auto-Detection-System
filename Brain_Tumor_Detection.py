#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('Training'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('Testing'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


get_ipython().system('pip install keras scikit-learn')


# In[4]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score


# In[5]:


import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf


# In[6]:


X_train=[]
Y_train=[]
image_size=150
labels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath=os.path.join('Training',i)
    for j in os.listdir(folderPath):
        img=cv2.imread(os.path.join(folderPath,j))
        img=cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
for i in labels:
    folderPath=os.path.join('Testing',i)
    for j in os.listdir(folderPath):
        img=cv2.imread(os.path.join(folderPath,j))
        img=cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
X_train=np.array(X_train)
Y_train=np.array(Y_train)


# In[7]:


X_train,Y_train=shuffle(X_train,Y_train,random_state=101)
print(X_train.shape)
print(Y_train.shape)


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.1,random_state=101)


# In[9]:


y_train_new=[] 
for i in y_train: 
    y_train_new.append(labels.index(i)) 
y_train=y_train_new
y_train=tf.keras.utils.to_categorical(y_train)

y_test_new=[] 
for i in y_test:
    y_test_new.append(labels.index(i)) 
y_test=y_test_new 
y_test=tf.keras.utils.to_categorical(y_test)


# In[10]:


model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))


# In[11]:


model.summary()


# In[12]:


from keras.losses import categorical_crossentropy
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[13]:


history=model.fit(X_train,y_train,epochs=20,validation_split=0.1)


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


#model.save(p (2).jpg)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(len(acc))
fig=plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[16]:


#model.save(p (2).jpg)
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(loss))
fig=plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.legend(loc='upper left')
plt.show()


# In[17]:


img=cv2.imread('1.jpg')
img=cv2.resize(img,(150,150))
img_array=np.array(img)
img_array.shape


# In[18]:


img_array=img_array.reshape(1,150,150,3)
img_array.shape


# In[19]:


from tensorflow.keras.preprocessing import image
img=image.load_img('1.jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[20]:


# a=model.predict(img_array)
# indices=a.argmax()
# indices 
# Predict the class indices
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)

# Define class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Map class indices to class names
predicted_class_names = [class_names[index] for index in predicted_class_index]

# Print the predicted class names
print(predicted_class_names)


# In[34]:


from sklearn.metrics import confusion_matrix
# Predicting the probabilities for test data
y_pred_probs_cnn = model.predict(X_test)

# Converting probabilities to class labels
y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)


# Computing the confusion matrix
conf_matrix_cnn = confusion_matrix(np.argmax(y_test, axis=1), y_pred_cnn)

# Displaying the confusion matrix
print("Confusion Matrix for CNN Model:")
print(conf_matrix_cnn)


# In[21]:


# Importing the VGG16 model
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout


# Loading the pre-trained VGG16 model without the fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Adding custom fully connected layers on top of VGG16
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Plotting training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[31]:


from sklearn.metrics import confusion_matrix
# Predicting the probabilities for test data
y_pred_probs_vgg16 = model.predict(X_test)

# Converting probabilities to class labels
y_pred_vgg16 = np.argmax(y_pred_probs_vgg16, axis=1)

# Computing the confusion matrix
conf_matrix_vgg16 = confusion_matrix(np.argmax(y_test, axis=1), y_pred_vgg16)

# Displaying the confusion matrix
print("Confusion Matrix for VGG16 Model:")
print(conf_matrix_vgg16)


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

# Defining class labels based on your dataset
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Provided confusion matrix values
conf_matrix_values = [[66, 17, 5, 1],
                      [6, 73, 7, 4],
                      [1, 6, 48, 0],
                      [7, 3, 0, 83]]

# Plotting the confusion matrix for the VGG16 model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_values, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for VGG16 Model')
plt.show()


# In[4]:


from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
# Define the confusion matrix
conf_matrix_vgg16 = np.array([[66, 17, 5, 1],
                              [6, 73, 7, 4],
                              [1, 6, 48, 0],
                              [7, 3, 0, 83]])

# Define the list of target class labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'] # Replace with your actual class labels

# Calculate true positives, false positives, false negatives for each class
tp = np.diag(conf_matrix_vgg16)
fp = np.sum(conf_matrix_vgg16, axis=0) - tp
fn = np.sum(conf_matrix_vgg16, axis=1) - tp

# Calculate precision, recall, and F1-score for each class
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Calculate support (number of true instances for each class)
support = np.sum(conf_matrix_vgg16, axis=1)

# Generate the classification report
report_data = {
    'precision': precision,
    'recall': recall,
    'f1-score': f1_score,
    'support': support
}
report_df = pd.DataFrame(report_data, index=labels)

# Print the classification report
print(report_df)


# In[ ]:




