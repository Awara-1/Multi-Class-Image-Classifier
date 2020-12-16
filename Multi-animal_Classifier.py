#!/usr/bin/env python
# coding: utf-8

# In[9]:


from zipfile import ZipFile
import os

# Extract folder containing 101 categories of different objects
# UPDATE ACCORDINGLY
ZIP_NAME = '101_ObjectCategories.zip'
PARENT_DIR = 'Users/'
SOURCE_DIR = 'Users/101_ObjectCategories'
TRAIN_DIR = 'Users/TRAIN'
VALID_DIR = 'Users/VALID'

zf = ZipFile(ZIP_NAME, 'r')
print(zf)
zf.extractall(PARENT_DIR)
zf.close()


# In[10]:


# Take input from user in terms of how many classes for classifier
print('Please enter the number of classes you want to compare')
num_classes = int(input())

# Intialises classes list
classes = []

# Take folder name and create directory variables
for num in range(0, num_classes, 1):
    print('Please enter the file path of the class you want to train e.g. crab')
    classes.append('/' + input())


# In[11]:


print(classes)


# In[12]:


# Create directory variables

# Train directory
train_ext = 'TRAIN'
TRAIN_DIR = PARENT_DIR + train_ext
CLASSES_TRAIN_DIR = []

for num in range(0, num_classes, 1):
    file_path = TRAIN_DIR + classes[num]
    CLASSES_TRAIN_DIR.append(file_path)
    
print(CLASSES_TRAIN_DIR)

# Validation directory
valid_ext = 'VALID'
VALID_DIR = PARENT_DIR + valid_ext
CLASSES_VALID_DIR = []

for num in range(0, num_classes, 1):
    file_path = VALID_DIR + classes[num]
    CLASSES_VALID_DIR.append(file_path)
    
print(CLASSES_VALID_DIR)


# In[13]:


# Create directories using variables

# Create train and valid directories
try:
    os.mkdir(TRAIN_DIR)
    os.mkdir(VALID_DIR)
except OSError:
    pass

# Create each class's valid and train directories
for num in range(0, num_classes, 1):
    try:
        os.mkdir(CLASSES_TRAIN_DIR[num])
        os.mkdir(CLASSES_VALID_DIR[num])
    except OSError:
        pass


# In[14]:


# Create a function that copies files from source folder and splits into train and validation directories
import random
import shutil
from shutil import copyfile
from os import getcwd

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # Randomize list
    file_names = os.listdir(SOURCE)
    file_names = random.sample(file_names, len(file_names))
    
    # Obtain lengths
    train_len = int(SPLIT_SIZE*len(file_names))
    test_len = int((1-SPLIT_SIZE)*len(file_names))
    
    #Ensure un-corrupt file types get moved over
    for file_name in file_names[:train_len]:
        if os.path.getsize(SOURCE + '/' + file_name) > 0:
            copyfile(SOURCE + '/' + file_name, TRAINING + '/' + file_name)
        else:
            print("File" + file_name + "has no length")

    for file_name in file_names[:test_len+1]:
         if os.path.getsize(SOURCE + '/' + file_name) > 0:
            copyfile(SOURCE + '/' + file_name, TESTING + '/' + file_name)
            
for num in range(0, num_classes, 1):
    SOURCE = SOURCE_DIR + classes[num]
    split_data(SOURCE, CLASSES_TRAIN_DIR[num], CLASSES_VALID_DIR[num], 0.9)


# In[19]:


# Set-up transfer learning model with pre-trained weights to improve accuracy (use inception _v3)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3 as myModel

# Create an instance of the inception model from the local pre-trained weights
pre_trained_model = myModel(
    input_shape = (150, 150, 3),
    include_top = False,
    weights = "imagenet"
)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False
  
# Print the model summary
pre_trained_model.summary()


# In[20]:


# Get the pre-trained layer that we want to be the last layer (top) to input pre-trained weights into our model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[22]:


import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                    
# Add a final softmax layer for multi-classification
x = layers.Dense  (len(classes), activation='softmax')(x)              

custom_model = Model(pre_trained_model.input, x)

custom_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# custom_model.summary()


# In[24]:


# Use ImageGenerator to feed images in to model from directory

# Use Image augmentation
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1.0/255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    batch_size = 1,
                                                    class_mode='categorical',
                                                    target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  valid_datagen.flow_from_directory(VALID_DIR,
                                                          batch_size  = 1,
                                                          class_mode='categorical',
                                                          target_size = (150, 150))


# In[25]:


# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True


# In[26]:


callback = myCallback()

history = custom_model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 1,
            epochs = 100,
            validation_steps = 1,
            verbose = 2,
#             callbacks = [callback]  
    )


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:




