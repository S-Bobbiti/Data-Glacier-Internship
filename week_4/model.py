import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np


# Preprocess data
def preprocess(train_data,test_data,train_labels,test_labels):
  """
  Preprocssing the data. General steps needed for all models
  Args: train_data, test_data, train_labels, test_labels
  Returns: train_data, test_data, train_labels, test_labels
  """
  # convert training and testing images into float
  train_data = train_data.astype('float32')
  test_data = test_data.astype('float32')

  # converting the training and testing labels into one-hot encoded values
  test_labels = to_categorical(test_labels)
  train_labels = to_categorical(train_labels)

  # normalizing the data to 0 - 1 range
  train_data = train_data / 255
  test_data = test_data / 255

  return (train_data,test_data,train_labels,test_labels)

(train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()
(train_data,test_data,train_labels,test_labels) = preprocess(train_data,
                                                               test_data,
                                                               train_labels,
                                                               test_labels)

def cnn1_model(train_data, train_labels):
  """
  Creates the specified 5 layered CNN model
  Args: None
  Returns: model  
  """
  output_classes = train_labels.shape[1]
  assert output_classes == 10

  model = Sequential()
  # 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function
  model.add(layers.Conv2D(64, (3,3), activation='relu', padding='valid',
                          input_shape=(32, 32, 3)))
  # 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function
  model.add(layers.Conv2D(64, (3,3), activation='relu', padding='valid'))
  #Flattening to arrange the 3D volume of numbers into a 1D vector.
  model.add(layers.Flatten())
  # Fully connected (Dense) layer with 512 units and a sigmoid activation function
  model.add(layers.Dense(512, activation='sigmoid'))
  # Fully connected layer with 512 units and a sigmoid activation function
  model.add(layers.Dense(512, activation='sigmoid'))
  # Output layer with the suitable activation function and number of neurons for the classification task
  model.add(layers.Dense(10, activation = 'softmax'))
  # Using Adam optimizer
  model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_test_cnn1():
  """
  Args: None
  Return: trained_cnn1_model
  """
  
  # Create a new MLP model as per assignment requirements
  model = cnn1_model(train_data, train_labels)
  # Fit the model and evaluate
  trained_cnn1_model = model.fit(train_data, train_labels, epochs=5, 
                                batch_size=32, verbose=1, 
                                validation_data=(test_data, test_labels))
  
  return trained_cnn1_model

train_cnn1_model = train_test_cnn1()

