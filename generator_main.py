import numpy as np

from keras.models import Sequential
from my_classes import DataGenerator

from create_dirs import create_dirs
import os, shutil
import numpy as np
import fileio
import aifc
import audio_processor as ap

from keras import layers
from keras import models

# Design model
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense

def generator_main():

  original_dataset_dir = 'C:\\svn\\dwatts\\dev\\datasets\\whale_data\\data'
  original_train_dataset_dir = 'C:\\svn\\dwatts\\dev\\datasets\\whale_data\\data\\train'
  base_dir = 'C:\\svn\\dwatts\\dev\\dl_with_python\\whale_small\\'

  # spectrogram parameters
  params = {'batch_size': 64,
            'dim': (48,126),
            'n_channels': 1,
            'n_classes': 2,
            'shuffle': True,
            'NFFT':64,
            'Fs':2000,
            'noverlap':32}

  # Number of time slice metrics
  maxTime = 126

  Ntrain = 5000
  Nval = 500
  Ntest = 500

  train_loc = 'whale_small/train'
  val_loc = 'whale_small/validation'
  # load data, parition and labels:
  # e.g. {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
  # e.g. {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
  #train = fileio.TrainData(original_dataset_dir+'\\train.csv',base_dir+'train')
  partition, labels = create_dirs(Ntrain, Nval, Ntest, original_dataset_dir, base_dir)

  # Generators
  training_generator = DataGenerator(partition['train'], labels, train_loc, **params)
  validation_generator = DataGenerator(partition['validation'], labels, val_loc, **params)

  for data_batch, labels_batch in training_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

  # Determine proper input shape
  input_shape = (params['dim'][0], params['dim'][1], params['n_channels'])

  '''
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu',
  input_shape=input_shape))
  model.add(layers.MaxPooling2D((2, 2),dim_ordering="th"))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2),dim_ordering="th"))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2),dim_ordering="th"))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2),dim_ordering="th"))
  model.add(layers.Flatten())
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(2, activation='softmax'))
  '''

  melgram_input = Input(shape=input_shape)

  # Only tf dimension ordering
  channel_axis = 3
  freq_axis = 1
  time_axis = 2

  # Input block
  x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)

  # Conv block 1
  x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
  x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
  x = ELU()(x)
  x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool1')(x)

  # Conv block 2
  x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
  x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
  x = ELU()(x)
  x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool2')(x)

  # Conv block 3
  x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
  x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
  x = ELU()(x)
  x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool3')(x)

  # Conv block 4
  x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
  x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
  x = ELU()(x)
  x = MaxPooling2D(pool_size=(3, 5), dim_ordering="th", name='pool4')(x)

  # Conv block 5
  x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
  x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
  x = ELU()(x)
  x = MaxPooling2D(pool_size=(4, 4), dim_ordering="th", name='pool5')(x)

  # Output
  x = Flatten()(x)
  x = Dense(50, activation='relu', name='hidden1')(x)
  x = Dense(2, activation='softmax', name='output')(x)

  # Create model
  model = Model(melgram_input, x)


  model.summary()

  from keras import optimizers

  # Compile the model
  model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])

  # Train model on dataset
  history = model.fit_generator(
    training_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)
    #use_multiprocessing=True,
    #workers=1)

  model.save('whale_small_1.h5')

  import matplotlib.pyplot as plt
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  generator_main();
