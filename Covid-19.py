import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers,losses
import numpy
import matplotlib.pyplot as plt

#step 1
BATCH_SIZE = 16

training_data_generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05)
validation_data_generator = ImageDataGenerator(
        rescale=1./255)

training_iterator = training_data_generator.flow_from_directory('augmented-data/train',class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)

validation_iterator = validation_data_generator.flow_from_directory('augmented-data/test',class_mode='categorical',color_mode='grayscale',batch_size = BATCH_SIZE)

#step 2

model = Sequential()
model.add(layers.Input(shape=training_iterator.image_shape))
model.add(layers.Conv2D(6,3,strides=2,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Conv2D(6,3,strides=2,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10,verbose=1)

history=model.fit(training_iterator,steps_per_epoch=training_iterator.samples/BATCH_SIZE,epochs =5,validation_data=validation_iterator,validation_steps=validation_iterator.samples/BATCH_SIZE,callbacks=[callback])

#step 3
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
