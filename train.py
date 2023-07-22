import numpy as np
import cv2
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# working with the data
train_dir = 'data/train' #create a train directory
val_dir = 'data/test' #create validation directory
train_datagen = ImageDataGenerator(rescale=1./255) #picture in pixcels with values ranging from 1 to 255 in an array
val_datagen = ImageDataGenerator(rescale=1./225)

#create generator
train_generator = train_datagen.flow_from_directory (
                train_dir,
                target_size=(48,48), #target size of data
                batch_size=64, #refers to the no of training examples utilized in one iteration
                color_mode="grayscale", #grayscaled images for black and white
                class_mode='categorical') #different actegories of emotions
validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(48,48),
                batch_size=64,
                color_mode="grayscale",
                class_mode='categorical')

#creation of model and prepare it for training

#models 2 types-sequential & functional
#sequential api - create models layer by layer but not suitable for models having multiple outputs ot inputs
#functiional api - more flexible having interconnection between layers but more complex

#inilialization iof a sequential model
emotion_model = Sequential()
#layes are added by add
#Conv2D-creates a conv kernel convolved with layer input to prduce a tensor of outputs
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50, #50
    validation_data = validation_generator,
    validation_steps=7178 // 64)



emotion_model.save_weights('model.h5')