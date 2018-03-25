# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:21:41 2018

@author: Rajkumar
"""
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import load_model

class Net:
    def __init__(self, train=True):
        self.train=train
    
    def encode(self):
        if self.train == True:
            return self.net()
        else:
            return self.loaded_model()
    
    def net(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.compile(optimizer='rmsprop', loss='mse')
        return model
    
    def loaded_model(self):
        return load_model("model.h5")        
        
        
        
        
        
        