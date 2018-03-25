# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 18:30:19 2018

@author: Rajkumar
"""
import numpy as np
import os
from PIL import Image
from resizeimage import resizeimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray

def decode(data_l, data_ab):
    """
    This method combines the L from data_l and AB from data_ab.
    It returns a LAB image.
    """
    
    canvas = np.zeros((256, 256, 3))
    canvas[:,:,0] = data_l
    canvas[:,:,1:] = data_ab
    return canvas

def resize_image(img, width, height):
    """
    This method resizes the image.
    """
    with open(img, 'r+b') as f_image:
        with Image.open(f_image) as image:
            cover = resizeimage.resize_cover(image, [width, height])
            cover.save(img, image.format)
    return img

datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

def image_l_a_b_gen(train_data, batch_size):
    """
    This method is used to zig-zag the images and to separate LAB
    """
    for batch in datagen.flow(train_data, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        l_batch = lab_batch[:, :, :, 0]
        a_b_batch = lab_batch[:, :, :, 1:] / 128        
        yield (l_batch.reshape(l_batch.shape+(1,)), a_b_batch)
        
def get_train_data(folder):
    """
    This method returns the data required for training.
    """
    train_data = []     
    for filename in os.listdir(folder):
        train_data.append(img_to_array(load_img(resize_image(folder+filename, 256, 256))))
        train_data = np.array(train_data, dtype=float)
        train_data = 1.0/255*train_data    
        #print(train_data.shape)
    return train_data
    
def get_test_data(folder):
    """
    This method returns the data required for testing.
    """
    test_data = []
    for filename in os.listdir(folder):
        test_data.append(img_to_array(load_img(resize_image(folder+filename, 256, 256))))
    test_data = np.array(test_data, dtype=float)
    test_data = rgb2lab(1.0/255*test_data)
    Xtest = test_data[:, :, :, 0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = test_data[:,:,:,1:]
    Ytest /= 128
    #print(test_data.shape)
    return Xtest,Ytest


