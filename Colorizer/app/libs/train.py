# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:00:43 2018

@author: Rajkumar
"""

import numpy as np
import utils
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb
from Net import Net
from keras.models import load_model
from keras.models import model_from_json

# Get Train Data.
TRAIN_FOLDER = 'Train/'
TRAIN_DATA = utils.get_train_data(TRAIN_FOLDER)
TRAIN_DATA_SIZE = len(TRAIN_DATA)

# Get the CNN model
net = Net(train=True)
CNN = net.encode()

# Define BatchSize
BATCH_SIZE = 50

# Train model
train_batch = utils.image_l_a_b_gen(TRAIN_DATA, BATCH_SIZE)
CNN.fit_generator(train_batch, epochs=1000)

# Save model
model.save("model.h5")

'''
UnComment this to test the model
# Load model
model.load("model.h5")

# Test model
TEST_FOLDER = 'Test/'
Xtest_l, Ytest_a_b = utils.get_test_data(TEST_FOLDER)

# Evaluate model
print(CNN.evaluate(Xtest_l, Ytest_a_b, batch_size=BATCH_SIZE))

# Getting a and b 
test_a_b = CNN.predict(Xtest_l)
test_a_b *= 128

# Output colorizations
for i in range(len(test_a_b)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = Xtest_l[i][:, :, 0]
    cur[:, :, 1:] = test_a_b[i]
    imsave("img_"+str(i)+".png", lab2rgb(cur))
'''



