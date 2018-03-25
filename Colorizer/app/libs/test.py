import tensorflow as tf
from app.tf.utils import *
from app.tf.net import Net
from skimage.io import imsave
from skimage.transform import resize
import cv2

def guess():

    img = cv2.imread('static/gray.jpg')
    if len(img.shape) == 3:
      img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = img[None, :, :, None]
    data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    #data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    autocolor = Net(train=False)

    model = autocolor.encode(data_l)

    img_rgb = decode(data_l, model)
    imsave('static/color.jpg', img_rgb)
