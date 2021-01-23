'''Script to resize all the images and merge them into one numpy array.
As the size of the dataset is small, it is a feasible way of handling it.
'''

import numpy as np
import cv2
import os
from PIL import Image

image_folder = 'JSRT'


IMG_X = 512 # target size X
IMG_Y = 512 # target size Y
N = 247 #number of input images
imageHeight = 2048
imageWidth = 2048
grayLevels = 4096

#data preparation
x = np.zeros((N, IMG_X, IMG_Y, 1))
y = np.zeros((N, IMG_X, IMG_Y, 1))
i = 0
for imageFilename in os.listdir(imageFolder):
    filename = os.path.splitext(imageFilename)[0] #splitting off the extension
    #read images
    imageArray = np.fromfile(os.path.join(imageFolder, filename + imageExtension),
                                  dtype='>u2')
    image = imageArray.reshape((2048, 2048))
    resizedImage = cv2.resize(image, dsize=(IMG_X, IMG_Y),
                              interpolation=cv2.INTER_LINEAR)
    #read left lungs
    leftLung = np.array(Image.open(os.path.join(leftLungFolder, filename + maskExtension)))
    resizedLeftLung = cv2.resize(leftLung, dsize=(IMG_X, IMG_Y),
                          interpolation=cv2.INTER_NEAREST)
    #read right lungs
    rightLung = np.array(Image.open(os.path.join(rightLungFolder, filename + maskExtension)))
    resizedRightLung = cv2.resize(rightLung, dsize=(IMG_X, IMG_Y),
                          interpolation=cv2.INTER_NEAREST)
    #calculate all mask areas
    resizedMask = np.where(resizedLeftLung + resizedRightLung > 0, 255, 0)
    #prepare x
    x[i, :, :, 0] = resizedImage
    #prepare y
    y[i, :, :, 0] = resizedMask
    #increase counter
    i = i+1
    if i % 10 == 0:
        print(i, "/", N)

np.save('x.npy', x)
np.save('y.npy', y)
print("Data preparation completed.")