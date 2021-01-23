'''Script to carry out CLAHE on the numpy array containing all the X-ray images.
'''

import numpy as np
import cv2

# Fix input and output paths
xray_path = 'x.npy'
output_path = 'clahe_x.npy'

x = np.load(xray_path)

clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8,8))

clahe_x = np.zeros(x.shape, dtype=float)

for i in range(x.shape[0]):
    clahe_x[i, :, :, 0] = (clahe.apply((x[i, :, :, 0] * (2 ** 16)).astype('uint16')) / 2 ** 16).astype('float32')
    
    
np.save(output_path, clahe_x)