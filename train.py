''' Script to run trainings.
'''

import numpy as np
import tensorflow as tf
import functools
import pickle
from tensorflow.model.callbacks import ModelCheckpoint

from losses import gen_focal_tversky, softIoU, dsc 
from models import attn_bcdu_net
from datagen import MixupGenerator, ValGenerator

image_path = 'clahe_x.npy'
segmentation_path = 'y.npy'
log_string = 'JSRT_model_logs/attn_bcdu_net'

input_size = (512,512,1)
train_size=10
starting_filter_size = 48
depth = 2
center_length = 3
epochs = 1000
alpha = 0.6
gamma = 0.675
delta = 0.2

x = np.load("clahe_x.npy")
y = np.load("y.npy")
focal_tversky = functools.partial(gen_focal_tversky, alpha=alpha, gamma=gamma)
focal_tversky = functools.update_wrapper(focal_tversky, gen_focal_tversky)

p = np.random.permutation(len(x))
x = x[p]
y = y[p]
x_train, x_val, y_train, y_val = x[:train_size], x[209:], y[:train_size], y[209:]
training_generator = MixupGenerator(x_train, y_train, batch_size = 5, delta=delta)
validation_generator = ValGenerator(x_val, y_val, batch_size = 1)

unet_model = attn_bcdu_net(input_size, starting_filter_size, depth, center_length, spatial_channel_attn=False)
unet_model.compile(loss=focal_tversky,optimizer=tf.keras.optimizers.Adam(lr = 0.0005), 
                   metrics=[softIoU, dsc])

checkpoint = ModelCheckpoint(log_string+'_val_dsc_{val_dsc:03f}.h5',
                             monitor='val_dsc', verbose=1, save_best_only=True, mode='max')

history = unet_model.fit_generator(generator=training_generator, 
                                   callbacks=[checkpoint]
                                   epochs=epochs,
                                   verbose=1, 
                                   shuffle=True, 
                                   validation_data=validation_generator)

pickle.dump(history.history, open(log_string+".p", "wb"))