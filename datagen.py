''' Source file containing the mixup data generator and a validation data generator.
The latter does not use mixup, it simply fetches data from the numpy arrays created by clahe.py.
In the no mixup setup, this validation data generator is used for training as well.
'''

import tensorflow as tf
import numpy as np

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.X_train.shape[0] / (2* self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        #generate batch indices
        batch_indices = self.indices[2 * index * self.batch_size : 2 * (index + 1) * self.batch_size]
        
        #generate data
        x_batch, y_batch = self.__data_generation(batch_indices)
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.random.permutation(self.X_train.shape[0])

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1, 1, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

class ValGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_val, y_val, batch_size=1, shuffle=True):
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(X_val)
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.X_val.shape[0] / (self.batch_size)))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        #generate batch indices
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        #generate data
        x_batch, y_batch = self.__data_generation(batch_indices)
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.random.permutation(self.X_val.shape[0])

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_val.shape

        X = self.X_val[batch_ids]
        y = self.y_val[batch_ids]    

        return X, y