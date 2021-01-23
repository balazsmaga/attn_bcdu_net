''' The source file containing the implementation of auxiliary layers and the final model

'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ConvLSTM2D, Dense
from tensorflow.keras.layers import Activation, add, multiply
from tensorflow.keras.layers import AveragePooling2D, average, UpSampling2D, Dropout, Reshape
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

def AttnGateLSTM(x, g, inter_shape, name):    
    '''
    Implementation of the attention gate used in the Attention BCDU-Net.
    '''
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16

    theta_x = tf.expand_dims(theta_x, axis=1)
    upsample_g = tf.expand_dims(upsample_g, axis=1)
    concat_xg = concatenate([upsample_g, theta_x], axis = 1)
    concat_xg = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(concat_xg)
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expand_as(upsample_psi, shape_x[3],  name)
    y = multiply([upsample_psi, x], name='q_attn'+name)

    result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn

def UnetConv2D(input, outdim, is_batchnorm, name):
    '''
    Implementation of the pair of convolutional layers used by the Attention BCDU-Net:
    two 3x3 convolution, between them a potential batch normalization and ReLU activation.
    '''
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer="glorot_normal", padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x =BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu',name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer="glorot_normal", padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x	

def UnetGatingSignal2D(input, is_batchnorm, name):
    '''
    Implementation of the gating signal appearing in the decoder of the Attention BCDU-Net:
    simply a 1x1 convolution followed by batch normalization and ReLU.
    '''
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer="glorot_normal", name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name = name + '_act')(x)
    return x
    
def SpatialAttention2D(input_tensor, filter_number, name):
    '''
    Implementation of Spatial Attention layer as inspired by SCAU-Net. 
    '''
    channel_number = input_tensor.shape[-1]
    aggregate = tf.math.reduce_sum(input_tensor, axis=-1, keepdims=True, name = name + '_agg')
    spatial_attn_conv1 = Conv2D(filter_number, (3, 3), activation='relu', padding='same', name = name + '_conv1')(aggregate)
    spatial_attn_conv2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = name + '_conv2')(spatial_attn_conv1)
    extended = tf.repeat(spatial_attn_conv2, repeats=channel_number, axis=3, name = name + '_ext')
    output = tf.keras.layers.multiply([input_tensor, extended])
    
    return output

def ChannelAttention2D(input_tensor, dense_unit_num, name):
    '''
    Implementation of Channel Attention layer as inspired by SCAU-Net. 
    '''
    channel_number = input_tensor.shape[-1]
    aggregate = tf.math.reduce_sum(input_tensor, axis=[1,2], keepdims=False, name = name + '_agg')
    hidden = Dense(dense_unit_num, activation='relu', name = name + '_dense1')(aggregate)
    weight_vector = Dense(channel_number, activation='sigmoid', name = name + '_dense2')(hidden)
    output = tf.keras.layers.multiply([input_tensor, weight_vector])
    
    return output
    


def bcdu_attn_unet(input_size, starting_filter_size=48, depth=2, center_length=3, spatial_channel_attn=False):
    '''
    The architecture of the Attention BCDU-Net with tuneable hyperparameters.
    
    starting_filter_size: int
        Filter_number of the first conv layers (gets doubled in each level).
        
    depth: int 
        Number of pairs of layers in the encoder.
        
    center_length: int
        Number of conv layer pairs in the center.
        
    spatial_channel_attn: boolean
        Whether to use additional spatial and channel attention layers.
    '''
    inputs = Input(shape=input_size)

    X = inputs

    downsampling_path = []

    for i in range(depth):
        level_filters = starting_filter_size * (2 ** i)

        conv1 = UnetConv2D(X, level_filters, is_batchnorm=True, name='conv1' + str(i))
        if i == 0 and spatial_channel_attn:
            conv1 = SpatialAttention2D(conv1, 16, name='spatial_attn1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetConv2D(pool1, level_filters, is_batchnorm=True, name='conv2' + str(i))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        downsampling_path.append(conv1)
        downsampling_path.append(conv2)
        X = pool2

    for i in range(center_length):
        level_filters = starting_filter_size * (2 ** depth)

        X = UnetConv2D(X, level_filters, is_batchnorm=True, name='center' + str(i))
    
    if spatial_channel_attn:
        X = ChannelAttention2D(X, 64, name='channel_attn')

    for i in range(depth - 1, 0 - 1, -1):
        level_filters = starting_filter_size * (2 ** i)

        g1 = UnetGatingSignal2D(X, is_batchnorm=True, name='g1' + str(i))
        skip1 = downsampling_path.pop()
        attn1 = AttnGateLSTM(skip1, g1, level_filters, '_1' + str(i))
        up1 = concatenate([Conv2DTranspose(level_filters, (3, 3), strides=(2, 2), padding='same',
                                           activation='relu', kernel_initializer="glorot_normal")(X), attn1],
                          name='up1' + str(i))

        g2 = UnetGatingSignal2D(X, is_batchnorm=True, name='g2' + str(i))
        skip2 = downsampling_path.pop()
        attn2 = AttnGateLSTM(skip2, g2, level_filters, '_2' + str(i))
        up2 = concatenate([Conv2DTranspose(level_filters, (3, 3), strides=(2, 2), padding='same',
                                           activation='relu', kernel_initializer="glorot_normal")(up1), attn2],
                          name='up2' + str(i))

        X = up2

    if spatial_channel_attn:
        X = SpatialAttention2D(X, 16, name='spatial_attn2')
    out = Conv2D(4, (1, 1), activation='sigmoid', kernel_initializer="glorot_normal", name='final')(X)

    model = Model(inputs=[inputs], outputs=[out])
    return model