''' Implementation of the losses and metrics used during trainings and evaluations:
DSC, Tversky, Focal Tversky, and IoU. Focal Tversky contains tuneable hyperparameters.

True negative, true positive, false negative, false positive rates are also implemented.
'''

import tensorflow.keras.backend as K
import numpy as np

def dsc(y_true, y_pred):
    smooth = 0.1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) 
                                            + smooth)
    return score

def np_dsc(y_true, y_pred):
    smooth = 0.1
    intersection = y_true * y_pred
    score = (2. * np.sum(intersection)) / (np.sum(y_true) + np.sum(y_pred) 
                                            + smooth)
    return score

def tp(y_true, y_pred):
    smooth = 0.1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 0.1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def fp(y_true, y_pred):
    smooth = 0.1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp = (K.sum(y_neg * y_pred_pos) + smooth)/ (K.sum(y_pred) + smooth) 
    return fp 

def fn(y_true, y_pred):
    smooth = 0.1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    fn = (K.sum(y_pos * y_pred_neg) + smooth) / (K.sum(y_pred_neg) + smooth )
    return fn

def gen_tversky(y_true, y_pred, alpha=0.7):
    smooth = 0.1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    #y_pred_pos=correct_mask(y_pred_pos,tresh=0.5)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def gen_tversky_loss(y_true, y_pred, alpha=0.7):
    return 1 - gen_tversky(y_true, y_pred, alpha)

def gen_focal_tversky(y_true, y_pred, sample_weight=None, alpha=0.7, gamma=0.75):
    pt_1 = gen_tversky(y_true, y_pred, alpha)
    return K.pow((1-pt_1), gamma)
    
def leftLungSoftIoU(groundTruth, prediction):
    left_lung_truth = groundTruth[:, :, :, 0]
    left_lung_pred = prediction[:, :, :, 0]
    scalarProduct = tf.einsum('ij,ij->i', left_lung_truth, left_lung_pred)
    squaredNorm1 = tf.norm(left_lung_truth, axis=-1) ** 2
    squaredNorm2 = tf.norm(left_lung_pred, axis=-1) ** 2
    return K.mean(scalarProduct / (squaredNorm1 + squaredNorm2 - scalarProduct),
                  axis=-1)

def rightLungSoftIoU(groundTruth, prediction):
    right_lung_truth = groundTruth[:, :, :, 1]
    right_lung_pred = prediction[:, :, :, 1]
    scalarProduct = tf.einsum('ij,ij->i', right_lung_truth, right_lung_pred)
    squaredNorm1 = tf.norm(right_lung_truth, axis=-1) ** 2
    squaredNorm2 = tf.norm(right_lung_pred, axis=-1) ** 2
    return K.mean(scalarProduct / (squaredNorm1 + squaredNorm2 - scalarProduct),
                  axis=-1)

def heartSoftIoU(groundTruth, prediction):
    heart_truth = groundTruth[:, :, :, 2]
    heart_pred = prediction[:, :, :, 2]
    scalarProduct = tf.einsum('ij,ij->i', heart_truth, heart_pred)
    squaredNorm1 = tf.norm(heart_truth, axis=-1) ** 2
    squaredNorm2 = tf.norm(heart_pred, axis=-1) ** 2
    return K.mean(scalarProduct / (squaredNorm1 + squaredNorm2 - scalarProduct),
                  axis=-1)

def backgroundSoftIoU(groundTruth, prediction):
    bg_truth = groundTruth[:, :, :, 3]
    bg_pred = prediction[:, :, :, 3]
    scalarProduct = tf.einsum('ij,ij->i', bg_truth, bg_pred)
    squaredNorm1 = tf.norm(bg_truth, axis=-1) ** 2
    squaredNorm2 = tf.norm(bg_pred, axis=-1) ** 2
    return K.mean(scalarProduct / (squaredNorm1 + squaredNorm2 - scalarProduct),
                  axis=-1)

def softIoU(groundTruth, prediction):
    sumIoU = (leftLungSoftIoU(groundTruth, prediction)
    + rightLungSoftIoU(groundTruth, prediction)
    + heartSoftIoU(groundTruth, prediction)
    + backgroundSoftIoU(groundTruth, prediction))
    return (- sumIoU / 4)