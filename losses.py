import tensorflow as tf
import numpy as np

def DiceLoss(gt, preds, smooth=1e-6):
    intersection = tf.reduce_sum(gt * preds)
    dice_coeff = (2 * intersection + smooth) / (tf.reduce_sum(gt) + tf.reduce_sum(preds) + smooth)
    return 1 - dice_coeff


def weighted_xent(gt, preds, weights, smooth=1e-6):
    losses = tf.nn.softmax_cross_entropy_with_logits(gt, preds)
    return tf.reduce_sum(weights * losses)

if __name__ =='__main__':
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    print(DiceLoss(boxes1, boxes2))
