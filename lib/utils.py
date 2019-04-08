import numpy as np
import tensorflow as tf


def lr_scheduler(current_epoch, init_lr, epoch_list, decay):
    lr_list = [init_lr*decay**(i) for i in range(len(epoch_list)+1)] 
    lridx = np.sum(np.array(epoch_list) <= current_epoch)
    return lr_list[lridx]

def l2_loss(var_list=None):
    var_list = tf.trainable_variables() \
            if var_list is None else var_list
    return tf.add_n([tf.nn.l2_loss(v) for v in var_list])
