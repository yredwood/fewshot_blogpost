import tensorflow as tf
import pdb

from lib.reslayers import Conv, Dense, BatchNorm, dropout, relu
from lib.reslayers import pool, flatten, global_avg_pool

class VGGNet(object):
    def __init__(
            self, 
            name='vgg', 
            in_channels=3, gpu_idx=0, 
            reuse=None, stride=2):
            
        if stride==2:
            self.pool_stride = 2
            self.is_avg_pool = False
        else:
            self.pool_stride = 3
            self.is_avg_pool = False

        def pool(x):
            out = tf.layers.max_pooling2d(x,
                    self.pool_stride, 
                    self.pool_stride,
                    data_format='channels_first')

        self.name = name
        self.n_units = [
                64, 64, 128, 128, 256, 256, 256,
                512, 512, 512, 512, 512, 512#, 512*4, 512*4
        ]
        self.conv_params = [] 
        self.bn_params = []
        self.dense_params = []
        def _create_block(l, n_in, n_out):
            self.conv_params.append(
                    Conv(n_in, n_out, 3, 
                        name='conv'+str(l), 
                        padding='SAME'))
            self.bn_params.append(
                    BatchNorm(n_out, 
                        name='bn'+str(l), 
                        gpu_idx=gpu_idx))

        with tf.variable_scope(name, reuse=reuse):
            n_units = self.n_units
            _create_block(1, in_channels, n_units[0])
            for i in range(1, 13):
                _create_block(i+1, n_units[i-1], n_units[i])

#            self.dense_params.append(
#                    Dense(n_units[13], n_units[14], 
#                        name='dense14'))
#            self.bn_params.append(
#                    BatchNorm(n_units[14], name='bn14', 
#                        gpu_idx=gpu_idx))

    def __call__(self, x, train):
        def _apply_block(x, train, l, p=None):
            conv = self.conv_params[l]
            bn = self.bn_params[l]
            x = relu(bn(conv(x), train))
            x = pool(x) if p is None \
                    else dropout(x, p, training=train)
            return x

        p_list = [
            0.3, None,
            0.4, None,
            0.4, 0.4, None,
            0.4, 0.4, None,
            0.4, 0.4, None
        ]
        for l, p in enumerate(p_list):
            x = _apply_block(x, train, l, p=p)

#        bn14 = self.bn_params[-1]
#        dense14 = self.dense_params[0]

        x = flatten(global_avg_pool(x)) \
                if self.is_avg_pool else flatten(x)
        #x = dropout(x, 0.5, training=train)
        #x = relu(bn14(dense14(x), train))
        #x = dropout(x, 0.5, training=train)
        return x
