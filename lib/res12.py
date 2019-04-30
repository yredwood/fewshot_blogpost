import tensorflow as tf
import numpy as np
import pdb

from lib.reslayers import Conv, Dense, relu, dropout, SS
from lib.reslayers import BatchNorm, global_avg_pool

lrelu = tf.nn.leaky_relu
maxpool = tf.layers.max_pooling2d

class ResNet12(object):
    def __init__(self, name, gpu_idx=0, reuse=None, config=None):

        self.univ_params = {}
        self.domain_params = {}

        def _create_block(n_in, n_out, name):
            self.univ_params[name+'/c0'] \
                    = Conv(n_in, n_out, 3, strides=1,
                            name=name+'/c0', padding='SAME')
            self.univ_params[name+'/b0'] \
                    = BatchNorm(n_out, name=name+'/b0',
                            gpu_idx=gpu_idx)

            self.domain_params[name+'/d0'] \
                    = SS([1,1,n_in,n_out], name=name+'/d0')

            for i in range(1, 3):
                self.univ_params[name+'/c{}'.format(i)] \
                        = Conv(n_out, n_out, 3, strides=1,
                                name=name+'/c{}'.format(i), 
                                padding='SAME')
                self.univ_params[name+'/b{}'.format(i)] \
                        = BatchNorm(n_out, name=name+'/b{}'.format(i),
                                gpu_idx=gpu_idx)

                self.domain_params[name+'/d{}'.format(i)] \
                        = SS([1,1,n_out,n_out], name=name+'/d{}'.format(i))
#
            self.univ_params[name+'/c_rs'] \
                        = Conv(n_in, n_out, 1, strides=1,
                                name=name+'/c_rs', 
                                padding='SAME')

        with tf.variable_scope(name, reuse=reuse):

#            # unfolded structure of resnet12
            _create_block(3,64,'b1')
            _create_block(64,128, 'b2')
            _create_block(128,256, 'b3')
            _create_block(256,512, 'b4')


    def __call__(self, x, train):
        def _apply_block(x, name, train):
            _rs = x # shortcut
            _rs = self.univ_params[name+'/c_rs'](_rs)
            for i in range(3):
                x = self.univ_params[name+'/c{}'.format(i)](x, 
                        SS=self.domain_params[name+'/d{}'.format(i)])
                x = self.univ_params[name+'/b{}'.format(i)](x, train)
                x = lrelu(x, alpha=0.1)
            x = _rs + x
            x = maxpool(x,2,2,data_format='channels_first')
            x = dropout(x, 0.1, training=train)

            return x

        # unfolded version
        x = _apply_block(x, 'b1', train)
        x = _apply_block(x, 'b2', train)
        x = _apply_block(x, 'b3', train)
        x = _apply_block(x, 'b4', train)
        x = global_avg_pool(x)
        #x = tf.layers.flatten(x)

        return x
