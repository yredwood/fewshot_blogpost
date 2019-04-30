import tensorflow as tf
import pdb

from lib.reslayers import Conv, Dense, relu
from lib.reslayers import BatchNorm, global_avg_pool

class SimpleNet(object):
    def __init__(self, name, gpu_idx=0, reuse=None):

        self.univ_params = {}
        self.domain_params = {}

        def _create_block(n_in, n_out, stride, name):
            # univ/dom params : blockwise params
            self.univ_params[name+'/conv1'] \
                    = Conv(n_in, n_out, 3, strides=1,
                            name=name+'/conv1', padding='SAME')

            self.univ_params[name+'/bn1'] \
                    = BatchNorm(n_out,
                            name=name+'/bn1',
                            gpu_idx=gpu_idx)

        with tf.variable_scope(name, reuse=reuse):
            # unfolded structure
            _create_block(3, 64, 2, 'c1')
            _create_block(64, 64, 2, 'c2')
            _create_block(64, 64, 2, 'c3')
            _create_block(64, 64, 2, 'c4')
        

    def __call__(self, x, train):

        def _apply_block(x, name):
            x = self.univ_params[name+'/conv1'](x)
            x = self.univ_params[name+'/bn1'](x, train)
            x = relu(x)
            x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 
                    'VALID', 'NCHW')
            return x

        x = _apply_block(x, 'c1')
        x = _apply_block(x, 'c2')
        x = _apply_block(x, 'c3')
        x = _apply_block(x, 'c4')
        x = tf.layers.flatten(x)

        return x
