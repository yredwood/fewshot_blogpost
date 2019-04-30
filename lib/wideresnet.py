import tensorflow as tf
import pdb

from lib.reslayers import Conv, Dense, relu
from lib.reslayers import BatchNorm, global_avg_pool

class WideResNet(object):
    def __init__(self, depth, filters, k, name='wdnet',
            gpu_idx=0, reuse=None, stride=2):

        self.univ_params = {}
        self.domain_params = {}
        self.depth = depth
        self.k = k
        self.s = stride
        self.f = [filters, filters*k, filters*2*k, filters*4*k]
        assert (depth-4)%6==0
        self.n_blocks = [(depth-4)//6] * 3

        def _create_block(n_in, n_out, stride, name):
            # univ/dom params : blockwise params
            self.univ_params[name+'/bn1'] \
                    = BatchNorm(n_in, name=name+'/bn1',
                            gpu_idx=gpu_idx)
            self.univ_params[name+'/conv1'] \
                    = Conv(n_in, n_out, 3, strides=stride,
                            name=name+'/conv1', padding='SAME')

            self.univ_params[name+'/bn2'] \
                    = BatchNorm(n_out, name=name+'/bn2',
                            gpu_idx=gpu_idx)
            self.univ_params[name+'/conv2'] \
                    = Conv(n_out, n_out, 3, strides=1,
                            name=name+'/conv2', padding='SAME')

            if n_in != n_out:
                self.univ_params[name+'/proj_conv'] \
                        = Conv(n_in, n_out, 1, strides=stride,
                                name=name+'/proj_conv', padding='SAME')


        with tf.variable_scope(name, reuse=reuse):
            self.univ_params['prebn1'] \
                    = BatchNorm(3,
                            name='prebn1',
                            gpu_idx=gpu_idx)
            self.univ_params['preconv'] \
                    = Conv(3, self.f[0],
                            3, strides=1,
                            name='preconv', padding='SAME')
            self.univ_params['prebn2'] \
                    = BatchNorm(self.f[0],
                            name='prebn2',
                            gpu_idx=gpu_idx)
        
            # unfolded structure of 16-10
            _create_block(self.f[0], self.f[1], 1, 'G1B1')
            _create_block(self.f[1], self.f[1], 1, 'G1B2')

            _create_block(self.f[1], self.f[2], self.s, 'G2B1')
            _create_block(self.f[2], self.f[2], 1, 'G2B2')

            _create_block(self.f[2], self.f[3], self.s, 'G3B1')
            _create_block(self.f[3], self.f[3], 1, 'G3B2')

            self.univ_params['postbn'] \
                    = BatchNorm(self.f[3], 
                            name='postbn',
                            gpu_idx=gpu_idx)
        


    def __call__(self, x, train):

        def _apply_block(x, name, downsample):
            residual = x
            x = self.univ_params[name+'/bn1'](x, train)
            x = relu(x)
            if downsample:
                residual = self.univ_params[name+'/proj_conv'](x)
            x = self.univ_params[name+'/conv1'](x)
    
            x = self.univ_params[name+'/bn2'](x, train)
            x = relu(x)
            x = self.univ_params[name+'/conv2'](x)
            x = x + residual
            return x

        f = self.f
        k = self.k

        x = self.univ_params['prebn1'](x, train) 
        x = self.univ_params['preconv'](x)
        x = self.univ_params['prebn2'](x, train)

        x = _apply_block(x, 'G1B1', 1)
        x = _apply_block(x, 'G1B2', 0)

        x = _apply_block(x, 'G2B1', 1)
        x = _apply_block(x, 'G2B2', 0)

        x = _apply_block(x, 'G3B1', 1)
        x = _apply_block(x, 'G3B2', 0)

        x = self.univ_params['postbn'](x, train)
        x = relu(x)
        x = global_avg_pool(x)
        return x
