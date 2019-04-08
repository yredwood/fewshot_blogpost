import tensorflow as tf
import numpy as np

from lib.reslayers import Conv, Dense, relu
from lib.reslayers import BatchNorm, global_avg_pool

class ResNet(object):
    def __init__(self, name, depth,
                    gpu_idx=0, reuse=None, config=None):

        self.depth = depth

#        self.pre_conv_params = []
#        self.pre_bn_params = []
#        self.proj_conv_params = []
        self.pre_params = []
        self.post_params = []

        if config=='cl_cifar10':
            self.stride_size = 2
        else:
            self.stride_size = 3

        if depth == 20:
            self.n_blocks = [3, 3, 3]
            self.n_filters = [16, 16, 32, 64]
        elif depth == 21:
            self.n_blocks = [3, 3, 3]
            self.n_filters = [128, 128, 256, 512]
        elif depth == 56:
            self.n_blocks = [9, 9, 9]
            self.n_filters = [16, 16, 32, 64]
        else:
            raise NotImplementedError("Invalid block size....")
        
        def _create_block(block_params, n_in, n_out, stride, name):
            block_params.append(
                    Conv(n_in, n_out, 3, strides=stride, 
                        name=name+'/conv1', padding='SAME'))
            block_params.append(
                    BatchNorm(n_out, name=name+'/bn1', 
                        gpu_idx=gpu_idx))

            block_params.append(
                    Conv(n_out, n_out, 3, strides=1,
                        name=name+'/conv2', padding='SAME'))
            block_params.append(
                    BatchNorm(n_out, name=name+'/bn2', 
                        gpu_idx=gpu_idx))

            if n_in != n_out: 
                block_params.append(
                        Conv(n_in, n_out, 3, strides=stride, 
                            name=name+'/proj_conv', padding='SAME'))
                block_params.append(
                        BatchNorm(n_out, name=name+'proj_bn', 
                            gpu_idx=gpu_idx))
            
        with tf.variable_scope(name, reuse=reuse):
            self.pre_params.append(
                    Conv(3, self.n_filters[0], 3, strides=1,
                        name='pre_conv', padding='SAME'))
            self.pre_params.append(
                    BatchNorm(self.n_filters[0], name='pre_bn', 
                        gpu_idx=gpu_idx))
            
            block_params = [[] for _ in range(np.sum(self.n_blocks))]
            block_cnt = 0
            for n_group, n_block in enumerate(self.n_blocks):
                name = 'Group{}'.format(n_group)
                n_in = self.n_filters[n_group]
                n_out = self.n_filters[n_group+1]
                stride = 1 if n_in == n_out else self.stride_size
                for bn in range(n_block):

                    _create_block(
                            block_params[block_cnt], 
                            n_in, n_out,
                            stride=stride, 
                            name=name+'/Block{}'.format(bn))

                    n_in = n_out
                    stride = 1
                    block_cnt += 1 

#            # unfolded structure of resnet20_v1
#            _create_block(block_params[0], 16, 16, stride=1, name='g1b1')
#            _create_block(block_params[1], 16, 16, stride=1, name='g1b2')
#            _create_block(block_params[2], 16, 16, stride=1, name='g1b3')
#                    
#            _create_block(block_params[3], 16, 32, stride=2, name='g2b1')
#            _create_block(block_params[4], 32, 32, stride=1, name='g2b2')
#            _create_block(block_params[5], 32, 32, stride=1, name='g2b3')
#
#            _create_block(block_params[6], 32, 64, stride=2, name='g3b1')
#            _create_block(block_params[7], 64, 64, stride=1, name='g3b2')
#            _create_block(block_params[8], 64, 64, stride=1, name='g3b3')

#            self.post_params.append(
#                    Dense(64, n_classes, name='dense_logit'))
            self.block_params = block_params


    def __call__(self, x, train):
        def _apply_block(x, stride, train, n_b):
            _rs = x # shortcut
            x = self.block_params[n_b][0](x)  #conv
            x = self.block_params[n_b][1](x, train) #bn
            x = relu(x)
            
            x = self.block_params[n_b][2](x)  #conv
            x = self.block_params[n_b][3](x, train) #bn

            if stride==self.stride_size:
                _rs = self.block_params[n_b][4](_rs, use_bias=False)
                _rs = self.block_params[n_b][5](_rs, train)
            x = relu(x + _rs)
            return x

        x = self.pre_params[0](x)
        x = self.pre_params[1](x, train)
        
        block_cnt = 0
        for n_group, n_block in enumerate(self.n_blocks):
            n_in = self.n_filters[n_group]
            n_out = self.n_filters[n_group+1]
            stride = 1 if n_in == n_out else self.stride_size
            for bn in range(n_block):
                x = _apply_block(x, stride, train, block_cnt)

                stride = 1
                block_cnt +=1 

#        # unfolded version
#        x = _apply_block(x, 1, train, 0)
#        x = _apply_block(x, 1, train, 1)
#        x = _apply_block(x, 1, train, 2)
#
#        x = _apply_block(x, 2, train, 3)
#        x = _apply_block(x, 1, train, 4)
#        x = _apply_block(x, 1, train, 5)
#                    
#        x = _apply_block(x, 2, train, 6)
#        x = _apply_block(x, 1, train, 7)
#        x = _apply_block(x, 1, train, 8)

        x = global_avg_pool(x)
        x = tf.layers.flatten(x)
#        logits = self.post_params[0](x)

        return x
