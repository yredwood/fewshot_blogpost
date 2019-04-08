import tensorflow as tf
import numpy as np
import pdb

from lib.reslayers import Conv, Dense, relu
from lib.reslayers import BatchNorm, global_avg_pool

class ResNetAdapter(object):
    def __init__(self, name, depth,
                    gpu_idx=0, reuse=None, config=None,
                    n_adapt=2):

        self.depth = depth
#        self.pre_conv_params = []
#        self.pre_bn_params = []
#        self.proj_conv_params = []
        self.pre_params = []
        self.post_params = []
        self.univ_params = []
        self.domain_params = []
        self.n_adapt = n_adapt

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
        
        def _create_block(univ_params, dom_params, n_in, n_out, stride, name):
            univ_params.append(
                    Conv(n_in, n_out, 3, strides=stride, 
                        name=name+'/conv1', padding='SAME'))
            
            for na in range(self.n_adapt):

                dom_params[na].append(
                        BatchNorm(n_out, 
                            name=name+'/adapt{}_bn1.1'.format(na),
                            gpu_idx=gpu_idx))

                dom_params[na].append(
                        Conv(n_out, n_out, 1, strides=1,
                            name=name+'/adapt{}_conv1'.format(na), 
                            padding='SAME'))

                dom_params[na].append(
                        BatchNorm(n_out, 
                            name=name+'/adapt{}_bn1.2'.format(na),
                            gpu_idx=gpu_idx))

            univ_params.append(
                    Conv(n_out, n_out, 3, strides=1, 
                        name=name+'/conv2', padding='SAME'))

            for na in range(self.n_adapt):

                dom_params[na].append(
                        BatchNorm(n_out, 
                            name=name+'/adapt{}_bn2.1'.format(na),
                            gpu_idx=gpu_idx))

                dom_params[na].append(
                        Conv(n_out, n_out, 1, strides=1,
                            name=name+'/adapt{}_conv2'.format(na), 
                            padding='SAME'))

                dom_params[na].append(
                        BatchNorm(n_out, 
                            name=name+'/adapt{}_bn2.2'.format(na),
                            gpu_idx=gpu_idx))

            if n_in != n_out: 
                univ_params.append(
                        Conv(n_in, n_out, 3, strides=stride, 
                            name=name+'/proj_conv', padding='SAME'))
                univ_params.append(
                        BatchNorm(n_out, name=name+'proj_bn', 
                            gpu_idx=gpu_idx))
            
        with tf.variable_scope(name, reuse=reuse):
            self.pre_params.append(
                    Conv(3, self.n_filters[0], 3, strides=1,
                        name='pre_conv', padding='SAME'))
            self.pre_params.append(
                    BatchNorm(self.n_filters[0], name='pre_bn', 
                        gpu_idx=gpu_idx))
            

            univ_params = [[] for _ in range(np.sum(self.n_blocks))]
            dom_params = []
            block_cnt = 0
            for n_group, n_block in enumerate(self.n_blocks):
                name = 'Group{}'.format(n_group)
                n_in = self.n_filters[n_group]
                n_out = self.n_filters[n_group+1]
                stride = 1 if n_in == n_out else self.stride_size
                for bn in range(n_block):
                    in_dom_params = [[] for _ in range(self.n_adapt)]
                    _create_block(
                            univ_params[block_cnt], # []
                            in_dom_params, # [[],[],[],..]
                            n_in, n_out, 
                            stride=stride, 
                            name=name+'/Block{}'.format(bn))
                    dom_params.append(in_dom_params)

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
                

            self.univ_vars = []
            self.dom_vars = [[] for _ in range(self.n_adapt)]
            vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for i,v in enumerate(vs):
                vtype = v.name.split('/')[-2]
                if 'adapt' in vtype:
                    na = int(vtype.split('_')[0][-1])
                    self.dom_vars[na].append(v)
                else:
                    self.univ_vars.append(v)
    
            self.univ_params = univ_params
            self.dom_params = dom_params


    def __call__(self, x, train, s_adapt):

        def _apply_block(x, stride, train, n_b):
            _rs = x # shortcut

            adp_x = self.univ_params[n_b][0](x)  #conv
            
            x = self.dom_params[n_b][s_adapt][0](adp_x, train)
            x = self.dom_params[n_b][s_adapt][1](x)
            x = adp_x + x
            x = self.dom_params[n_b][s_adapt][2](x, train)
            x = relu(x)

            adp_x = self.univ_params[n_b][1](x)  #conv

            x = self.dom_params[n_b][s_adapt][3](adp_x, train)
            x = self.dom_params[n_b][s_adapt][4](x)
            x = adp_x + x
            x = self.dom_params[n_b][s_adapt][5](x, train)

            if stride==self.stride_size:
                _rs = self.univ_params[n_b][2](_rs, use_bias=False)
                _rs = self.univ_params[n_b][3](_rs, train)
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


        x = global_avg_pool(x)
        x = tf.layers.flatten(x)
#        logits = self.post_params[0](x)
        return x
