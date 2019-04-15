import tensorflow as tf
import numpy as np
import pdb

from lib.reslayers import Conv, Dense, relu
from lib.reslayers import BatchNorm, global_avg_pool

class ConvNetAdapter(object):
    def __init__(self, name, gpu_idx=0, reuse=None, n_adapt=2):

        self.univ_params = {}
        self.domain_params = {}
        self.n_adapt = n_adapt

        def _create_block(n_in, n_out, stride, name):
            # univ/dom params : blockwise params
            self.univ_params[name+'/conv1'] \
                    = Conv(n_in, n_out, 3, strides=1,
                            name=name+'/conv1', padding='SAME')

            for na in range(self.n_adapt):
                self.domain_params[name+'/adapt{}_bn1.1'.format(na)] \
                        = BatchNorm(n_out,
                                name=name+'/adapt{}_bn1.1'.format(na),
                                gpu_idx=gpu_idx)
                
                self.domain_params[name+'/adapt{}_conv1'.format(na)] \
                        = Conv(n_out, n_out, 1, strides=1,
                                name=name+'/adapt{}_conv1'.format(na), 
                                padding='SAME')
                
                self.domain_params[name+'/adapt{}_bn1.2'.format(na)] \
                        = BatchNorm(n_out,
                                name=name+'/adapt{}_bn1.2'.format(na),
                                gpu_idx=gpu_idx)
            

        with tf.variable_scope(name, reuse=reuse):
            # unfolded structure
            _create_block(3, 64, 2, 'c1')
            _create_block(64, 64, 2, 'c2')
            _create_block(64, 64, 2, 'c3')
            _create_block(64, 64, 2, 'c4')
        
        
        # this is really weired 
        self.univ_vars = []
        self.dom_vars = [[] for _ in range(self.n_adapt)]
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for i,v in enumerate(vs):
            vt = v.name.split('/')
            if len(vt) < 2:
                continue
            if 'adapt' in vt[-2]:
                na = int(vt[-2].replace('adapt','')[0])
                self.dom_vars[na].append(v)
            else:
                self.univ_vars.append(v)
        
        def cnt_params(vlist):
            npms = 0
            for v in vlist:
                pms = 1
                for s in v.shape:
                    pms*=s
                npms+=pms
            return npms
        print ('# of domain specific params: {}'\
                .format(cnt_params(self.dom_vars[0])))
        print ('# of universial params: {}'\
                .format(cnt_params(self.univ_vars)))


    def __call__(self, x, train, s_adapt):

        def _apply_block(x, name):
            x = self.univ_params[name+'/conv1'](x)
            x = self.domain_params[name+'/adapt{}_bn1.1'\
                    .format(s_adapt)](x, train)
            x = self.domain_params[name+'/adapt{}_conv1'\
                    .format(s_adapt)](x)
            x = self.domain_params[name+'/adapt{}_bn1.2'\
                    .format(s_adapt)](x, train)
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
