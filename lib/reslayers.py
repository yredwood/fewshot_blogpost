import tensorflow as tf
import numpy as np
import pdb

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
tanh = tf.nn.tanh
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
flatten = tf.layers.flatten
dropout = tf.layers.dropout

class Dense(object):
    def __init__(self, n_in, n_out, name='dense', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/cpu:0'):
                self.W = tf.get_variable('W', shape=[n_in, n_out])
                self.b = tf.get_variable('b', shape=[n_out]) 

    def __call__(self, x, activation=None):
        x = tf.matmul(x, self.W) + self.b
        x = x if activation is None else activation(x)
        return x

    def params(self, trainable=None):
        return [self.W, self.b]

class _GroupConv(object): 
    # this is too much slow
    # https://stackoverflow.com/questions/
    # 48994369/grouped-depthwise-convolution-performance
    def __init__(self, n_in, n_out, kernel_size, strides=1, group_size=1,
            padding='VALID', name='conv', reuse=None, use_bias=False):
        with tf.variable_scope(name, reuse=reuse):
            M = int(n_out * group_size / n_in)
            self.W = tf.get_variable('W',
                    shape=[kernel_size,kernel_size,n_in,M])
        self.strides=strides
        self.padding=padding
        self.group_size=int(group_size)
        self.M=M

    def __call__(self, x):
        y = tf.nn.depthwise_conv2d(x, self.W,
                strides=[1,1,self.strides,self.strides],
                padding=self.padding, data_format='NCHW')
#        print (nchw)
#        print ([nchw[0],int(nchw[1]/self.group_size/self.M),
#            self.group_size,self.M,nchw[2],nchw[3]])
        nchw = y.shape.as_list()
        y = tf.reshape(y, [-1,int(nchw[1]/self.group_size/self.M),
            self.group_size,self.M,nchw[2],nchw[3]])
        y = tf.reduce_sum(y, axis=2)
        y = tf.reshape(y, [-1,int(nchw[1]/self.group_size),
            nchw[2],nchw[3]])
        return y

class GroupConv(object):
    def __init__(self, n_in, n_out, kernel_size, strides=1, group_size=1,
            padding='VALID', name='conv', reuse=None, use_bias=False):
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/cpu:0'):
                self.W = tf.get_variable('W',
                        shape=[kernel_size,kernel_size, n_in / group_size, n_out])
                if use_bias:
                    self.b = tf.get_variable('b', shape=[n_out])
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding
        self.group_size = group_size

    def __call__(self, x, activation=None, use_bias=False):
        group_conv = lambda x, w: tf.nn.conv2d(x, w, 
                strides=[1,1,self.strides,self.strides],
                padding=self.padding,
                data_format='NCHW')
        w_group = tf.split(self.W, self.group_size, axis=3)
        x_group = tf.split(x, self.group_size, axis=1) # NCHW assumed
        y_group = [group_conv(_x, _w) for _x, _w in zip(x_group, w_group)]
        # bias and activation will be added later
        y = tf.concat(y_group, axis=1)
        return y

class SS(object):
    def __init__(self, shape, name, reuse=None, use_bias=False):
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/cpu:0'):
                self.W = tf.get_variable('W',
                        shape=shape, initializer=tf.ones_initializer())
                if use_bias:
                    self.b = tf.get_variable('b', shape=shape[-1], 
                            initializer=tf.zeros_initializer())
        self.use_bias = use_bias
        self.name = name

    def __call__(self):
        if self.use_bias:
            return (self.W, self.b)
        else:
            return (self.W, 0)


class Conv(object):
    def __init__(self, n_in, n_out, kernel_size, strides=1, padding='VALID',
                    name='conv', reuse=None, use_bias=False):
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/cpu:0'):
                self.W = tf.get_variable('W',
                        shape=[kernel_size, kernel_size, n_in, n_out])
                if use_bias:
                    self.b = tf.get_variable('b', shape=[n_out])
        self.use_bias = use_bias
        self.strides = strides
        self.padding = padding

    def __call__(self, x, activation=None, use_bias=False, SS=None):
        if SS is not None:
            w,b = SS()
            WW = w * self.W
        else:
            WW = self.W

        x = tf.nn.conv2d(x, WW,
                strides=[1, 1, self.strides, self.strides],
                padding=self.padding,
                data_format='NCHW')
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b, data_format='NCHW')
        x = x if activation is None else activation(x)
        return x

    def params(self, trainable=None):
        if self.bias == True:
            return [self.W, self.b]
        else:
            return [self.W]

class BatchNorm(object):
    def __init__(self, n_in, momentum=0.9,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            name='batch_norm', gpu_idx=0, reuse=None):
        self.momentum = momentum
        self.name = name
        self.gpu_idx = gpu_idx
        self.n_in = n_in

        with tf.variable_scope(name, reuse=reuse):
            self.moving_mean = tf.get_variable('moving_mean', [n_in],
                    initializer=tf.zeros_initializer(), trainable=False)
            self.moving_var = tf.get_variable('moving_var', [n_in],
                    initializer=tf.ones_initializer(), trainable=False)
            self.beta = tf.get_variable('beta', [n_in],
                    initializer=beta_initializer)
            self.gamma = tf.get_variable('gamma', [n_in],
                    initializer=gamma_initializer)

    def __call__(self, x, train):
        if train:
            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, self.gamma,
                    self.beta, data_format='NCHW')
            update_mean = self.moving_mean.assign_sub(
                    (1-self.momentum)*(self.moving_mean-batch_mean))
            update_var = self.moving_var.assign_sub(
                    (1-self.momentum)*(self.moving_var-batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
        else: 
            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, self.gamma, self.beta,
                    mean=self.moving_mean, variance=self.moving_var, is_training=False,
                    data_format='NCHW')
        return x

class _BatchNorm(object): # doesn't work well in tf.map_fn => don't know why # chek len() part
    def __init__(self, n_in, momentum=0.99,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            name='batch_norm', gpu_idx=0, reuse=None):
        self.momentum = momentum
        self.name = name
        self.gpu_idx = gpu_idx
        self.n_in = n_in
        with tf.variable_scope(name, reuse=reuse):
            with tf.device('/cpu:0'):
                self.moving_mean = tf.get_variable('moving_mean', [n_in],
                        initializer=tf.zeros_initializer(), trainable=False)
                self.moving_var = tf.get_variable('moving_var', [n_in],
                        initializer=tf.ones_initializer(), trainable=False)
                self.beta = tf.get_variable('beta', [n_in],
                        initializer=beta_initializer)
                self.gamma = tf.get_variable('gamma', [n_in],
                        initializer=gamma_initializer)

    def __call__(self, x, train, cs=True):
        if train:
            if len(x.shape) == 4:
                if cs:
                    x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                            self.gamma, self.beta, data_format='NCHW')
                else:
                    # wideresnet needs to use this
                    x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                            tf.ones([self.n_in]), tf.zeros([self.n_in]), 
                            data_format='NCHW')
            else:
                batch_mean, batch_var = tf.nn.moments(x, [0])
                x = tf.nn.batch_normalization(x, batch_mean, batch_var,
                        self.beta, self.gamma, 1e-3)
            if self.gpu_idx == 0:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                        self.moving_mean.assign_sub(
                            (1-self.momentum)*(self.moving_mean - batch_mean)))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                        self.moving_var.assign_sub(
                            (1-self.momentum)*(self.moving_var - batch_var)))
        else:
            if len(x.shape) == 4:
                if cs:
                    x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                            self.gamma, self.beta,
                            mean=self.moving_mean, variance=self.moving_var,
                            is_training=False, data_format='NCHW')
                else:
                    x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                            tf.ones([self.n_in]), tf.zeros([self.n_in]),
                            mean=self.moving_mean, variance=self.moving_var,
                            is_training=False, data_format='NCHW')
            else:
                x = tf.nn.batch_normalization(x, self.moving_mean, self.moving_var,
                        self.beta, self.gamma, 1e-3)
        return x

    def params(self, trainable=None):
        params = [self.beta, self.gamma]
        params = params + [self.moving_mean, self.moving_var] \
                    if trainable is None else params
        return params

def avg_pool(x, **kwargs):
    return tf.layers.average_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2, 
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

def softmax_cross_entropy(logits, labels):
    return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def binary_cross_entropy(logits, labels):
    preds = tf.nn.sigmoid(logits)
    ce = labels * log(preds) + (1-labels) * log(1-preds)
    ce = -tf.reduce_mean(ce, axis=0)
    return ce

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.to_float(correct))

def get_staircase_lr(global_step, bdrs, vals):
    lr = tf.train.piecewise_constant(tf.to_int32(global_step), bdrs, vals)
    return lr

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, axis=0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
