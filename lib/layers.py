import tensorflow as tf

relu = tf.nn.relu
elu = tf.nn.elu
normal = tf.distributions.Normal
kldv = tf.distributions.kl_divergence

class Network(object):
    def __init__(self, name):
        self.name = name
        self.eps = 1e-3

    def dense(self, x, units, name='dense', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            kernel = tf.get_variable('kernel', [x.shape[1].value, units])
            bias = tf.get_variable('bias', [units],
                    initializer=tf.zeros_initializer())
            x = tf.matmul(x, kernel) + bias
            return x

    def conv(self, x, filters, kernel_size=3, strides=1, padding='SAME',
            name='conv', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            kernel = tf.get_variable('kernel',
                    [kernel_size, kernel_size, x.shape[1].value, filters])
            x = tf.nn.conv2d(x, kernel, [1, 1, strides, strides],
                    padding=padding, data_format='NCHW')
            return x

    def deconv(self, x, filters, kernel_size=3, strides=1, padding='SAME', 
            name='deconv', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, data_format='channels_first',
                    reuse=reuse, padding=padding)
            return x

    def batch_norm(self, x, training, decay=0.9, name='batch_norm', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            dim = x.shape[1].value
            moving_mean = tf.get_variable('moving_mean', [dim],
                    initializer=tf.zeros_initializer(), trainable=False)
            moving_var = tf.get_variable('moving_var', [dim],
                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', [dim],
                    initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', [dim],
                    initializer=tf.ones_initializer())

            if training:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta, data_format='NCHW')
                update_mean = moving_mean.assign_sub((1-decay)*(moving_mean - batch_mean))
                update_var = moving_var.assign_sub((1-decay)*(moving_var - batch_var))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
            else:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                        mean=moving_mean, variance=moving_var, is_training=False,
                        data_format='NCHW')
            return x

    def global_avg_pool(self, x):
        return tf.reduce_mean(x, [2, 3])

    def simple_conv(self, in_x, reuse=False, isTr=True):
        def conv_block(x, name, reuse, isTr):
            x = self.conv(x, 64, name=name+'/conv', reuse=reuse)
            x = self.batch_norm(x, isTr, name=name+'/bn', reuse=reuse)
            x = relu(x)
            x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')
            return x
        x = in_x
        for i in range(4):
            x = conv_block(x, 'b{}'.format(i+1), reuse=reuse, isTr=isTr)
        x = tf.layers.flatten(x)
        return x

def ce_logit(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)

def cross_entropy(pred, label): 
    return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=1))

def cross_entropy_with_metabatch(pred, label):
    # shape of pred, label: (metabatch, batch, nway)
    return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=2), axis=1)

def tf_acc(p, y): 
    acc = tf.equal(tf.argmax(y,1), tf.argmax(p,1))
    acc = tf.reduce_mean(tf.cast(acc, 'float'))
    return acc

def ckpt_restore_with_prefix(sess, ckpt_dir, prefix):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
    var_list_name = [i.name.split(':')[0] for i in var_list]

    for var_name, _ in tf.contrib.framework.list_variables(ckpt_dir):
        var = tf.contrib.framework.load_variable(ckpt_dir, var_name)
        new_name = prefix + '/' + var_name
        if new_name in var_list_name:
            with tf.variable_scope(prefix, reuse=True):
                tfvar = tf.get_variable(var_name)
                sess.run(tfvar.assign(var))
