import tensorflow as tf 
from lib.layers import Network

class VGGNet_CIFAR(Network):
    def __init__(self, name):
        self.name = name

    def net(self, in_x, reuse=False, isTr=True):
        def conv_block(x, name, filt, reuse, isTr):
            x = self.conv(x, filt, name=name+'/conv', reuse=reuse)
            x = tf.nn.relu(x)
            x = self.batch_norm(x, isTr, name=name+'/bn', reuse=reuse)
            return x
        
        kp = [0.3,0.4,0.5] if isTr else [1.,1.,1.]

        x = conv_block(in_x, 'block01', 64, reuse, isTr)
        x = tf.nn.dropout(x, kp[0])
        x = conv_block(x, 'block02', 64, reuse, isTr)
        x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')

        x = conv_block(x, 'block03', 128, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block04', 128, reuse, isTr)
        x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')

        x = conv_block(x, 'block05', 256, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block06', 256, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block07', 256, reuse, isTr)
        x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')

        x = conv_block(x, 'block08', 512, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block09', 512, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block10', 512, reuse, isTr)
        x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')

        x = conv_block(x, 'block11', 512, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block12', 512, reuse, isTr)
        x = tf.nn.dropout(x, kp[1])
        x = conv_block(x, 'block13', 512, reuse, isTr)
        x = tf.nn.max_pool(x, [1,1,2,2], [1,1,2,2], 'VALID', 'NCHW')

        x = tf.nn.dropout(x, kp[2])
        x = tf.layers.flatten(x)
        x = self.dense(x, 512, name='fc14', reuse=reuse)

        x = tf.nn.dropout(x, kp[2])
        # last fully connected layer and softmax layer will be applied in 
        # each task specific networks
        return x
