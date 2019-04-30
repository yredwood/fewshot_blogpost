import tensorflow as tf
import pdb

from lib.simplenet import SimpleNet 
from lib.vggnet import VGGNet
from lib.layers import Network, cross_entropy, tf_acc

class ConvNet(Network):
    def __init__(
            self, name, n_classes, isTr,
            reuse=False, input_dict=None,
            config='cl_cifar10', 
            metalearn=False,
            arch=None,
            mbsize=1, nway=5, kshot=1, qsize=15):

        self.name = name
        self.n_classes = n_classes
        self.hw = hw = config['hw']
        self.metalearn = metalearn
        self.mbsize = mbsize
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.arch = arch
        if input_dict is None:
            if metalearn:
                self.inputs = {
                    'sx': tf.placeholder(tf.float32, [mbsize,nway*kshot,hw,hw,3]),
                    'qx': tf.placeholder(tf.float32, [mbsize,nway*qsize,hw,hw,3]),
                    'qy': tf.placeholder(tf.float32, [mbsize,nway*qsize,nway]),
                }
            else:
                self.inputs = {
                    'x': tf.placeholder(tf.float32, [None,self.hw,self.hw,3]),
                    'y': tf.placeholder(tf.float32, [None,n_classes])
                }
        else:
            self.inputs = input_dict
        self.outputs = {}


        with tf.variable_scope(name, reuse=reuse):
            self._init_net(reuse)
            if metalearn:
                self._build_proto(isTr, reuse)
            else:
                self._build_network(isTr, reuse)
        
    def _init_net(self, reuse):
        stride = 2
        if self.arch=='simple':
            self.net = SimpleNet('simple', reuse=reuse)
        elif self.arch=='vgg':
            self.net = VGGNet('vgg', reuse=reuse, stride=stride)
        else:
            print ('No such architecture')
            exit()

    def _build_network(self, isTr, reuse):
        x = tf.transpose(self.inputs['x'], [0,3,1,2])
        f = self.net(x, isTr)
    
        pred = self.dense(f, self.n_classes, name='trtask_dense', reuse=reuse)
        loss = cross_entropy(tf.nn.softmax(pred), self.inputs['y'])
        acc = tf_acc(pred, self.inputs['y'])

        self.outputs['embedding'] = f
        self.outputs['pred'] = pred
        self.outputs['loss'] = loss
        self.outputs['acc'] = acc
