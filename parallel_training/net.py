import tensorflow as tf
import pdb

#from lib.resadapters import ResNetAdapter
from lib.simplenet import SimpleNet 
from lib.wideresnet import WideResNet
from lib.res12 import ResNet12
from lib.layers import Network, cross_entropy, tf_acc

class ConvNet(Network):
    def __init__(
            self, name, isTr,
            reuse=False, input_dict=None,
            config='cl_cifar10', 
            metalearn=False,
            arch=None,
            mbsize=1, nway=5, kshot=1, qsize=15,
            var_list=None):

        self.name = name
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
                    'lr': tf.placeholder(tf.float32)
                }
            else:
                self.inputs = {
                    'x': tf.placeholder(tf.float32, [None,self.hw,self.hw,3]),
                    'y': tf.placeholder(tf.float32, [None,nway]),
                    'lr': tf.placeholder(tf.float32)
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
            
            if isTr:
                self._init_optimizer(var_strlist=var_list)
        
    def _init_net(self, reuse):
        if self.hw == 84:
            stride = 3
        else:
            stride = 2
        if self.arch=='wdres':
            self.net = WideResNet(depth=16, filters=16, k=10,
                    name='wdnet', reuse=reuse, stride=stride)
        elif self.arch=='simple':
            self.net = SimpleNet('simple', reuse=reuse)
        elif self.arch=='res12':
            self.net = ResNet12('res12', reuse=reuse)
        else:
            print ('No such architecture')
            exit()

#        self.alpha = tf.get_variable('h_alpha',  
#                initializer=[1.])
            
    def _build_network(self, isTr, reuse):
        x = tf.transpose(self.inputs['x'], [0,3,1,2])
        f = self.net(x, isTr)
    
        pred = self.dense(f, self.nway, name='11trtask_dense', reuse=reuse)
        loss = cross_entropy(tf.nn.softmax(pred), self.inputs['y'])
        acc = tf_acc(pred, self.inputs['y'])

        self.outputs['embedding'] = f
        self.outputs['pred'] = pred
        self.outputs['loss'] = loss
        self.outputs['acc'] = acc

    def _build_proto(self, isTr, reuse):

        def cnn(in_x, isTr, reuse):
            x = tf.transpose(in_x, [0,3,1,2])
            f = self.net(x, isTr)  
#            f = self.dense(f, 64, name='trtask_dense', reuse=reuse) 
##            f = self.dense(f, 512, name='new_dense', reuse=reuse)
            return f

        ip = self.inputs
        mball = tf.concat([
            tf.reshape(ip['sx'], [-1,self.hw,self.hw,3]),
            tf.reshape(ip['qx'], [-1,self.hw,self.hw,3])],
            axis=0)

        all_feats = cnn(mball, isTr=isTr, reuse=reuse) * self.alpha 
        # .shape = (MN(K+Q),H)
        self.hdim = all_feats.get_shape().as_list()[-1]

        xs = tf.reshape(
                all_feats[:self.mbsize*self.nway*self.kshot],
                [self.mbsize,self.nway*self.kshot,self.hdim])
        xq = tf.reshape(
                all_feats[self.mbsize*self.nway*self.kshot:],
                [self.mbsize,self.nway*self.qsize,self.hdim])

        def singlebatch_graph(inputs):
            xs, xq, yq = inputs
            
            proto_vec = tf.reshape(xs, [self.nway,-1,self.hdim])
            proto_vec = tf.reduce_mean(proto_vec, axis=1)
            
            sim = self.uclidean_sim(proto_vec, xq)
            pred = tf.nn.softmax(sim)
            loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=yq, logits=sim)
            acc = tf_acc(pred, yq)
            return sim, pred, loss, acc

        elems = (xs, xq, ip['qy'])
        out_dtype = (tf.float32, tf.float32, tf.float32, tf.float32)
        meta_sim, meta_pred, meta_loss, meta_acc = \
                tf.map_fn(singlebatch_graph, elems, dtype=out_dtype,
                        parallel_iterations=self.mbsize)

        self.outputs['embedding'] = 0
        self.outputs['pred'] = meta_sim
        
        self.outputs['loss'] = tf.reduce_mean(meta_loss)
        self.outputs['acc'] = tf.reduce_mean(meta_acc)

    def uclidean_sim(self, xs, xq):
        xs = tf.expand_dims(xs, 0)
        xq = tf.expand_dims(xq, 1)
        emb = (xs-xq)**2
        dist = tf.reduce_mean(emb, axis=2)
        return -dist

    def _init_optimizer(self, var_strlist=None):
        trvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if var_strlist is not None:
            trvars = []
            for vstr in var_strlist:
                trvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                        scope=vstr)
        print ('----training var. list: ----')
        for i,v in enumerate(trvars):
            print (i,v.name)

        opt = tf.train.MomentumOptimizer(self.inputs['lr'], 0.9)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            # TODO get 
            self.train_op = opt.minimize(self.outputs['loss'], var_list=trvars)
                






#
