import tensorflow as tf

from lib.resadapters import ResNetAdapter
from lib.layers import Network, cross_entropy, tf_acc

class AdapterNet(Network):
    def __init__(
            self, name, n_classes, isTr,
            reuse=False, input_dict=None,
            config='cl_cifar10', 
            depth=20,
            n_adapt=2, 
            use_adapt=0,
            fixed_univ=False,
            metalearn=False,
            mbsize=1, nway=5, kshot=1, qsize=15):

        self.name = name
        self.n_classes = n_classes
        self.hw = 32 if config=='cl_cifar10' else 84
        self.metalearn = metalearn
        self.mbsize = mbsize
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        if input_dict is None:
            if metalearn:
                self.inputs = {
                    'sx': tf.placeholder(tf.float32, [mbsize,nway*kshot,84,84,3]),
                    'qx': tf.placeholder(tf.float32, [mbsize,nway*qsize,84,84,3]),
                    'qy': tf.placeholder(tf.float32, [mbsize,nway*qsize,nway]),
                    'lr': tf.placeholder(tf.float32),
                }
            else:
                self.inputs = {
                    'x': tf.placeholder(tf.float32, [None,self.hw,self.hw,3]),
                    'y': tf.placeholder(tf.float32, [None,n_classes])
                }
        else:
            self.inputs = input_dict
        self.outputs = {}

        self.config = config
        self.fixed_univ = fixed_univ
        self.depth = depth
        self.n_adapt = 2
        self.use_adapt = use_adapt 
        assert n_adapt > use_adapt

        with tf.variable_scope(name):
            if metalearn:
                self._build_proto(isTr, reuse)
            else:
                self._build_network(isTr, reuse)

    def _build_network(self, isTr, reuse):
        x = tf.transpose(self.inputs['x'], [0,3,1,2])
        resnet = ResNetAdapter('ra', self.depth, 
                reuse=reuse, config=self.config, 
                n_adapt=self.n_adapt)
        f = resnet(x, isTr, self.use_adapt)
        f = tf.layers.flatten(f)

        self.var_list = resnet.dom_vars[self.use_adapt] \
                if self.fixed_univ else None

        pred = self.dense(f, self.n_classes, name='trtask_dense', reuse=reuse)
        loss = cross_entropy(tf.nn.softmax(pred), self.inputs['y'])
        acc = tf_acc(pred, self.inputs['y'])

        self.outputs['embedding'] = f
        self.outputs['pred'] = pred
        self.outputs['loss'] = loss
        self.outputs['acc'] = acc

    def _build_proto(self, isTr, reuse):
        resnet = ResNetAdapter('ra', self.depth,
                reuse=reuse, config=self.config,
                n_adapt=self.n_adapt)
        self.var_list = resnet.dom_vars[self.use_adapt] \
                if self.fixed_univ else None

        def cnn(in_x, isTr, reuse):
            x = tf.transpose(in_x, [0,3,1,2])
            f = resnet(x, isTr, self.use_adapt)
            f = tf.layers.flatten(f)
            return f

        ip = self.inputs
        mball = tf.concat([
            tf.reshape(ip['sx'], [-1,84,84,3]),
            tf.reshape(ip['qx'], [-1,84,84,3])],
            axis=0)

        all_feats = cnn(mball, isTr=isTr, reuse=reuse)
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
