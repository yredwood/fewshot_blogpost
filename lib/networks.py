import tensorflow as tf 
import os
import pdb
from lib.layers import Network, ce_logit, cross_entropy, tf_acc
from lib.vggnet import VGGNet
from lib.resnet import ResNet
from lib.resadapters import ResNetAdapter


class TransferNet(Network):
    def __init__(self, name, n_classes, isTr, reuse=False, 
            input_dict=None, config='cl_cifar10', architecture='simple'):
        self.name = name
        self.n_classes = n_classes
        self.hw = 32 if config=='cl_cifar10' else 84
        if input_dict is None:
            self.inputs = {
                    'x': tf.placeholder(tf.float32, [None,self.hw,self.hw,3]),
                    'y': tf.placeholder(tf.float32, [None,n_classes])
            }
        else:
            self.inputs = input_dict
        self.architecture = architecture
        self.config = config

        self.outputs = {}
        with tf.variable_scope(name):
            self._build_network(isTr, reuse, architecture)
    
    def _build_network(self, isTr, reuse, architecture):
        x = tf.transpose(self.inputs['x'], [0,3,1,2])
        if architecture == 'simple':
            f = self.simple_conv(x, reuse=reuse, isTr=isTr)
        elif architecture == 'vggnet':
            vgg = VGGNet(architecture, reuse=reuse, config=self.config)
            f = vgg.net(x, isTr)
        elif architecture == 'resnet':
            resnet = ResNet('resnet', 20, reuse=reuse, config=self.config)
            f = resnet(x, isTr)
            f = tf.layers.flatten(f)
#        elif architecture == 'resadapt':
#            resnet = ResNetAdapter('resadapt', 20, reuse=reuse, 
#                    config=self.config, n_adapt=2)
#            f = resnet(x, isTr, 0)
#            f = tf.layers.flatten(f)

        pred = self.dense(f, self.n_classes, name='trtask_dense', reuse=reuse)
        loss = cross_entropy(tf.nn.softmax(pred), self.inputs['y'])
        acc = tf_acc(pred, self.inputs['y'])

        self.outputs['embedding'] = f
        self.outputs['pred'] = pred
        self.outputs['loss'] = loss
        self.outputs['acc'] = acc

class ProtoNet(Network):
    def __init__(self, name, nway, kshot, qsize, isTr, 
            reuse=False, mbsize=1,
            arch='simple', config=None):
        self.name = name
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.mbsize = mbsize
#        if arch=='simple':
#            self.hdim = 1600
#        elif arch=='resnet':
#            self.hdim = 512
#        elif arch=='vgg':
#            self.hdim = 2048
        
        self.arch = arch
        self.config = config # dataset type to use

        self.inputs = {\
                'sx': tf.placeholder(tf.float32, [mbsize,nway*kshot,84,84,3]),
                'qx': tf.placeholder(tf.float32, [mbsize,nway*qsize,84,84,3]),
                'qy': tf.placeholder(tf.float32, [mbsize,nway*qsize,nway]),
                'lr': tf.placeholder(tf.float32),
                'tr': tf.placeholder(tf.bool)}
        self.outputs = {}

        with tf.variable_scope(name):
            self._build_network(isTr, reuse=reuse, arch=arch)

    def _build_network(self, isTr, reuse, arch='simple'):
        ip = self.inputs
            
        all_mb_data = tf.concat([
                tf.reshape(ip['sx'], [-1,84,84,3]),
                tf.reshape(ip['qx'], [-1,84,84,3])],
                axis=0)
        
        all_feats = self.base_cnn(all_mb_data,
                isTr=isTr, reuse=reuse, arch=arch)

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

        self.probe = [meta_acc, meta_pred]
#        self.outputs['loss'] = tf.losses.softmax_cross_entropy(
#                onehot_labels=ip['qy'], logits=sim)
#        self.outputs['acc'] = tf_acc(prediction, ip['qy'])

    def base_cnn(self, in_x, isTr, reuse=False, arch='simple'):
        in_x = tf.transpose(in_x, [0,3,1,2])
        if arch == 'simple':
            f = self.simple_conv(in_x, reuse=reuse, isTr=isTr)
        elif arch == 'resnet':
            resnet = ResNet('resnet', 20,
                    reuse=reuse, config=self.config)
            f = resnet(in_x, isTr)
            f = tf.layers.flatten(f)
        elif arch == 'vggnet':
            vgg = VGGNet('vggnet', reuse=reuse, 
                    config=self.config)
            f = vgg.net(in_x, isTr)
        self.hdim = f.get_shape().as_list()[-1]
        return f

    def uclidean_sim(self, xs, xq):
        xs = tf.expand_dims(xs, 0)
        xq = tf.expand_dims(xq, 1)
        emb = (xs-xq)**2
        dist = tf.reduce_mean(emb, axis=2)
        return -dist

    def cosine_sim(self, xs, xq):
        xs = tf.nn.l2_normalize(xs, 1)
        xq = tf.nn.l2_normalize(xq, 1)
        
        xs = tf.expand_dims(xs, 0)
        xq = tf.expand_dims(xq, 1)
        return tf.reduce_sum(tf.multiply(xs, xq), axis=2)
        

class MAMLNet(Network):
    def __init__(self, name, nway, kshot, qsize, mbsize, norm=False, reuse=False, 
            inner_loop_iter=5, stop_grad=True, isTr=True, input_dict=None):
        self.name = name
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.mbsize = mbsize
        self.norm = norm 
        self.inner_loop_iter = inner_loop_iter
        self.stop_grad = stop_grad
        self.hdim = 800
        
        if input_dict is None:
            self.inputs = {\
                'sx': tf.placeholder(tf.float32, [mbsize,nway*kshot,84,84,3], 
                    name='ph_sx'),
                'sy': tf.placeholder(tf.float32, [mbsize,nway*kshot,nway],
                    name='ph_sy'),
                'qx': tf.placeholder(tf.float32, [mbsize,nway*qsize,84,84,3],
                    name='ph_qx'),
                'qy': tf.placeholder(tf.float32, [mbsize,nway*qsize,nway],
                    name='ph_qy'),
                'tr': tf.placeholder(tf.bool, name='ph_isTr'),
                'lr_alpha': tf.placeholder(tf.float32, [], name='ph_alpha'),
                'lr_beta': tf.placeholder(tf.float32, [], name='ph_beta')}
        else:
            self.inputs = input_dict

        self.outputs = {\
                'preda': None,
                'predb': None,
                'lossa': None,
                'lossb': None,
                'accuracy': None,
                'embedding': None}

        with tf.variable_scope(self.name, reuse=reuse):
            self._build_network(isTr, reuse=reuse)

    def _build_network(self, isTr, reuse=False):
        ip = self.inputs
        weights = self.construct_weights()

        meta_lossbs = []; meta_predbs = []; # (# metabatch, # inner updates)
        _ = self.forward(ip['sx'][0], weights, reuse=False)
        
        def singlebatch_graph(inputs):
            sx, sy, qx, qy = inputs
            # when sinlge batch is given, get the lossbs, predbs
            # sy.shape : (nk, n)  / qy.shape : (nq, n)
            lossbs, predbs = [], []
            preda = self.forward(sx, weights, reuse=True)
            lossa = tf.reduce_mean(ce_logit(preda, sy))
            grads = tf.gradients(lossa, list(weights.values()))
            if self.stop_grad:
                grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))
            adapted_weights = dict(zip(weights.keys(), 
                [weights[key] - ip['lr_alpha'] * gradients[key] \
                        for key in weights.keys()]))
            predb = self.forward(qx, adapted_weights, reuse=True)
            lossb = ce_logit(predb, qy)

            predbs.append(predb)
            lossbs.append(lossb)

            for _ in range(self.inner_loop_iter - 1):
                preda, emba = self.forward(sx, adapted_weights, reuse=True, get_emb=True)
                lossa = tf.reduce_mean(ce_logit(preda, sy))
                grads = tf.gradients(lossa, list(adapted_weights.values()))
                if self.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(adapted_weights.keys(), grads))
                adapted_weights = dict(zip(adapted_weights.keys(),
                    [adapted_weights[key] - ip['lr_alpha'] * gradients[key] \
                            for key in adapted_weights.keys()]))
                predb, embb = self.forward(qx, adapted_weights, reuse=True, get_emb=True)
                lossb = ce_logit(predb, qy)

                predbs.append(predb)
                lossbs.append(lossb)
            # predbs.shape = (# of inner iter, nq, n)
            # lossbs.shape = (# of inner iter, nq)
            return predbs, lossbs, emba, embb

        elems = (ip['sx'], ip['sy'], ip['qx'], ip['qy'])
        out_dtype = ([tf.float32]*self.inner_loop_iter,
                [tf.float32]*self.inner_loop_iter, tf.float32, tf.float32)
        meta_predbs, meta_lossbs, meta_emba, meta_embb = \
                tf.map_fn(singlebatch_graph, elems, dtype=out_dtype,
                parallel_iterations=self.mbsize)

        # meta_predbs.shape = (#inner_loop, meta_batch, nq, n)
        # meta_lossbs.shape = (#inner_loop, meta_batch, nq)
        # meta_emba.shape = (meta_batch, nk, hdim)
        # meta_embb.shape = (meta_batch, nq, hdim)
        _p = tf.reshape(meta_emba, [self.mbsize,self.nway,self.kshot,self.hdim])
        _p = tf.reduce_mean(_p, axis=2)
        _p = tf.expand_dims(_p, axis=1) # (mb, 1, n, h)

        _q = tf.expand_dims(meta_embb, 2) # (mb, nk, 1, h)
        embedding = (_p - _q)**2

        
        self.outputs['predb'] = meta_predbs
        self.outputs['lossb'] = meta_lossbs 
        self.outputs['embedding'] = embedding
        #self.outputs['lossa'] = meta_lossas
        # lossb (metabatch, innerup, nq)

        if isTr:
            opt_loss = tf.reduce_mean(meta_lossbs, (1,2))[-1] 
            opt = tf.train.AdamOptimizer(ip['lr_beta'])
            gvs = opt.compute_gradients(opt_loss)
            gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
            self.train_op = opt.apply_gradients(gvs)
            self.gvs = gvs

        self.outputs['accuracy'] = \
                [tf_acc(meta_predbs[-1][mi], ip['qy'][mi]) \
                for mi in range(self.mbsize)]
#        self.outputs['accuracy'] = [tf_acc(mp, ip['qy'][mi]) \
#                for mi, mp in enumerate(meta_predbs[-1])]

    def construct_weights(self):
        weights = {}
        f32 = tf.float32
        conv_init = tf.contrib.layers.xavier_initializer_conv2d(dtype=f32)
        fc_init = tf.contrib.layers.xavier_initializer(dtype=f32)
        
        # first conv 3->32
        weights['conv{}'.format(1)] = tf.get_variable('conv{}'.format(1),
                [3,3,3,32], initializer=conv_init, dtype=f32)
        weights['bias{}'.format(1)] = tf.get_variable('bias{}'.format(1),
                initializer=tf.zeros([32]))
        for i in range(2, 5):
            weights['conv{}'.format(i)] = tf.get_variable('conv{}'.format(i),
                    [3,3,32,32], initializer=conv_init, dtype=f32)
            weights['bias{}'.format(i)] = tf.get_variable('bias{}'.format(i),
                    initializer=tf.zeros([32]))
        weights['w5'] = tf.get_variable('w5', 
                [32*5*5, self.nway], initializer=fc_init)
        weights['b5'] = tf.get_variable('b5',
                initializer=tf.zeros([self.nway]))
        return weights

    def forward(self, x, weights, reuse=False, scope='', get_emb=False):
        def conv(x, w, b, reuse, scope):
            h = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b
            h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu,
                    reuse=reuse, scope=scope)
            return tf.nn.max_pool(h, [1,2,2,1], [1,2,2,1], 'VALID')
    
        for i in range(1, 5):
            x = conv(x, weights['conv{}'.format(i)], weights['bias{}'.format(i)],
                    reuse, scope+'{}'.format(i))
        dim = 1
        for s in x.get_shape().as_list()[1:]:
            dim *= s
        x = tf.reshape(x, [-1, dim])
        x = x / (tf.norm(x, axis=1, keep_dims=True) + 1e-8) * 1e+1
        out = tf.matmul(x, weights['w5']) + weights['b5']
        if get_emb:
            return out, x
        return out
