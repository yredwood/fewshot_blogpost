import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import BatchGenerator
from lib.networks import TransferNet, cross_entropy, tf_acc
from config.loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='pretrain the network')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
    parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
    parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
    parser.add_argument('--pr', dest='pretrained', default=1, type=int)
    parser.add_argument('--data', dest='dataset_dir', default='../data_npy')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--dset', dest='dataset_name', default='miniImagenet')
    parser.add_argument('--name', dest='model_name', default='protonet')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--config', dest='config', default='miniimg')
    parser.add_argument('--qs', dest='qsize', default=15, type=int)
    parser.add_argument('--nw', dest='nway', default=5, type=int)
    parser.add_argument('--ks', dest='kshot', default=1, type=int)
    args = parser.parse_args()
    return args

def validate(net, dataset):
    accs, losss = [], [] 
    for i in range(args.val_iter):
        x, y = dataset.get_batch(args.batch_size, 'val')
        fd = {net.inputs['x']: x, net.inputs['y']: y, lr_ph: lr}
        runs = [net.outputs['acc'], net.outputs['loss']]
        acc, loss = sess.run(runs, fd)
        accs.append(acc)
        losss.append(loss)

    print ('Validation - Acc: {:.3f} ({:.3f})'
        ' | Loss {:.3f} '\
        .format(np.mean(accs)*100,
            np.std(accs)*100.*1.96/np.sqrt(args.val_iter),
            np.mean(losss)))


if __name__=='__main__': 
    args = parse_args() 
    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway 
    kshot = args.kshot
    qsize = args.qsize

    config = load_config(args.config)
    lr_ph = tf.placeholder(tf.float32) 
    y_ph = tf.placeholder(tf.float32, [None,nway])
    # 64: train_n_classes
    trainnet = TransferNet(args.model_name, 64, isTr=True) 
    testnet = TransferNet(args.model_name, 64, isTr=False, reuse=True)

    feat_emb = trainnet.outputs['embedding']
    ff = trainnet.dense(feat_emb, nway, name='specific_dense')
    pred = tf.nn.softmax(ff)
    loss = cross_entropy(pred, y_ph)

    ff = testnet.outputs['embedding']
    ff = testnet.dense(ff, nway, name='specific_dense', reuse=True)
    pred = tf.nn.softmax(ff)
    acc = tf_acc(pred, y_ph)

    rvlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, args.model_name+'/*')

    opt = tf.train.AdamOptimizer(lr_ph) 
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_op = opt.minimize(loss) 


    # placeholder for initializing
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_holder = [tf.placeholder(tf.float32, v.shape) for v in var_list]
    init_ops = [var.assign(val) for var, val in zip(var_list, var_holder)]

    saver = tf.train.Saver(rvlist)
    saver_init = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.dataset_name + '.ckpt')
        saver.restore(sess, loc)
        print ('restored from {}'.format(loc))
    
    # save the initial value
    var_val = sess.run(var_list)
    init_fd = {}
    for kk in range(len(var_list)):
        init_fd[var_holder[kk]] = var_val[kk]

    dataset = BatchGenerator(args.dataset_dir, phase='test', config=config)
    # let's first train the model with 5-class 500-examples and test it 
    accs = []
    for i in range(args.val_iter):
        # restore initial status
        sess.run(init_ops, init_fd)
        
        xtr, ytr, xte, yte = dataset.get_episode(nway, kshot, qsize)
        
        batch_size = int(nway*kshot/2)
        for _ in range(50):
            rnd_idx = np.random.choice(nway*kshot, size=batch_size)
            fd = {trainnet.inputs['x']: xtr[rnd_idx],
                    y_ph: ytr[rnd_idx], lr_ph: 1e-3}
            ploss, _ = sess.run([loss, train_op], fd)
        
        # test it 
        fd = {testnet.inputs['x']: xte, y_ph: yte}
        pacc = sess.run(acc, fd)
        accs.append(pacc)
        if i % 10 == 0:
            print ('{:4d} / {:4d} | acc : {:.3f}'.format(\
                i, args.val_iter, pacc))
    
    print ('Test acc: {:.3f} ({:.3f})'.format(\
            np.mean(accs)*100., np.std(accs)*100.*1.96/np.sqrt(args.val_iter)))
