import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import EpisodeGenerator

from net import ConvNet
from lib.utils import lr_scheduler, l2_loss, op_restore_possible_vars
from lib.layers import cross_entropy, tf_acc
from config.loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='protonet')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_epoch', default=150, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int)
    parser.add_argument('--nw', dest='nway', default=5, type=int)
    parser.add_argument('--ks', dest='kshot', default=1, type=int)
    parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
    parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
    parser.add_argument('--pr', dest='pretrained', default='')
    parser.add_argument('--data', dest='dataset_dir', default='../data_npy')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--gpuf', dest='gpu_frac', default=0.41, type=float)
    parser.add_argument('--name', dest='model_name', default='protonet')
    parser.add_argument('--lr', dest='lr', default=1e-1, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--config', dest='config', default='miniimg')
    parser.add_argument('--tk', dest='test_kshot', default=0, type=int)
    parser.add_argument('--arch', dest='arch', default='simple')
    parser.add_argument('--mbsize', dest='mbsize', default=4, type=int)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--wd_rate', dest='wd_rate', default=5e-4, type=float)
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    parser.add_argument('--ttype', dest='test_type', default='learner')
    args = parser.parse_args()
    return args

np.random.seed(2019)
def validate(test_net, test_gen):
    accs, losses = [], []
    test_kshot = args.kshot if args.test_kshot==0 else args.test_kshot
    for _ in range(args.val_iter):
        #sx, sy, qx, qy = test_gen.get_episode(args.nway, test_kshot, args.qsize)
        sx, sy, qx, qy = [], [], [], []
        for _ in range(1):
            _sx, _sy, _qx, _qy = test_gen.get_episode(
                    args.nway, test_kshot, args.qsize)
            sx.append(_sx)
            sy.append(_sy)
            qx.append(_qx)
            qy.append(_qy)

        fd = {\
            test_net.inputs['sx']: sx,
            test_net.inputs['qx']: qx,
            test_net.inputs['qy']: qy}
        outputs = [test_net.outputs['acc'], test_net.outputs['loss']]
        acc, loss = sess.run(outputs, fd)
        accs.append(acc)
        losses.append(loss)
    print ('Validation - ACC: {:.3f} ({:.3f})'
        '| LOSS: {:.3f}   '\
        .format(np.mean(accs) * 100., 
        np.std(accs) * 100. * 1.96 / np.sqrt(args.val_iter),
        np.mean(losses)))

if __name__=='__main__': 
    args = parse_args() 
    args.lr= 1e-3
    config = load_config(args.config)

    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway
    kshot = args.kshot 
    qsize = args.qsize 
    test_kshot = args.kshot if args.test_kshot==0 else args.test_kshot

    if args.test_type=='learner':
        train_net = ConvNet(args.model_name, nway,
                isTr=True, config=config, 
                arch=args.arch, metalearn=False)

        test_net = ConvNet(args.model_name, nway,
                reuse=True, isTr=False, config=config, 
                arch=args.arch, metalearn=False)

        f = train_net.outputs['embedding']
        trpred = train_net.dense(f, nway, name='newdense', reuse=False)
        trloss = cross_entropy(tf.nn.softmax(trpred), train_net.inputs['y'])
        tracc = tf_acc(trpred, train_net.inputs['y'])
        
        f = test_net.outputs['embedding']
        tepred = test_net.dense(f, nway, name='newdense', reuse=True)
        teacc = tf_acc(tepred, test_net.inputs['y'])
        lr_ph = tf.placeholder(tf.float32)

    opt = tf.train.MomentumOptimizer(lr_ph, args.momentum)

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_op = opt.minimize(trloss)
    saver = tf.train.Saver()
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    
    if args.pretrained!='none':
        restore_op = op_restore_possible_vars(args.pretrained, 
                print_details=True)   
        sess.run(restore_op)
    
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    init_vals = sess.run(var_list)
    
    val_holder = [tf.placeholder(tf.float32, v.shape) for v in var_list]
    init_ops = [var.assign(val) for var, val in zip(var_list, val_holder)]

    init_fd = {}
    for kk in range(len(var_list)):
        init_fd[val_holder[kk]] = init_vals[kk]

    test_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    accs = [] 
    for i in range(args.val_iter):
        stt = time.time()
        sess.run(init_ops, init_fd)
        batch_size = np.minimum(32, nway*kshot)
        max_iter = np.maximum(10, 50*nway*kshot//32) # min 10, or 50 epoch

        sx, sy, qx, qy = test_gen.get_episode(
                nway, kshot, qsize, aug=args.aug)
        inner_acc = []
        for _ii in range(max_iter):
            lr = 1e-3 if _ii < max_iter*.7 else 1e-4
            rnd_idx = np.random.choice(nway*kshot, 
                    size=batch_size, replace=False)
            fd = {train_net.inputs['x']: sx[rnd_idx],
                    train_net.inputs['y']: sy[rnd_idx], lr_ph: lr}
            _loss, _acc, _ = sess.run([trloss, tracc, train_op], fd)

            if _ii % 50 == 0:
                fd = {test_net.inputs['x']: qx,
                        test_net.inputs['y']: qy}
                acc = sess.run(teacc, fd)
                inner_acc.append(int(acc*100.))
        print (inner_acc)
        accs.append(acc)

        print ('validation {}/{}, Acc: {:.3f}  in {:.2f} sec'\
                .format(i, args.val_iter, acc*100., time.time()-stt))
            
    print ('=======================')
    print ('avg acc: {:.3f} ({:.3f})'\
            .format(np.mean(accs)*100., 
                np.std(accs)*100*1.96/np.sqrt(args.val_iter)))



#    if args.train:
#        train_gen = EpisodeGenerator(args.dataset_dir, 'train', config)
#        max_iter = config['size'] \
#                * args.max_epoch \
#                // (nway * qsize * args.mbsize)
#        show_step = args.show_epoch * max_iter // args.max_epoch
#        save_step = args.save_epoch * max_iter // args.max_epoch
#        avger = np.zeros([4])
#        for i in range(1, max_iter+1): 
#            stt = time.time()
#            cur_epoch = i * (nway * qsize * args.mbsize) \
#                    // config['size']
#            lr = lr_scheduler(cur_epoch, args.lr, 
#                    epoch_list=decay_epoch, decay=args.lr_decay)
#
#            sx, sy, qx, qy = [], [], [], []
#            for _ in range(args.mbsize):
#                _sx, _sy, _qx, _qy = train_gen.get_episode(
#                        nway, kshot, qsize, aug=args.aug)
#                sx.append(_sx)
#                sy.append(_sy)
#                qx.append(_qx)
#                qy.append(_qy)
#
#            fd = {
#                protonet.inputs['sx']: sx,
#                protonet.inputs['qx']: qx,
#                protonet.inputs['qy']: qy,
#                lr_ph: lr}
#
#            p1, p2, _ = sess.run([acc, loss, train_op], fd)
#            avger += [p1, p2, 0, time.time() - stt] 
#
#            if i % show_step == 0 and i != 0: 
#                avger /= show_step
#                print ('========= epoch : {:8d}/{} ========='\
#                        .format(cur_epoch, args.max_epoch))
#                print ('Training - ACC: {:.3f} '
#                    '| LOSS: {:.3f}   '
#                    '| lr : {:.3f}    '
#                    '| in {:.2f} secs '\
#                    .format(avger[0], 
#                        avger[1], lr, avger[3]*show_step))
#                validate(test_net, test_gen)
#                avger[:] = 0
#
#            if (i % save_step == 0 and i != 0) or i == max_iter: 
#                out_loc = os.path.join(args.model_dir, # models/
#                        args.model_name, # bline/
#                        args.config + '.ckpt')  # cifar100.ckpt
#                print ('saved at : {}'.format(out_loc))
#                saver.save(sess, out_loc)
#    else: # if test only
#        validate(test_net, test_gen)
