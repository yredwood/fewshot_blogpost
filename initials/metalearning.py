import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import EpisodeGenerator
from net import ConvNet
from mamlnet import MAMLNet
from lib.utils import lr_scheduler, l2_loss
from lib.utils import op_restore_possible_vars
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
    parser.add_argument('--meta_arch', dest='meta_arch', default='maml')
    parser.add_argument('--mbsize', dest='mbsize', default=4, type=int)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    parser.add_argument('--wd', dest='weight_decay', default=0.0005, type=float)
    parser.add_argument('--inlr', dest='inner_lr', default=0.01, type=float)
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
                    5, test_kshot, args.qsize)
            sx.append(_sx)
            sy.append(_sy)
            qx.append(_qx)
            qy.append(_qy)
        
        if args.meta_arch=='proto':
            fd = {\
                test_net.inputs['sx']: sx,
                test_net.inputs['qx']: qx,
                test_net.inputs['qy']: qy}
        elif args.meta_arch=='maml':
            fd = {\
                test_net.inputs['sx']: sx,
                test_net.inputs['sy']: sy,
                test_net.inputs['qx']: qx,
                test_net.inputs['qy']: qy,
                test_net.inputs['lr_alpha']: args.inner_lr
                }
        outputs = [test_net.outputs['acc'], test_net.outputs['loss']]
        acc, loss = sess.run(outputs, fd)
        accs.append(np.mean(acc))
        losses.append(np.mean(loss))
    print ('Validation - ACC: {:.3f} ({:.3f})'
        '| LOSS: {:.3f}   '\
        .format(np.mean(accs) * 100., 
        np.std(accs) * 100. * 1.96 / np.sqrt(args.val_iter),
        np.mean(losses)))
#    a = sess.run(test_net.alpha)
#    print ('alpha: {:.4f}'.format(np.mean(a)))

if __name__=='__main__': 
    args = parse_args() 
    if args.arch=='simple':
        pass
        #args.lr= 1e-2
        #args.epoch_list = '0.5,0.8'
#        args.lr = 1e-3
#        args.epoch_list = '0.7'
    elif args.arch=='res':
        #args.lr = 1e-1
        #args.epoch_list = '0.3,0.6,0.8'
        args.lr_decay = 0.2
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

    lr_ph = tf.placeholder(tf.float32) 

    if args.meta_arch=='maml':
        Net = MAMLNet
    elif args.meta_arch=='proto':
        Net = ConvNet
    train_net = Net(args.model_name,
            isTr=True, reuse=False, config=config, 
            arch=args.arch, metalearn=True,
            mbsize=args.mbsize, nway=nway,
            kshot=kshot, qsize=qsize)

    test_net = Net(args.model_name,
            isTr=False, reuse=True, config=config,
            arch=args.arch, metalearn=True,
            mbsize=1, nway=5, kshot=test_kshot, qsize=qsize)

    opt = tf.train.MomentumOptimizer(lr_ph, args.momentum)
    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]

    tr_vars_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    tr_vars = []
    for i,v in enumerate(tr_vars_all):
        vname = v.name.replace(args.model_name, '')
        for kk in range(3):
            if 'd{}'.format(kk) in vname:
                tr_vars.append(v)
        if 'h_alpha' in vname:
            tr_vars.append(v)

    for i,v in enumerate(tr_vars):
        print (i,v)
    
#    wd_loss = l2_loss() * args.weight_decay
#    if args.meta_arch=='maml':
#        train_op = opt.minimize(train_net.outputs['loss'])
#    elif args.meta_arch=='proto':
#        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        with tf.control_dependencies(update_op):
#            train_op = opt.minimize(train_net.outputs['loss']) #, var_list=tr_vars) 
        
    saver = tf.train.Saver()
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    
    if not args.train:
        loc = os.path.join(args.model_dir,
                args.model_name,
                args.config + '.ckpt')
        print ('restored from {}'.format(loc))
        saver.restore(sess, loc)
    else:
        if args.pretrained!='none':
            restore_op = op_restore_possible_vars(args.pretrained,
                    print_details=True)
            sess.run(restore_op)
    
    test_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    train_gen = EpisodeGenerator(args.dataset_dir, 'train', config)


    vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    w0 = sess.run(vs)

    if args.train:
        max_iter = config['size'] \
                * args.max_epoch \
                // (nway * qsize * args.mbsize)
        #max_iter = 1000
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * (nway * qsize * args.mbsize) \
                    // config['size']
            lr = lr_scheduler(cur_epoch, args.lr, 
                    epoch_list=decay_epoch, decay=args.lr_decay)

            sx, sy, qx, qy = [], [], [], []
            for _ in range(args.mbsize):
                _sx, _sy, _qx, _qy = train_gen.get_episode(
                        nway, kshot, qsize, aug=args.aug)
                sx.append(_sx)
                sy.append(_sy)
                qx.append(_qx)
                qy.append(_qy)
            
            if args.meta_arch=='proto':
                fd = {
                    train_net.inputs['sx']: sx,
                    train_net.inputs['qx']: qx,
                    train_net.inputs['qy']: qy,
                    lr_ph: lr}
            elif args.meta_arch=='maml':
                fd = {
                    train_net.inputs['sx']: sx, 
                    train_net.inputs['sy']: sy,
                    train_net.inputs['qx']: qx,
                    train_net.inputs['qy']: qy,
                    train_net.inputs['lr_alpha']: args.inner_lr,
                    train_net.inputs['lr_beta']: lr}

            p1, p2, _ = sess.run([\
                    train_net.outputs['acc'], 
                    train_net.outputs['loss'],
                    train_net.train_op], fd)
                    #tf.nn.softmax(train_net.outputs['pred']),
            
            avger += [np.mean(p1), np.mean(p2), 0, time.time() - stt] 
            #print (sess.run(train_net.alpha))
            #print (np.mean(sess.run(tr_vars[-1])))

            #print (p1, p2, np.mean(p3, 1))
            #pdb.set_trace()

            if i % show_step == 0 and i != 0: 

                avger /= show_step
                print ('========= epoch : {:8d}/{} ========='\
                        .format(cur_epoch, args.max_epoch))
                print ('Training - ACC: {:.3f} '
                    '| LOSS: {:.3f}   '
                    '| lr : {:.3f}    '
                    '| in {:.2f} secs '\
                    .format(avger[0], 
                        avger[1], lr, avger[3]*show_step))
                validate(test_net, test_gen)
                avger[:] = 0

                    

#                w1 = sess.run(vs)
#                wds = []
#                for wn in range(len(w1)):
#                    if len(w1[wn].shape) > 1:
#                        wd = np.linalg.norm(w0[wn]-w1[wn])
#                        wds.append(wd)
#                print (wds)
#                print ('Avg diff: {}'.format(np.mean(wds)))

            if (i % save_step == 0 and i != 0) or i == max_iter: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # bline/
                        args.config + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(test_net, test_gen)
