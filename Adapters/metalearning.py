import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import EpisodeGenerator
from net import AdapterNet
from lib.utils import lr_scheduler, l2_loss
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
    parser.add_argument('--fix_univ', dest='fix_univ', default=1, type=int)
    parser.add_argument('--use_adapt', dest='use_adapt', default=1, type=int)
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
    #args.lr= 1e-3
    args.epoch_list = '0.7'
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
    protonet = AdapterNet(args.model_name, n_classes=None,
            isTr=True, reuse=False, 
            config=config, 
            n_adapt=2, use_adapt=args.use_adapt,
            fixed_univ=args.fix_univ, metalearn=True,
            mbsize=args.mbsize, nway=nway,
            kshot=kshot, qsize=qsize)

    loss = protonet.outputs['loss'] 
    acc = protonet.outputs['acc']
    
    # only evaluates 5way - kshot
    test_net = AdapterNet(args.model_name, n_classes=None,
            isTr=False, reuse=True, 
            config=config,
            n_adapt=2, use_adapt=args.use_adapt,
            fixed_univ=args.fix_univ, metalearn=True,
            mbsize=1, nway=nway, kshot=kshot,
            qsize=qsize)

#    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#    for i,v in enumerate(vs):
#        print (i,v)
    update_var_list = protonet.var_list
    w2loss = tf.add_n([tf.nn.l2_loss(v) for v in update_var_list]) \
            * args.wd_rate
    #opt = tf.train.MomentumOptimizer(lr_ph, args.momentum)
    opt = tf.train.AdamOptimizer(lr_ph)

    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]

#    for i,v in enumerate(update_var_list):
#        print (i,v)

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_op = opt.minimize(loss, var_list=update_var_list) 
    saver = tf.train.Saver()
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    if args.pretrained!='none':
#        loc = os.path.join(args.model_dir,
#                args.model_name, 
#                args.dataset_name + '_64.learneing.ckpt')
        if 'cifar' in args.model_name:
            loc = '../models/AdaptNet_cl_cifar100_aug1/cl_cifar100.ckpt'
        elif 'tiered' in args.model_name:
            loc = '../models/AdaptNet_cl_tieredImagenet_aug1/cl_tieredImagenet.ckpt'
        elif 'omniglot' in args.model_name:
            loc = '../models/AdaptNet_cl_omniglot_aug1/cl_omniglot.ckpt'
        else:
            loc = '../models/AdaptNet_cl_miniImagenet_aug1/cl_miniImagenet.ckpt'
            loc = '../models/AdaptNet_miniImagenet_aug0_fix0_U0_1205/miniImagenet.ckpt'
        loc = args.pretrained
        saved_list = tf.contrib.framework.list_variables(loc)
        svname_list = []
        for svname, svshape in saved_list:
            svnew = svname.split('/')
            if len(svnew) <= 2:
                svname_list.append(svname)
                continue
            svname_list.append('/'.join(svnew[2:]))

        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # todo: if there's any var of vs in saved_list, then restore it
        assign_op = []
        for v in vs:
            # remove prefix
            nn = v.name.split('/')
            if len(nn) <= 2:
                continue
            vname = '/'.join(nn[2:]).split(':')[0]
            if vname in svname_list:
                # restore it 
                orig_svname = saved_list[svname_list.index(vname)][0]
                svar = tf.contrib.framework.load_variable(loc, orig_svname)
                assign_op.append(tf.assign(v, svar))
                #print ('=={} is loaded'.format(vname))
            else:
                pass
                #print ('--{} is not in the pretrained model. Not restored'\
                #        .format(vname))
        sess.run(assign_op)
        print ('retore success from {}'.format(loc))


    if not args.train:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.config + '.ckpt')
        saver.restore(sess, loc)

    
    test_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    if args.train:
        train_gen = EpisodeGenerator(args.dataset_dir, 'train', config)
        max_iter = config['size'] \
                * args.max_epoch \
                // (nway * qsize * args.mbsize)
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * (nway * qsize * args.mbsize) \
                    // config['size']
            lr = lr_scheduler(cur_epoch, args.lr, 
                    epoch_list=decay_epoch, decay=args.lr_decay)
#            sx, sy, qx, qy = train_gen.get_episode(
#                    nway, kshot, qsize, aug=False) 
            sx, sy, qx, qy = [], [], [], []
            for _ in range(args.mbsize):
                _sx, _sy, _qx, _qy = train_gen.get_episode(
                        nway, kshot, qsize, aug=args.aug)
                sx.append(_sx)
                sy.append(_sy)
                qx.append(_qx)
                qy.append(_qy)

            fd = {
                protonet.inputs['sx']: sx,
                protonet.inputs['qx']: qx,
                protonet.inputs['qy']: qy,
                lr_ph: lr}

            p1, p2, _ = sess.run([acc, loss, train_op], fd)
            #p1, p2 = sess.run([acc, loss], fd)
            avger += [p1, p2, 0, time.time() - stt] 

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

            if (i % save_step == 0 and i != 0) or i == max_iter: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # bline/
                        args.config + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(test_net, test_gen)
