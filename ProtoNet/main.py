import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import EpisodeGenerator
from lib.networks import ProtoNet 
from lib.utils import lr_scheduler
from config.loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='protonet')
    parser.add_argument('--init', dest='initial_step', default=0, type=int)
    parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
    parser.add_argument('--qs', dest='qsize', default=15, type=int)
    parser.add_argument('--nw', dest='nway', default=5, type=int)
    parser.add_argument('--ks', dest='kshot', default=1, type=int)
    parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
    parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../data_npy')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--dset', dest='dataset_name', default='miniImagenet')
    parser.add_argument('--gpuf', dest='gpu_frac', default=0.41, type=float)
    parser.add_argument('--name', dest='model_name', default='protonet')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--config', dest='config', default='miniimg')
    parser.add_argument('--tk', dest='test_kshot', default=0, type=int)
    parser.add_argument('--arch', dest='arch', default='simple')
    parser.add_argument('--mbsize', dest='mbsize', default=4, type=int)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--wd_rate', dest='wd_rate', default=1e-4, type=float)
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    args = parser.parse_args()
    return args

def validate(test_net, test_gen):
    accs, losses = [], []
    #np.random.seed(299)
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
#    np.random.seed()

if __name__=='__main__': 
    args = parse_args() 
    print ('='*50) 
    print ('args::') 
    if args.arch=='resnet':
        args.epoch_list = '0.5,0.8'
        args.lr = 1e-1
    elif args.arch=='simple' or args.arch=='vggnet':
        args.lr = 1e-3
        args.epoch_list = '0.7'
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway
    kshot = args.kshot 
    qsize = args.qsize 
    test_kshot = args.kshot if args.test_kshot==0 else args.test_kshot

    lr_ph = tf.placeholder(tf.float32) 
    protonet = ProtoNet(args.model_name, nway, test_kshot, qsize, 
            isTr=True, arch=args.arch, config=args.config, mbsize=args.mbsize)

    loss = protonet.outputs['loss'] #+ args.wd_rate*w2loss
    acc = protonet.outputs['acc']
    
    # only evaluates 5way - kshot
    test_net = ProtoNet(args.model_name, args.nway, test_kshot, qsize, 
            isTr=False, reuse=True, arch=args.arch, config=args.config, 
            mbsize=1)

#    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#    for i,v in enumerate(vs):
#        print (i,v)
    
    if args.arch=='simple' or args.arch=='vggnet':
        print ('Adam opt used')
        opt = tf.train.AdamOptimizer(lr_ph)
    else:
        print ('momentum opt used')
        opt = tf.train.MomentumOptimizer(lr_ph, args.momentum)
    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_op = opt.minimize(loss) 
    saver = tf.train.Saver()
        
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.dataset_name + '.ckpt')
        saver.restore(sess, loc)
    
    config = load_config(args.config)
    train_gen = EpisodeGenerator(args.dataset_dir, 'train', config)
    test_gen = EpisodeGenerator(args.dataset_dir, 'test', config)
    if args.train:
        max_iter = train_gen.dataset_size[args.dataset_name] \
                * args.max_epoch \
                // (nway * qsize * args.mbsize)
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * (nway * qsize * args.mbsize) \
                    // train_gen.dataset_size[args.dataset_name]
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

            if i % save_step == 0 and i != 0: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # bline/
                        args.dataset_name + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(test_net, test_gen)
