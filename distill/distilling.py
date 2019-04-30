import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import BatchGenerator
from lib.utils import lr_scheduler, l2_loss
from net import ConvNet
from config.loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='pretrain the network')
    parser.add_argument('--init', dest='initial_step', default=0, type=int) 
    parser.add_argument('--maxe', dest='max_epoch', default=100, type=int)
    parser.add_argument('--sh', dest='show_epoch', default=1, type=int)
    parser.add_argument('--sv', dest='save_epoch', default=10, type=int)
    parser.add_argument('--pr', dest='pretrained', default=False, type=bool)
    parser.add_argument('--data', dest='dataset_dir', default='../data_npy')
    parser.add_argument('--model', dest='model_dir', default='../models')
    parser.add_argument('--name', dest='model_name', default='distilling')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--gpuf', dest='gpu_frac', default=0.41, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--config', dest='config', default='miniimg')
    parser.add_argument('--bs', dest='batch_size', default=32, type=int)
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    parser.add_argument('--tarch', dest='teacher_arch', default='vgg')
    parser.add_argument('--sarch', dest='student_arch', default='simple')
    parser.add_argument('--tname', dest='tmodel_name', default='')
    parser.add_argument('--sname', dest='smodel_name', default='')
    parser.add_argument('--tpr', dest='teacher_predtrained', default='')

    args = parser.parse_args()
    return args

def validate(net, dataset):
    accs, losss = [], [] 
    for i in range(args.val_iter):
        x, y = dataset.get_batch(args.batch_size, 'val')
        fd = {net.inputs['x']: x, net.inputs['y']: y}
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
    if args.student_arch=='simple':
        args.lr = 1e-2
        args.lr_decay = 0.1
        args.epoch_list = '0.5,0.8'
    else: 
        print ('somethings wrong')
        exit()
    config = load_config(args.config)
    tr_dataset = config['TRAIN_DATASET']

    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    dataset = BatchGenerator(args.dataset_dir, 'train', config)
    test_dataset = BatchGenerator(args.dataset_dir, 'test', config)
    lr_ph = tf.placeholder(tf.float32) 

    trainnet = ConvNet(args.model_name, dataset.n_classes,
            isTr=True, config=config, arch=args.student_arch)
    testnet = ConvNet(args.model_name, dataset.n_classes, reuse=True,
            isTr=False, config=config, arch=args.student_arch)


    teachernet = ConvNet(args.tmodel_name, dataset.n_classes,
            isTr=False, config=config, arch=args.teacher_arch)

#    logit_diff = tf.reduce_mean(tf.reduce_sum(trainnet.outputs['pred'] \
#            - teachernet.outputs['pred'], axis=1))
    logit_diff = 0.05*(trainnet.outputs['pred'] - teachernet.outputs['pred'])**2
    logit_diff = tf.reduce_mean(tf.reduce_sum(logit_diff, axis=1))
    
    opt = tf.train.MomentumOptimizer(lr_ph, 0.9)
    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        #train_op = opt.minimize(trainnet.outputs['loss']) 
        train_op = opt.minimize(logit_diff+trainnet.outputs['loss'])
    saver = tf.train.Saver()

    vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i,v in enumerate(vs):
        print (i,v)
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
        
    # restore teacher model
    loc = os.path.join(args.model_dir,
            args.tmodel_name,
            args.config+'.ckpt')
    print ('teacher model is restored from {}'.format(loc))
    tvs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            scope=args.tmodel_name+'/*')
    tsaver = tf.train.Saver(tvs)
    tsaver.restore(sess, loc)
    
#    print ('teacher model performace')
#    validate(teachernet, test_dataset)


    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.config + '.ckpt')
        saver.restore(sess, loc)
        print ('student restored from {}'.format(loc))

    
    if args.train:
        max_iter = config['size'] * args.max_epoch \
                // (args.batch_size)
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * args.batch_size \
                    // config['size']
            lr = lr_scheduler(cur_epoch, args.lr,
                    epoch_list=decay_epoch, decay=args.lr_decay)

            x, y = dataset.get_batch(args.batch_size, 'train', aug=args.aug)
            fd = {trainnet.inputs['x']: x, trainnet.inputs['y']: y, lr_ph: lr,
                    teachernet.inputs['x']: x}
            runs = [trainnet.outputs['acc'], trainnet.outputs['loss'], 
                    logit_diff, train_op]
            p1, p2, p3, _ = sess.run(runs, fd)

            avger += [p1, p2, p3, time.time() - stt] 

            if i % show_step == 0 and i != 0: 
                avger /= show_step
                print ('========= epoch : {:8d}/{} ========='\
                        .format(cur_epoch, args.max_epoch))
                print ('Training - ACC: {:.3f} '
                    '| LOSS: {:.3f}   '
                    '| logit df: {:.3f}'
                    '| lr : {:.3f}    '
                    '| in {:.2f} secs '\
                    .format(avger[0], 
                        avger[1], 
                        avger[2],
                        lr, avger[3]*show_step))
                validate(testnet, test_dataset)
                avger[:] = 0

            if i % save_step == 0 and i != 0: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # bline/
                        args.config + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(testnet, test_dataset)
