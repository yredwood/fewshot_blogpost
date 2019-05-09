import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

#from lib.episode_generator import BatchGenerator
from tieredloader import TieredImageNet
from lib.utils import lr_scheduler, l2_loss
from net import ConvNet
from mamlnet import MAMLNet 
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
    parser.add_argument('--name', dest='model_name', default='transfer_pretrain')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--gpuf', dest='gpu_frac', default=0.41, type=float)
    parser.add_argument('--train', dest='train', default=1, type=int)
    parser.add_argument('--vali', dest='val_iter', default=60, type=int)
    parser.add_argument('--config', dest='config', default='miniimg')
    parser.add_argument('--bs', dest='batch_size', default=32, type=int)
    parser.add_argument('--arch', dest='arch', default='simple', type=str)
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    parser.add_argument('--wd', dest='weight_decay', default=0.0005, type=float)
    parser.add_argument('--meta_arch', dest='meta_arch', default='maml')
    args = parser.parse_args()
    return args

def validate(net, dataset):
    accs, losss = [], [] 
    for i in range(args.val_iter):
        #fd = {net.inputs['x']: x, net.inputs['y']: y}
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
    if args.arch=='simple':
        args.lr = 1e-2
        args.lr_decay = 0.1
        args.epoch_list = '0.5,0.8'
    elif args.arch=='wdres':
        args.lr = 1e-1
        args.lr_decay = 0.2
        args.epoch_list = '0.3,0.6,0.8'
    elif args.arch=='res12':
        args.lr = 1e-3
        args.lr_decay = 0.5
        args.epoch_list = '0.3,0.6,0.8'
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

    lr_ph = tf.placeholder(tf.float32) 
    
    if args.meta_arch == 'maml':
        Net = MAMLNet
    elif args.meta_arch == 'proto':
        Net = ConvNet

    
    train_dataset = TieredImageNet('tiered', 'images/train')
    tr_x, tr_y = train_dataset.input_fn(args.batch_size, train_mode=True)
    train_input_dict = {'x': tr_x, 'y': tr_y, 'lr': lr_ph}

    test_dataset = TieredImageNet('tiered', 'images/train')
    te_x, te_y = train_dataset.input_fn(args.batch_size, train_mode=False)
    test_input_dict = {'x': te_x, 'y': te_y}
    

    trainnet = Net(args.model_name, nway=train_dataset.n_classes,
            isTr=True, config=config, arch=args.arch, input_dict=train_input_dict)
    testnet = Net(args.model_name, nway=test_dataset.n_classes, reuse=True,
            isTr=False, config=config, arch=args.arch, input_dict=test_input_dict)
    
    opt = tf.train.MomentumOptimizer(lr_ph, 0.9)
    #opt = tf.train.AdamOptimizer(lr_ph)
    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]
    
    wd_loss = l2_loss() * args.weight_decay

    if args.meta_arch == 'maml':
        train_op = opt.minimize(trainnet.outputs['loss'] + wd_loss) 
    elif args.meta_arch == 'proto':
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op = opt.minimize(trainnet.outputs['loss'] + wd_loss) 
    saver = tf.train.Saver()
    
#    # get # of params
#    num_params = 0
#    for i, v in enumerate(update_var_list):
#        shape = v.get_shape()
#        nv = 1
#        for dim in shape:
#            nv *= dim.value
#        num_params += nv
#        print (i, v.name.replace(args.model_name,''), v.shape)
#    print ('TOTAL NUM OF PARAMS: {}'.format(num_params))

    vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for i,v in enumerate(vs):
        print (i,v.name.replace(args.model_name, ''))
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())

    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.config + '.ckpt')
        saver.restore(sess, loc)
    
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

            fd = {lr_ph: lr}
            runs = [trainnet.outputs['acc'], trainnet.outputs['loss'], train_op]
            p1, p2, _ = sess.run(runs, fd)

            avger += [p1, p2, 0, time.time() - stt] 
            #print (p1, p2, time.time()-stt)
#            print ('expect time : {:.3f}'.format(\
#                    (time.time()-stt) * train_dataset.n_images))

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
