import sys, os
sys.path.append('../')

import numpy as np 
import tensorflow as tf 
import argparse
import time
import pdb

from lib.episode_generator import BatchGenerator
from net import AdapterNet
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

def lr_scheduler(cepoch, epl=[50,80], decay=0.1):
    lr_list = [args.lr*decay**(i) for i in range(len(epl)+1)]
    # [lr, lr*decay, lr*decay**2]
    lridx = np.sum(np.array(epl) <= cepoch)
    return lr_list[lridx]
    
if __name__=='__main__': 
    args = parse_args() 
    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    use_adapt = 0
    fixed_univ = False
    print ('='*50) 

    config = load_config(args.config)
    dataset = BatchGenerator(args.dataset_dir, 'train', config)
    test_dataset = BatchGenerator(args.dataset_dir, 'test', config)
    lr_ph = tf.placeholder(tf.float32) 
    args.dataset_name = '+'.join(config['TRAIN_DATASET'])
    hw = 32 if 'cifar' in args.dataset_name else 84
    trainnet = AdapterNet(args.model_name, dataset.n_classes, 
            isTr=True, config=args.config, depth=20, n_adapt=2, 
            use_adapt=use_adapt, fixed_univ=fixed_univ)
    testnet = AdapterNet(args.model_name, dataset.n_classes, reuse=True,
            isTr=False, config=args.config, depth=20, n_adapt=2, 
            use_adapt=use_adapt)
    opt = tf.train.MomentumOptimizer(lr_ph, 0.9)

    update_var_list = trainnet.var_list
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_op = opt.minimize(trainnet.outputs['loss'], var_list=update_var_list) 
    saver = tf.train.Saver()
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    if args.pretrained:
        loc = os.path.join(args.model_dir,
                args.model_name, 
                args.dataset_name + '.ckpt')
        saver.restore(sess, loc)
    
    if args.train:
        max_iter = dataset.dataset_size[args.dataset_name] * args.max_epoch \
                // (args.batch_size)
        show_step = args.show_epoch * max_iter // args.max_epoch
        save_step = args.save_epoch * max_iter // args.max_epoch
        avger = np.zeros([4])
        for i in range(1, max_iter+1): 
            stt = time.time()
            cur_epoch = i * args.batch_size // dataset.dataset_size[args.dataset_name]
            #lr = args.lr if i < 0.7 * max_iter else args.lr*.1
            lr = lr_scheduler(cur_epoch)

            x, y = dataset.get_batch(args.batch_size, 'train', aug=False)
            fd = {trainnet.inputs['x']: x, trainnet.inputs['y']: y, lr_ph: lr}
            runs = [trainnet.outputs['acc'], trainnet.outputs['loss'], train_op]
            p1, p2, _ = sess.run(runs, fd)
            #gvs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            #pdb.set_trace()


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
                validate(testnet, test_dataset)
                avger[:] = 0

            if i % save_step == 0 and i != 0: 
                out_loc = os.path.join(args.model_dir, # models/
                        args.model_name, # bline/
                        args.dataset_name + '.ckpt')  # cifar100.ckpt
                print ('saved at : {}'.format(out_loc))
                saver.save(sess, out_loc)
    else: # if test only
        validate(testnet, test_dataset)
