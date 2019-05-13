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
from lib.utils import cluster_config
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
    parser.add_argument('--meta_arch', dest='meta_arch', default='proto')
    parser.add_argument('--mbsize', dest='mbsize', default=4, type=int)
    parser.add_argument('--epl', dest='epoch_list', default='0.5,0.8')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float)
    parser.add_argument('--aug', dest='aug', default=0, type=int)
    parser.add_argument('--pn', dest='parall_n', default=0, type=int)
    parser.add_argument('--cl', dest='cluster_npy', default='')
    args = parser.parse_args()
    return args

def cos_sim(pred, support, nway, hdim):
    # pred: (q, d)
    # sup: (n*k, d)
    proto = np.reshape(support, [nway,-1,hdim])
    proto = np.mean(proto, axis=1)

    proto = np.expand_dims(proto, 0)
    pred = np.expand_dims(pred, 1)
    

    sim = np.sum(proto*pred, axis=2) \
            / np.linalg.norm(proto, axis=2) / np.linalg.norm(pred, axis=2)
    return sim

def protonet(pred, support, nway, hdim):
    proto = np.reshape(support, [nway,-1,hdim])
    proto = np.mean(proto, axis=1)
    
    proto = np.expand_dims(proto, 0)
    pred = np.expand_dims(pred, 1)
    emb = (proto-pred)**2
    dist = np.mean(emb, axis=2)
    return -dist


if __name__=='__main__': 
    args = parse_args() 
    np.random.seed(0)
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
    trg_classes = cluster_config(args.cluster_npy)

    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    nway = args.nway
    kshot = args.kshot 
    qsize = args.qsize 
    test_kshot = args.kshot if args.test_kshot==0 else args.test_kshot

    input_dict = {'x': tf.placeholder(tf.float32, [None,config['hw'],config['hw'],3]),
            'y': tf.placeholder(tf.float32, [None,None])}
       
    num_parallel = args.parall_n
    net = ConvNet(args.model_name+'_{}'.format(num_parallel),
            isTr=False, reuse=False, config=config, arch=args.arch, nway=len(trg_classes[num_parallel]),
            input_dict=input_dict)
    feats = net.outputs['pred']
    tnvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            scope=args.model_name+'_{}/*'.format(num_parallel))
    loader = tf.train.Saver(tnvars)
    
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.global_variables_initializer())
    
    if not args.train:
        loc = os.path.join(args.model_dir,
                args.model_name+'_{}'.format(num_parallel),
                args.config + '.ckpt')
        print ('restored from {}'.format(loc))
        loader.restore(sess, loc)
    
    test_gen = EpisodeGenerator(args.dataset_dir, 'val', config)

    
    accs = []
    for _ in range(args.val_iter):
        sx, sy, qx, qy = test_gen.get_episode(args.nway, test_kshot, 15)
        fd = {input_dict['x']: sx}
        supports_list = sess.run(feats, fd)
        fd = {input_dict['x']: qx}
        preds_list = sess.run(feats, fd)
        
        
        sim_mat = cos_sim(preds_list, supports_list, args.nway, preds_list.shape[-1])
        acc = np.mean(np.argmax(sim_mat, 1) == np.argmax(qy, 1))
        accs.append(acc)
    np.save('clusters/task_features/f{}.npy'.format(num_parallel), np.array(accs))
    print ('saved f{}.npy'.format(num_parallel))
