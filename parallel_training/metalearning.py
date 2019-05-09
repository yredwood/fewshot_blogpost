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
    np.random.seed(args.parall_n)
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
    trg_class = cluster_config(args.cluster_npy, args.parall_n)

    print ('='*50) 
    print ('args::') 
    for arg in vars(args):
        print ('%15s: %s'%(arg, getattr(args, arg)))
    print ('='*50) 

    #nway = args.nway
    nway = len(trg_class)
    kshot = args.kshot 
    qsize = args.qsize 
    test_kshot = args.kshot if args.test_kshot==0 else args.test_kshot

    if args.meta_arch=='maml':
        Net = MAMLNet
    elif args.meta_arch=='proto':
        Net = ConvNet
    train_net = Net(args.model_name+'_{}'.format(args.parall_n),
            isTr=True, reuse=False, config=config, 
            arch=args.arch, nway=nway)
            #var_list=['{}_{}/trtask_dense/*'.format(args.model_name, args.parall_n)])
            #var_list=['{}/{}/*'.format(args.model_name, 'new_dense')])

    test_net = Net(args.model_name+'_{}'.format(args.parall_n),
            isTr=False, reuse=True, config=config,
            arch=args.arch, nway=nway)

    decay_epoch = [int(float(e)*args.max_epoch) for e in args.epoch_list.split(',')]

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

    if args.train:
        sx, sy, qx, qy = train_gen.get_episode(
                nway, kshot, qsize, aug=False, trg_class=trg_class)
        

        batch_size = 128
        max_iter = nway*kshot*100//batch_size
        show_step = max_iter//10
        avger = np.zeros([4])
        for i in range(1, max_iter+1):
            lr = args.lr if i < max_iter*0.7 else args.lr*0.1
            stt = time.time()
            ridx = np.random.randint(nway*kshot, size=batch_size)
            fd = {train_net.inputs['x']: sx[ridx], # TODO check this out
                    train_net.inputs['y']: sy[ridx],
                    train_net.inputs['lr']: args.lr}
            runs = [train_net.outputs['acc'], train_net.outputs['loss'], 
                    train_net.train_op]
            p1, p2, _ = sess.run(runs, fd)
            avger += [p1, p2, 0, time.time() - stt]

            if i % show_step==0 and i!=0:
                avger /= show_step
                print ('======step : {:5d}/{}========'\
                        .format(i,max_iter))
                
                idxs = np.split(np.arange(nway*qsize), 10)
                tacc = 0
                for idx in idxs:
                    fd = {test_net.inputs['x']: qx[idx],
                            test_net.inputs['y']: qy[idx]}
                    runs = test_net.outputs['acc']
                    tacc += sess.run(runs, fd)
                tacc /= 10

                print ('Training acc: {:.3f}  '
                        '| Training Loss: {:.3f}  '
                        '| Test Acc : {:.3f}  '
                        '| in {:.2f} sec.'\
                        .format(avger[0], 
                            avger[1], 
                            tacc, 
                            avger[3]*show_step))
                avger[:] = 0
    
    # after training
    out_loc = os.path.join(args.model_dir, # models/
            args.model_name+'_{}'.format(args.parall_n), # bline/
            args.config + '.ckpt')  # cifar100.ckpt
    print ('saved at : {}'.format(out_loc))
    saver.save(sess, out_loc)
