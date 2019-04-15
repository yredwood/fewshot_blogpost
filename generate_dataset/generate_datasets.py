import numpy as np
import os
import pickle
import pdb
import cv2
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', 
            default='/home/mike/DataSet/fewshot.dataset/')
    parser.add_argument('--dataset-name',
            default='sdf',
            help='tieredImagenet or miniImagenet or miniImagenet_cy')
    parser.add_argument('--output-path',
            default='../data_npy')
    args = parser.parse_args()
    return args

args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
 
if args.dataset_name == 'miniImagenet':
    # miniImagenet pkl file can be downloaded from
    # https://github.com/renmengye/few-shot-ssl-public
    # and locate it data_root/dataset_name and unzip it
    file_root = os.path.join(args.data_root, args.dataset_name)
    for dsettype in ['train', 'val', 'test']:
        fname = os.path.join(file_root, 'mini-imagenet-cache-{}.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        
        out_data = []
        for key, value in data['class_dict'].items():
            # key: classname
            # value : index of an image
            img = data['image_data'][value]
            out_data.append(img)

        dataset_output_path = os.path.join(args.output_path, dsettype)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
        np.save(outfile_name, np.array(out_data))
        print ('saved in {}'.format(outfile_name), np.shape(out_data))

if args.dataset_name == 'cl_miniImagenet':
    # this is for the standard learning 
    # only training dataset will be split 
    # into train/test
    file_root = os.path.join(args.data_root, 'miniImagenet')
    fname = os.path.join(file_root, 'mini-imagenet-cache-{}.pkl'.format('train'))
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    out_data = []
    for key, value in data['class_dict'].items():
        img = data['image_data'][value]
        out_data.append(img)
    out_data = np.array(out_data)
    # out_data.shape : (64,600,84,84,3)
    train_data = out_data[:,:500]
    test_data = out_data[:,500:]
    
    dataset_output_path = os.path.join(args.output_path, 'train')
    if not os.path.exists(dataset_output_path):
        os.makedirs(dataset_output_path)
    outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
    np.save(outfile_name, train_data)
    print ('saved in {}'.format(outfile_name), np.shape(train_data))

    dataset_output_path = os.path.join(args.output_path, 'test')
    if not os.path.exists(dataset_output_path):
        os.makedirs(dataset_output_path)
    outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
    np.save(outfile_name, test_data)
    print ('saved in {}'.format(outfile_name), np.shape(train_data))

if args.dataset_name == 'tieredImagenet' or args.dataset_name=='cl_tieredImagenet':
    # tieredImagenet pkl file can be downloaded from
    # https://github.com/renmengye/few-shot-ssl-public
    # and locate it data_root/dataset_name and unzip it
    file_root = os.path.join(args.data_root, 'tieredImagenet')
    for dsettype in ['train', 'val', 'test']:
        fname = os.path.join(file_root, '{}_images_png.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        images = np.zeros([len(data),84,84,3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(data), desc='decompress'):
            img = cv2.imdecode(item, 1)
            images[ii] = img
        
        fname = os.path.join(file_root, '{}_labels.pkl'.format(dsettype))
        with open(fname, 'rb') as f:
            label = pickle.load(f, encoding='latin1')

        out_data = []
        labsp = label['label_specific']
        num_classes = np.unique(labsp)
        for i in num_classes:
            out_data.append(images[labsp==i])
        if args.dataset_name=='cl_tieredImagenet':
            tr_data = []
            te_data = []
            for i in num_classes:
                tr_data.append(out_data[i][:-100])
                te_data.append(out_data[i][-100:])

            outpath = os.path.join(args.output_path, 'train')
            outfile = os.path.join(outpath, args.dataset_name + '.npy')
            np.save(outfile, np.array(tr_data))
            print ('saved at ', outfile, np.shape(tr_data))
            outpath = os.path.join(args.output_path, 'test')
            outfile = os.path.join(outpath, args.dataset_name + '.npy') 
            np.save(outfile, np.array(te_data), np.shape(te_data))
            print ('saved at ', outfile)
            exit()

        dataset_output_path = os.path.join(args.output_path, dsettype)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
        np.save(outfile_name, np.array(out_data))
        print ('saved in {}'.format(outfile_name))

if args.dataset_name=='cl_cifar10' or args.dataset_name=='cifar10':
    # dataset for validating structure performance
    # -> so this is not for meta-learning, it's for classical learning
    for dsettype in ['train', 'test']:
        if dsettype=='train':
            data = []
            label = []
            for i in range(5):
                data_dir = os.path.join(args.data_root, 
                        'cifar10/cifar-10-batches-py/data_batch_{}'\
                                .format(i+1))
                f = open(data_dir, 'rb')
                d = pickle.load(f, encoding='bytes')
                data.append(np.array(d[b'data']))
                label.append(np.array(d[b'labels']))
                f.close()

            data = np.reshape(data, [-1,3,32,32])
            data = np.transpose(data, [0,2,3,1])
            label = np.reshape(label, [-1])
        else:
            data_dir = os.path.join(args.data_root, 
                    'cifar10/cifar-10-batches-py/test_batch')
            f = open(data_dir, 'rb')
            d = pickle.load(f, encoding='bytes')
            data = np.array(d[b'data'])
            data = np.reshape(data, [-1,3,32,32])
            data = np.transpose(data, [0,2,3,1])
            label = np.array(d[b'labels'])
            f.close()
                    
        classwise_data = []
        for lb in np.unique(label):
            classwise_data.append(data[label==lb])
            
        if args.dataset_name=='cl_cifar10':
            # we'll not separate the validation set
            dataset_output_path = os.path.join(args.output_path, dsettype)
            if not os.path.exists(dataset_output_path):
                os.makedirs(dataset_output_path)
            outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
            np.save(outfile_name, np.array(classwise_data))
            print ('saved in ', outfile_name, np.shape(classwise_data))
        else:
            dataset_output_path = os.path.join(args.output_path, 'test')
            if not os.path.exists(dataset_output_path):
                os.makedirs(dataset_output_path)
            outfile = os.path.join(dataset_output_path, args.dataset_name + '.npy')
            np.save(outfile, np.array(classwise_data))
            print ('saved in ', outfile, np.shape(classwise_data))
            exit()

if args.dataset_name=='cl_omniglot' or args.dataset_name=='omniglot':
    # this data is generated from Chelsea Finn's MAML code
    root = os.path.join(args.data_root, 'omniglot_resized')
    cfs = [os.path.join(root, family, character) \
    for family in os.listdir(root) \
    if os.path.isdir(os.path.join(root, family)) \
    for character in os.listdir(os.path.join(root, family))]
    
    np.random.seed(1)
    np.random.shuffle(cfs)
    num_val = 100
    num_train = 1200 - num_val
    train_folders = cfs[:num_train]
    val_folders = cfs[num_train:num_train+num_val]
    test_folders = cfs[num_train+num_val:]
    

    def get_imgs(folders):
        out_imgs = []
        for folder in folders:
            for root, _, files in os.walk(folder):
                imgs = [cv2.imread(os.path.join(root, f)) \
                        for f in files]
                out_imgs.append(imgs)
        return out_imgs
    
    tr_data = get_imgs(train_folders)
    val_data = get_imgs(val_folders)
    te_data =get_imgs(test_folders)

    if args.dataset_name=='omniglot':
        tr_data = np.concatenate(\
                [np.rot90(tr_data, i, (2,3)) for i in range(4)])
        val_data = np.concatenate(\
                [np.rot90(val_data, i, (2,3)) for i in range(4)])
        te_data = np.concatenate(\
                [np.rot90(te_data, i, (2,3)) for i in range(4)])

        outname = os.path.join(args.output_path, 'train', args.dataset_name + '.npy')
        np.save(outname, tr_data)
        print ('saved at ', outname, tr_data.shape)

        outname = os.path.join(args.output_path, 'val', args.dataset_name + '.npy')
        np.save(outname, val_data)
        print ('saved at ', outname, val_data.shape)
        
        outname = os.path.join(args.output_path, 'test', args.dataset_name + '.npy')
        np.save(outname, te_data)
        print ('saved at ', outname, te_data.shape)
    
    if args.dataset_name=='cl_omniglot':
        train_data = np.array(tr_data)[:,:16]
        test_data = np.array(tr_data)[:,16:] 
    
        outname = os.path.join(args.output_path, 'train', args.dataset_name + '.npy')
        np.save(outname, train_data)
        print ('saved at ', outname, train_data.shape)

        outname = os.path.join(args.output_path, 'test', args.dataset_name + '.npy')
        np.save(outname, test_data)
        print ('saved at ', outname, test_data.shape)


        
if args.dataset_name=='cl_cifar100' or args.dataset_name=='cifar100':
    # you need to change cifar100 to contain both train and test
    for dsettype in ['train', 'test']:
        data_dir = os.path.join(args.data_root, 'cifar100/cifar-100-python', dsettype)
        with open(data_dir, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='latin1')
        imgs = np.array(dict_data['data'])
        imgs = np.reshape(imgs, (imgs.shape[0],3,32,32))
        imgs = np.transpose(imgs, [0,2,3,1])
        labels = np.array(dict_data['fine_labels'])
        
        cls_data = []
        for lb in np.unique(labels):
            cls_data.append(imgs[labels==lb])
        out_path = os.path.join(args.output_path, dsettype)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, args.dataset_name + '.npy')
        np.save(out_file, np.array(cls_data))
        print ('saved in ', out_file, 'shape: ', np.shape(cls_data))
