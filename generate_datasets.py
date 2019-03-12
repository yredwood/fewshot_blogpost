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
            default='miniImagenet',
            help='tieredImagenet or miniImagenet or miniImagenet_cy')
    parser.add_argument('--output-path',
            default='./data_npy')
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
        print ('saved in {}'.format(outfile_name))

if args.dataset_name == 'tieredImagenet':
    # tieredImagenet pkl file can be downloaded from
    # https://github.com/renmengye/few-shot-ssl-public
    # and locate it data_root/dataset_name and unzip it
    file_root = os.path.join(args.data_root, args.dataset_name)
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

        dataset_output_path = os.path.join(args.output_path, dsettype)
        if not os.path.exists(dataset_output_path):
            os.makedirs(dataset_output_path)
        outfile_name = os.path.join(dataset_output_path, args.dataset_name + '.npy')
        np.save(outfile_name, np.array(out_data))
        print ('saved in {}'.format(outfile_name))
