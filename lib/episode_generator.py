import numpy as np 
import os 
import time 
import pdb
from tflearn.data_augmentation import ImageAugmentation
try: 
    import cPickle as pickle
except:
    import pickle # python3

class EpisodeGenerator(): 
    def __init__(self, data_dir, phase, config):
        if phase.upper() in ['TRAIN', 'VALID', 'TEST']:
            self.dataset_list = config['{}_DATASET'.format(phase.upper())]
        else:
            raise ValueError('select from train/test/val')

        self.data_root = data_dir
        self.dataset = {}
        self.data_all = []
        self.y_all = []
        self.phase = phase
        for i, dname in enumerate(self.dataset_list): 
            load_dir = os.path.join(data_dir, phase,
                    dname + '.npy')
            self.dataset[dname] = np.load(load_dir)
        
        self.hw = config['hw']
        self.aug = ImageAugmentation()
        self.aug.add_random_flip_leftright()
        self.aug.add_random_crop([self.hw,self.hw], padding=self.hw//8)
        
    def get_episode(self, nway, kshot, qsize, 
            dataset_name=None, 
            onehot=True, 
            printname=False, 
            normalize=True,
            aug=False):

        if (dataset_name is None) or (dataset_name=='multiple'):
            dataset_name = self.dataset_list[np.random.randint(len(self.dataset_list))] 
        if printname:
            print (dataset_name)
        dd = self.dataset[dataset_name]
        random_class = np.random.choice(len(dd), size=nway, replace=False)
        support_set_data = []; query_set_data = []
        support_set_label = []; query_set_label = []
        
        for n, rnd_c in enumerate(random_class):
            data = dd[rnd_c]
            rnd_ind = np.random.choice(len(data), size=kshot+qsize, replace=False)
            rnd_data = data[rnd_ind]

            label = np.array([n for _ in range(kshot+qsize)])
            support_set_data += [r for r in rnd_data[:kshot]]
            support_set_label += [l for l in label[:kshot]]

            query_set_data += [r for r in rnd_data[kshot:]] 
            query_set_label += [l for l in label[kshot:]]

        support_set_data = np.reshape(support_set_data, 
                [-1] + list(rnd_data.shape[1:]))
        query_set_data = np.reshape(query_set_data,
                [-1] + list(rnd_data.shape[1:]))
        
        if normalize:
            support_set_data = support_set_data.astype(np.float32) / 255. 
            query_set_data = query_set_data.astype(np.float32) / 255. 

        if onehot:
            s_1hot = np.zeros([nway*kshot, nway])
            s_1hot[np.arange(nway*kshot), support_set_label] = 1
            q_1hot = np.zeros([nway*qsize, nway]) 
            q_1hot[np.arange(nway*qsize), query_set_label] = 1
            support_set_label = s_1hot
            query_set_label = q_1hot

        if aug:
            support_set_data = self.aug.apply(support_set_data)
            query_set_data = self.aug.apply(query_set_data)
        
        return support_set_data, support_set_label, query_set_data, query_set_label


class BatchGenerator():
    def __init__(self, data_dir, phase, config=None):
        if phase.upper() in ['TRAIN', 'VAL', 'TEST']:
            self.dataset_list = config['{}_DATASET'.format(phase.upper())]
        else:
            raise ValueError('select only from train')

        # its not list in classical setting ? 
        self.data_root = data_dir
        self.phase = phase
        for i, dname in enumerate(self.dataset_list): 
            load_dir = os.path.join(data_dir, phase,
                    dname + '.npy')
            self.dataset = np.load(load_dir)

        self.n_classes = len(self.dataset)
        self.hw = config['hw']

        y = []
        for i in range(len(self.dataset)):
            y.append(np.zeros([len(self.dataset[i])])+i)
#        self.x = np.reshape(self.dataset, [-1,self.hw,self.hw,3]) / 255.
#        self.y = np.reshape(y, [-1]) # guess concat is more appropriate
        self.x = np.concatenate(self.dataset, axis=0) / 255.
        self.y = np.concatenate(y, axis=0)

        self.aug = ImageAugmentation()
        self.aug.add_random_flip_leftright()
        self.aug.add_random_crop([self.hw,self.hw], padding=self.hw//8)

    def get_batch(self, batch_size, onehot=True, aug=False):
        rndidx = np.random.choice(len(self.y), size=batch_size, replace=False)
        x = self.x[rndidx]
        y = self.y[rndidx]
        if onehot:
            y1hot = np.zeros([batch_size, self.n_classes])
            y1hot[np.arange(batch_size), y.astype(int)] = 1
            y = y1hot
        if aug:
            x = self.aug.apply(x)
        return x, y

        

if __name__ == '__main__': 
#    epgen = EpisodeGenerator('../../datasets', 'test')
#    st = time.time()
#    dset = 'cifar10'
#    for i in range(10):
#        epgen.get_random_batch(16, onehot=False)
#
#    print ('time consumed for {} : {:.3f}'.format(dset, time.time()-st))
    cfg = {'TRAIN_DATASET': ['miniImagenet'], 'TEST_DATASET': ['miniImagenet']}
    bcgen = BatchGenerator('../data_npy', 'test', cfg)
    x, y = bcgen.get_batch(32, 'train')
    pdb.set_trace()
