import numpy as np 
import os 
import time 
import pdb
try: 
    import cPickle as pickle
except:
    import pickle # python3

DATASET_SIZE = {'awa2': int(37322*0.8), 'mnist': 70000, 'cub200_2011': 11788,
        'omniglot': int(32460*0.8), 'caltech101': 9144, 'caltech256': int(30607*0.8),
        'cifar100': int(60000*0.8), 'cifar10': 60000, 'voc2012': 11540,
        'miniImagenet': int(60000*0.64), 'tieredImagenet': 448695}

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
        self.dataset_size = DATASET_SIZE
        self.phase = phase
        for i, dname in enumerate(self.dataset_list): 
            load_dir = os.path.join(data_dir, phase,
                    dname + '.npy')
            self.dataset[dname] = np.load(load_dir)
        
    def get_episode(self, nway, kshot, qsize, 
            dataset_name=None, 
            onehot=True, 
            printname=False, 
            normalize=True):

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
        
        return support_set_data, support_set_label, query_set_data, query_set_label


class BatchGenerator():
    def __init__(self, data_dir, phase, config=None):
        # phase would be only train 
        if phase.upper() in ['TRAIN', 'VAL', 'TEST']:
            self.dataset_list = config['{}_DATASET'.format(phase.upper())]
        else:
            raise ValueError('select only from train')
        
        self.data_root = data_dir
        self.dataset_size = DATASET_SIZE
        self.phase = phase
        for i, dname in enumerate(self.dataset_list): 
            load_dir = os.path.join(data_dir, phase,
                    dname + '.npy')
            self.dataset = np.load(load_dir)
        
        self.n_classes = len(self.dataset)
        # train/val 500/100 data_split 
        self.xtr = np.reshape(self.dataset[:,:500], [-1,84,84,3])
        self.xval = np.reshape(self.dataset[:,500:], [-1,84,84,3])

        self.ytr = np.zeros([len(self.dataset), 500])
        self.yval = np.zeros([len(self.dataset), 100])
        for i in range(len(self.dataset)):
            self.ytr[i] = i
            self.yval[i] = i
        self.ytr = np.reshape(self.ytr, [-1])
        self.yval = np.reshape(self.yval, [-1])

    def get_batch(self, batch_size, phase, onehot=True):
        xx = self.xtr if phase=='train' else self.xval
        yy = self.ytr if phase=='train' else self.yval
        randidx = np.random.choice(len(xx), size=batch_size, replace=False)
        x = xx[randidx]
        y = yy[randidx]
        if onehot:
            y1hot = np.zeros([batch_size, len(self.dataset)], dtype=int)
            y1hot[np.arange(batch_size),y.astype(int)] = 1
            y = y1hot
        return x, y

    def get_episode(self, nway, kshot, qsize, onehot=True):
        # from dataset, lets split S/Q
        rnd_cls = np.random.choice(self.n_classes, size=nway, replace=False)
        subdataset = self.dataset[rnd_cls] # (nway,600,84,84,3)
        rnd_idx = [np.random.choice(600,kshot+qsize) for _ in range(nway)]
        Dx = [sd[ri] for sd, ri in zip(subdataset, rnd_idx)] # (n,kq,-)
        Dxtr = np.reshape([dx[:kshot] for dx in Dx], [-1,84,84,3])
        Dxte = np.reshape([dx[kshot:] for dx in Dx], [-1,84,84,3])
        
        Dytr = np.reshape([np.zeros([kshot],dtype=int)+i for i in range(nway)], [-1])
        Dyte = np.reshape([np.zeros([qsize],dtype=int)+i for i in range(nway)], [-1])

        if onehot:
            Dytr1hot = np.zeros([len(Dytr), nway])
            Dytr1hot[np.arange(len(Dytr)), Dytr] = 1
            Dytr = Dytr1hot

            Dyte1hot = np.zeros([len(Dyte), nway])
            Dyte1hot[np.arange(len(Dyte)), Dyte] = 1
            Dyte = Dyte1hot
            
        return Dxtr, Dytr, Dxte, Dyte

        

        
        

        
        
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
