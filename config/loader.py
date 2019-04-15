SIZES = {
        'miniImagenet': 64*600,
        'tieredImagenet': 448695,
        'cifar100': 100*500,
        'cifar10': 10*5000,
        'omniglot': 4400*20,
        'cl_miniImagenet': 64*500,
        'cl_tieredImagenet': 413595,
        'cl_cifar10': 10*5000,
        'cl_cifar100': 100*500,
        'cl_omniglot': 1100*16,
        }


def load_config(config_type):
    if config_type=='general':
        # when you want to train from multiple datasets
        config = { 
            'TRAIN_DATASET': ['awa2', 'cifar100', 'omniglot', 'voc2012', 'caltech256'],
            'TEST_DATASET': ['mnist', 'cub200_2011', 'cifar10', 'caltech101', 'miniImagenet'],
            'VALID_DATASET': ['awa2', 'cifar100', 'omniglot', 'caltech256'],
            'hw': 84,
        }
    elif config_type=='cifar':
        config = {
            'TRAIN_DATASET': ['cifar100'],
            'VAL_DATASET': None,
            'TEST_DATASET': ['cifar10'],
            'hw': 32,
            'size': SIZES['cifar100'],
        }
    else:
        # when training from specific datasets
        config = {
            'TRAIN_DATASET': [config_type],
            'VAL_DATASET': [config_type],
            'TEST_DATASET': [config_type],
            'size': SIZES[config_type],
        }
        if 'cifar' in config_type:
            config['hw'] = 32
        elif 'omniglot' in config_type:
            config['hw'] = 28
        else:
            config['hw'] = 84
    return config
