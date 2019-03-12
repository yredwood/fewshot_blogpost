def load_config(config_type):
    if config_type=='general':
        config = { 
            'TRAIN_DATASET': ['awa2', 'cifar100', 'omniglot', 'voc2012', 'caltech256'],
            'TEST_DATASET': ['mnist', 'cub200_2011', 'cifar10', 'caltech101', 'miniImagenet'],
            'VALID_DATASET': ['awa2', 'cifar100', 'omniglot', 'caltech256']
            }
    elif config_type=='miniimg':
        config = {
            'TRAIN_DATASET': ['miniImagenet'],
            'TEST_DATASET': ['miniImagenet'],
            'VALID_DATASET': ['miniImagenet']
            }
    return config
