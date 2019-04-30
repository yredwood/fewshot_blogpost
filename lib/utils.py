import numpy as np
import tensorflow as tf
import pdb


def lr_scheduler(current_epoch, init_lr, epoch_list, decay):
    lr_list = [init_lr*decay**(i) for i in range(len(epoch_list)+1)] 
    lridx = np.sum(np.array(epoch_list) <= current_epoch)
    return lr_list[lridx]

def l2_loss(var_list=None):
    var_list = tf.trainable_variables() \
            if var_list is None else var_list
    return tf.add_n([tf.nn.l2_loss(v) for v in var_list])

def op_restore_possible_vars(loc, print_details=False):
    # function that returns the tf operation of 
    # restoring possible variables: you should check parsing-it's quite 
    # dependent on naming criterian
    print ('restoring from {}'.format(loc))
    model_name = loc.split('/')[-2]
    saved_list = tf.contrib.framework.list_variables(loc) # (name, shape)
    name_dict = {}
    for svname, svshape in saved_list:
        if model_name in svname:
            name_dict[svname.replace(model_name, '')] = svname
            # name_dict[simpler_name] = original_variable_name
            # to handle that the model name changes

    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_op = []
    for v in vs:
        nn = v.name.split('/')
#        if len(nn) <= 2:
#            continue
        vname = '/'+'/'.join(nn[1:]).split(':')[0]
        if vname in name_dict.keys():
            org_var_name = name_dict[vname]
            svar = tf.contrib.framework.load_variable(loc, org_var_name)
            if svar.shape==v.shape:
                assign_op.append(tf.assign(v, svar))
                print ('=={} is loaded successfully'.format(vname))
            else:
                print ('--{} has different shape. Not restored'.format(vname))
        else:
            if print_details:
                print ('--{} is not in the pretrained model. Not restored'\
                        .format(vname))
    return assign_op
