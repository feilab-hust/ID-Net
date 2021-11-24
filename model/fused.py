import tensorflow as tf
import numpy as np

from .custom import conv3d


def load_ckpt_partial(ckpt_path, net, begin, removed, sess):
    """
    load an npz ckpt file and assign to the 'net'
    Params:
        -removed: layers of which the weights and bias are removed from net, but still saved in npz file
    """
    
    d = np.load( ckpt_path, encoding='latin1')
    params = d['params']    

    ops = []
    i = begin
    for idx, param in enumerate(params):
        if idx not in removed:
            print('loading %d : %s' % (idx, str(param.shape)))
            ops.append(net.all_params[i].assign(param)) 
            i += 1
        else:
            print('omitting %d : %s' % (idx, str(param.shape)))
    if sess is not None:
        sess.run(ops)
   

