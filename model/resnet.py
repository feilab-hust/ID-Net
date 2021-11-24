import tensorflow as tf
import tensorlayer as tl

from .custom import conv3d, conv3d_transpose, batch_norm, concat, upscale
from tensorlayer.layers import InputLayer, ElementwiseLayer

__all__ = ['resnet']

def bottleneck_block(input, filters, project_shortcut=False, act=tf.nn.relu, name='btlnk_blk'):
    """
    channels of the 'input' must be equal to filters * 4 if 'project_shortcut' is False (assured in block_layer where only the 1st block uses 'project_shortcut=True')
    The n_channels of outputs is equal to that of the input if 'project_shortcut=False' 
    """
    filters_out = filters * 4 

    with tf.variable_scope(name):
        shortcut = conv3d(input, out_channels=filters_out, filter_size=1, stride=1, name='shortcut') if project_shortcut else input

        x = conv3d(input, out_channels=filters, filter_size=1, act=act, name='conv1')
        x = conv3d(x, out_channels=filters, filter_size=3, act=act, name='conv2')
        x = conv3d(x, out_channels=filters_out, filter_size=1, act=tf.identity, name='conv3')

        x = ElementwiseLayer(layer=[x, shortcut], combine_fn=tf.add, name ='merge')
        x.outputs = tf.nn.relu(x.outputs)
        
        return x

def block_layer(input, filters, n_blocks, name='block_layer'):
    
    with tf.variable_scope(name):
        #x = conv3d(input, out_channels=filters * 4, filter_size=1, act=act, name='conv1')
        x = bottleneck_block(input, filters=filters, project_shortcut=True, name='btlnk_blk1')

        for n in range(1, n_blocks):
            x = bottleneck_block(x, filters=filters, project_shortcut=False,  name='btlnk_blk%d' % (n + 1))

        return x

def resnet(input, factor=4, filters=16, kernel_size=5, block_sizes=[2,2,2,2], reuse=False, name='resnet'):
    """
    choices for block_sizes 
        18 : [2,2,2,2]
        34 and 50 :  [3,4,6,3]
        101: [3, 4, 23, 3]
    """
    with tf.variable_scope(name, reuse=reuse):
        net = InputLayer(input, name='input')
        net = conv3d(net, out_channels=filters, filter_size=kernel_size, act=tf.nn.relu, name='init_conv')

        for i, n_blocks in enumerate(block_sizes):
            n_filters = filters * (2**i)
            net = block_layer(net, filters=n_filters, n_blocks=n_blocks, name='blocklayer%d' % (i + 1))
        
        with tf.variable_scope('upsampling'):
            if factor == 4:
                net = conv3d(net, out_channels=64, filter_size=1, name='conv')
                net = upscale(net, scale=2, name='upscale1')
                net = upscale(net, scale=2, name='upscale2')
            elif factor == 3:
                net = conv3d(net, out_channels=27, filter_size=1, name='conv')
                net = upscale(net, scale=3, name='upscale1')
            elif factor == 2:
                net = conv3d(net, out_channels=8, filter_size=1, name='conv')
                net = upscale(net, scale=2, name='upscale1')
            
        out = conv3d(net, out_channels=1, filter_size=1, act=tf.tanh, name='out')
        return out

        