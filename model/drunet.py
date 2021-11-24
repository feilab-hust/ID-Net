import tensorflow as tf
import tensorlayer as tl

from .custom import conv2d, conv3d, concat, LReluLayer, ReluLayer, SubVoxelConv, max_pool3d, max_pool2d
from tensorlayer.layers import Layer, InputLayer, ElementwiseLayer

__all__ = ['drunet3d', 'drunet2d']


def residual_block_3d(input, out_channels, act=tf.nn.relu, name='residual_block'):
    with tf.variable_scope(name):
        x = conv3d(input, out_channels=out_channels, filter_size=3, act=act, name='conv1')
        x = conv3d(x, out_channels=out_channels, filter_size=3, act=tf.identity, name='conv2')
        x = ElementwiseLayer(layer=[x, input], combine_fn=tf.add, name='merge')
        x.outputs = tf.nn.relu(x.outputs)
        return x

def residual_block_2d(input, out_channels, act=tf.nn.relu, name='residual_block'):
    with tf.variable_scope(name):
        x = conv2d(input, out_channels=out_channels, filter_size=3, act=act, name='conv1')
        x = conv2d(x, out_channels=out_channels, filter_size=3, act=tf.identity, name='conv2')
        x = ElementwiseLayer(layer=[x, input], combine_fn=tf.add, name='merge')
        x.outputs = tf.nn.relu(x.outputs)
        return x

def block_layer_3d(input, out_channels, n_blocks, name='block_layer'):
    with tf.variable_scope(name):
        # x = conv3d(input, out_channels=filters * 4, filter_size=1, act=act, name='conv1')
        x = residual_block_3d(input, out_channels=out_channels, name='residual_block1')
        for n in range(1, n_blocks):
            x = residual_block_3d(x, out_channels=out_channels, name='btlnk_blk%d' % (n + 1))
        return x

def block_layer_2d(input, out_channels, n_blocks, name='block_layer'):
    with tf.variable_scope(name):
        # x = conv3d(input, out_channels=filters * 4, filter_size=1, act=act, name='conv1')
        x = residual_block_2d(input, out_channels=out_channels, name='residual_block1')
        for n in range(1, n_blocks):
            x = residual_block_2d(x, out_channels=out_channels, name='btlnk_blk%d' % (n + 1))
        return x

def upconv3d(layer, out_channels, factor=2, mode='subpixel', output_shape=None, act=tf.identity, name='upconv'):
    with tf.variable_scope(name):
        if mode == 'subpixel':
            n = conv3d(layer, out_channels * (factor ** 3), 1, 1, act=tf.identity)
            n = SubVoxelConv(n, scale=factor, n_out_channel=None, act=act, name=mode)
            return n

        elif mode == 'deconv':
            batch, depth, height, width, in_channels = layer.outputs.shape.as_list()

            if output_shape is None:
                output_shape = (batch, depth * factor, height * factor, width * factor, out_channels)
            else:
                if len(output_shape) == 3:
                    output_shape = [batch] + output_shape + [out_channels]

            n = tl.layers.DeConv3dLayer(layer, act=act,
                                        shape=(1, 1, 1, out_channels, in_channels),
                                        output_shape=output_shape,
                                        strides=(1, factor, factor, factor, 1), padding='SAME',
                                        W_init=tf.truncated_normal_initializer(stddev=0.02),
                                        b_init=tf.constant_initializer(value=0.0), name=mode)

            return n

        else:
            raise Exception('unknown mode : %s' % mode)

def upconv2d(layer, out_channels, factor=2, mode='subpixel', output_shape=None, act=tf.identity, name='upconv'):
    with tf.variable_scope(name):
        if mode == 'subpixel':
            n = conv2d(layer, out_channels * (factor ** 2), 1, 1, act=tf.identity)
            n = SubVoxelConv(n, scale=factor, n_out_channel=None, act=act, name=mode)
            return n

        elif mode == 'deconv':
            batch, height, width, in_channels = layer.outputs.shape.as_list()

            if output_shape is None:
                output_shape = (batch, height * factor, width * factor, out_channels)
            else:
                if len(output_shape) == 2:
                    output_shape = [batch] + output_shape + [out_channels]

            n = tl.layers.DeConv2dLayer(layer, act=act,
                                        shape=(1, 1, out_channels, in_channels),
                                        output_shape=output_shape,
                                        strides=(1, factor, factor, 1), padding='SAME',
                                        W_init=tf.truncated_normal_initializer(stddev=0.02),
                                        b_init=tf.constant_initializer(value=0.0), name=mode)

            return n

        else:
            raise Exception('unknown mode : %s' % mode)

def drunet3d(LR, G0=64, upscale=False, reuse=False, is_train=False, name='drunet3d'):
    '''
    Params:
        LR - [batch, depth, height, width, channels]
    '''
    conv_kernel = 3
    block_num = 3
    act = tf.nn.leaky_relu
    layers = []
    with tf.variable_scope(name, reuse=reuse):
        n = tl.layers.InputLayer(LR, name='lr_input')
        n = conv3d(n, G0, conv_kernel, 1, act=act, name='conv0')
        n = block_layer_3d(n, G0, block_num, name='block_layer_input')
        layers.append(n)

        layer_specs = [
            G0 * 2,
            G0 * 4,
            # G0 * 8
        ]

        for out_channels in layer_specs:
            with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
                # rect = LReluLayer(layers[-1], alpha=0.2)
                conv = conv3d(layers[-1], out_channels, conv_kernel, 1, act=act)
                pool = max_pool3d(conv, filter_size=conv_kernel, stride=2, padding='SAME', name='maxpool')
                pool = block_layer_3d(pool, out_channels, block_num, name='block_layer_unet')
                layers.append(pool)
                print(pool.outputs.shape)

        layer_specs.reverse()
        layer_specs = [l // 2 for l in layer_specs]

        encoder_layers_num = len(layers)
        for decoder_layer, out_channels in enumerate(layer_specs):
            skip_layer = encoder_layers_num - decoder_layer - 1
            _, d, h, w, _ = layers[skip_layer - 1].outputs.shape.as_list()
            print([d, h, w])
            with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = layers[-1]
                else:
                    input = concat([layers[-1], layers[skip_layer]])
                rect = ReluLayer(input)
                out = upconv3d(rect, out_channels, mode='deconv', output_shape=[d, h, w])
                # out = batch_norm(out, is_train=is_train)
                layers.append(out)

        with tf.variable_scope('out'):
            n = concat([layers[-1], layers[0]])
            n = block_layer_3d(n, 2*G0, block_num, name='block_layer_output')
            if upscale is False:
                output = conv3d(n, 1, conv_kernel, 1, act=tf.identity, name='conv')
            else:
                output = upconv3d(n, out_channels=8, name='upsampling1')
                output = upconv3d(output, out_channels=1, act=tf.identity, name='upsampling2')

            return output

def drunet2d(LR, G0=32, upscale=False, reuse=False, is_train=False, name='drunet3d'):
    '''
    Params:
        LR - [batch, depth, height, width, channels]
    '''
    conv_kernel = 5
    block_num = 3
    act = tf.nn.relu
    layers = []
    with tf.variable_scope(name, reuse=reuse):
        n = tl.layers.InputLayer(LR, name='lr_input')
        n = conv2d(n, G0, conv_kernel, 1, act=act, name='conv0')
        n = block_layer_2d(n, G0, block_num, name='block_layer_input')
        layers.append(n)

        layer_specs = [
            G0 * 2,
            G0 * 4,
            # G0 * 8
        ]

        for out_channels in layer_specs:
            with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
                # rect = LReluLayer(layers[-1], alpha=0.2)
                conv = conv2d(layers[-1], out_channels, conv_kernel, 1, act=act)
                pool = max_pool2d(conv, filter_size=conv_kernel, stride=2, padding='SAME', name='maxpool')
                pool = block_layer_2d(pool, out_channels, block_num, name='block_layer_unet')
                layers.append(pool)
                print(pool.outputs.shape)

        layer_specs.reverse()
        layer_specs = [l // 2 for l in layer_specs]

        encoder_layers_num = len(layers)
        for decoder_layer, out_channels in enumerate(layer_specs):
            skip_layer = encoder_layers_num - decoder_layer - 1
            _, h, w, _ = layers[skip_layer - 1].outputs.shape.as_list()
            print([h, w])
            with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = layers[-1]
                else:
                    input = concat([layers[-1], layers[skip_layer]])
                rect = ReluLayer(input)
                out = upconv2d(rect, out_channels, mode='subpixel', output_shape=[h, w])
                # out = upconv2d(rect, out_channels, mode='deconv', output_shape=[h, w])
                # out = batch_norm(out, is_train=is_train)
                layers.append(out)

        with tf.variable_scope('out'):
            n = concat([layers[-1], layers[0]], name='concat_layer')
            n = block_layer_2d(n, 2*G0, block_num, name='block_layer_output')
            if upscale is False:
                output = conv2d(n, 1, conv_kernel, 1, act=tf.identity, name='conv')
            else:
                output = upconv2d(n, out_channels=8, name='upsampling1')
                output = upconv2d(output, out_channels=1, act=tf.identity, name='upsampling2')

            return output