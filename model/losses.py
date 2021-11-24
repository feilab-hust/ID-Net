import tensorflow as tf
import numpy as np
from lpips_tensorflow import lp
from keras import backend as K

def min_max(x, eps=1e-7):
    max_ = tf.reduce_max(x)
    min_ = tf.reduce_min(x)
    return (x - min_) / (max_ - min_ + eps)

def sobel_edges(input):
    '''
    find the edges of the input image, using the bulit-in tf function

    Params:
        -input : tensor of shape [batch, depth, height, width, channels]
    return:
        -tensor of the edges: [batch, height, width, depth]
    '''
    # transpose the image shape into [batch, h, w, d] to meet the requirement of tf.image.sobel_edges
    img = tf.squeeze(tf.transpose(input, perm=[0, 2, 3, 1, 4]), axis=-1)

    # the last dim holds the dx and dy results respectively
    edges_xy = tf.image.sobel_edges(img)
    # edges = tf.sqrt(tf.reduce_sum(tf.square(edges_xy), axis=-1))

    return edges_xy


def sobel_edges2(input):
    '''
    custom sobel operator for edges detection
    Params
        - input : tensor of shape [batch, depth, height, width, channels=1]
    '''
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_x = tf.constant(filter_x, dtype=tf.float32, name='sobel_x')
    filter_y = tf.constant(filter_y, dtype=tf.float32, name='sobel_y')

    filter_x = tf.reshape(filter_x, [3, 3, 1, 1])
    filter_y = tf.reshape(filter_y, [3, 3, 1, 1])

    batch, depth, height, width, _ = input.shape.as_list()

    with tf.name_scope('sobel'):

        for d in range(0, depth):
            edges_x = tf.nn.conv2d(input[:, d, :, :, :], filter_x, strides=(1, 1, 1, 1), padding='SAME', name='edge_x')
            edges_y = tf.nn.conv2d(input[:, d, :, :, :], filter_y, strides=(1, 1, 1, 1), padding='SAME', name='edge_y')
            edges = tf.sqrt(tf.square(edges_x) + tf.square(edges_y))

            edges = tf.expand_dims(edges, axis=1)
            if d == 0:
                stack = edges
            else:
                stack = tf.concat([stack, edges], axis=1)
            '''
            edges_x_t = tf.nn.conv2d(input[:,d,:,:,:], filter_x, strides=(1,1,1,1), padding='SAME', name='edge_x')
            edges_y_t = tf.nn.conv2d(input[:,d,:,:,:], filter_y, strides=(1,1,1,1), padding='SAME', name='edge_y')

            edges_x = tf.expand_dims(edges_x_t, axis=1) 
            edges_y = tf.expand_dims(edges_y_t, axis=1) 
            if d == 0:
                stack_x = edges_x
                stack_y = edges_y
            else : 
                stack_x = tf.concat([stack_x, edges_x], axis=1)
                stack_y = tf.concat([stack_y, edges_y], axis=1)
            stack = tf.sqrt(tf.square(stack_x) + tf.square(stack_y))
            '''
        return stack

def lpips_loss(y_true, y_pred):
    shape = y_true.get_shape().as_list()
    batch_size, z_depth, W, H, C = shape
    # ups_times = 1

    def pre_prosses(y):
        y1 = K.permute_dimensions(y, [0, 3, 2, 1])
        # y2 = tf.log(y1 + 1)
        y3 = y1 / (K.mean(y1) + 1e-5)
        return y3

    # def up_sample(y):
    #     k = K.ones([ups_times, ups_times, 1, 1])
    #     y1 = K.conv2d_transpose(y, kernel=k, output_shape=[batch_size, ups_times*W, ups_times*H, 1],
    #                                 strides=(ups_times, ups_times), padding='valid')
    #     return y1

    # def get_losses(y_true, y_pred):
    #     for i in range(z_depth):
    #         y1 = K.tile(up_sample(y_true[..., i:i + 1]), [1, 1, 1, 3])
    #         y2 = K.tile(up_sample(y_pred[..., i:i + 1]), [1, 1, 1, 3])
    #         temp = lp.lpips(y1, y2)
    #         if i == 0:
    #             loss_lp = temp
    #         else:
    #             loss_lp = temp + loss_lp
    #     return loss_lp / z_depth
    def get_losses(y_true, y_pred):
        z_c = z_depth//3
        for i in range(z_c):
            y1 = y_true[..., 3*i:3*(i + 1)]
            y2 = y_pred[..., 3*i:3*(i + 1)]
            temp = lp.lpips(y1, y2)
            if i == 0:
                loss_lp = temp
            else:
                loss_lp = temp + loss_lp
        loss_sum = loss_lp / z_c

        if z_depth%3 != 0:
            y1_add = y_true[..., z_depth-3 :z_depth]
            y2_add = y_pred[..., z_depth-3 :z_depth]
            temp_add = lp.lpips(y1_add, y2_add)
            loss_sum = (temp_add + loss_lp)/(z_c+1)
        return loss_sum

    def lpips(y_true, y_pred):
        n = K.shape(y_true)[-1]
        y1 = pre_prosses(K.mean(y_true, axis=4))
        y2 = pre_prosses(K.mean(y_pred[..., :n], axis=4))
        return K.mean(get_losses(y1, y2))

    return lpips(y_true, y_pred)

def l2_loss(image, reference):
    with tf.variable_scope('l2_loss'):
        return tf.reduce_mean(tf.squared_difference(image, reference))


def l1_loss(image, reference):
    with tf.variable_scope('l1_loss'):
        return tf.reduce_mean(tf.abs(image - reference))

def edges_loss(image, reference):
    '''
    params:
        -image : tensor of shape [batch, depth, height, width, channels], the output of DVSR
        -reference : same shape as the image
    '''
    with tf.variable_scope('edges_loss'):
        edges_sr = sobel_edges(image)
        edges_hr = sobel_edges(reference)

        # return tf.reduce_mean(tf.abs(edges_sr - edges_hr))
        return l2_loss(edges_sr, edges_hr)


def img_gradient_loss(image, reference):
    '''
    params:
        -image : tensor of shape [batch, depth, height, width, channels]
        -reference : same shape as the image
    '''
    with tf.variable_scope('gradient_loss'):
        img = tf.squeeze(tf.transpose(image, perm=[0, 2, 3, 1, 4]), axis=-1)
        ref = tf.squeeze(tf.transpose(reference, perm=[0, 2, 3, 1, 4]), axis=-1)
        grad_i = tf.image.image_gradients(img)
        grad_r = tf.image.image_gradients(ref)
        g_loss = tf.reduce_mean(tf.squared_difference(grad_i, grad_r))
        return g_loss


def mean_squared_error(target, output, is_mean=False, name="mean_squared_error"):
    """ Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : 2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, w, h] or [batch_size, w, h, c].
    target : 2D, 3D or 4D tensor.
    is_mean : boolean, if True, use ``tf.reduce_mean`` to compute the loss of one data, otherwise, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """
    output = tf.cast(output, tf.float32)
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))

        elif output.get_shape().ndims == 5:  # [batch_size, depth, height, width, channels]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3, 4]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3, 4]))
        else:
            raise Exception("Unknow dimension")
        return mse


def cross_entropy(labels, probs):
    return tf.reduce_mean(
        tf.sigmoid(labels) * tf.log_sigmoid(probs) + tf.sigmoid(1 - labels) * tf.log_sigmoid(1 - probs))
