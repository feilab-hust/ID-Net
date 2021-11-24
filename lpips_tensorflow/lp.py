from . import lpips_tf

def lpips(im_pred, im_true):
    distance = lpips_tf.lpips(im_pred, im_true, model='net-lin', net='alex')
    return distance