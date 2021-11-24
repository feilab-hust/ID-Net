# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorlayer as tl
import numpy as np
import tifffile
import imageio
import os
import re
import time

from scipy.ndimage.interpolation import zoom
from utils import get_file_list, _raise, exists_or_mkdir
from model import DBPN3d, RDN3d, unet3d, Conv_net3d, drunet3d, unet2d, drunet2d
from model.util import Model, Predictor, Model_iso
from configs.config_test import *

valid_lr_img_path = valid_lr_img_path
using_batch_norm = valid_using_batch_norm
device_id = valid_device_id
has_denoise = is_denoise
has_sr = is_sr
has_iso = is_iso
z_sub_factor = z_sub_factor
input_op_name = 'Placeholder'
output_op_name = 'net_s2/out/Tanh'  #


def reverse_norm(im):
    max_ = np.max(im)
    min_ = np.percentile(im, 0.2)
    im = np.clip(im, min_, max_)
    im = (im - min_) / (max_ - min_) * 65535
    return im.astype(np.uint16)


def build_model_and_load_npz(archi1, archi2, factor, conv_kernel, lr_size, epoch, checkpoint_dir, use_cpu=False):
    epoch = 'best' if epoch == 0 else epoch

    # # search for ckpt files
    def _search_for_ckpt_npz(file_dir, tags):
        filelist = os.listdir(checkpoint_dir)
        for filename in filelist:
            if '.npz' in filename:
                if all(tag in filename for tag in tags):
                    return filename
        return None

    if (archi1 is not None):
        resolve_ckpt_file = _search_for_ckpt_npz(checkpoint_dir, ['resolve', str(epoch)])
        interp_ckpt_file = _search_for_ckpt_npz(checkpoint_dir, ['interp', str(epoch)])

        (resolve_ckpt_file is not None and interp_ckpt_file is not None) or _raise(
            Exception('checkpoint file not found'))
    else:
        ckpt_file = _search_for_ckpt_npz(checkpoint_dir, [str(epoch)])
        ckpt_file is not None or _raise(Exception('checkpoint file not found'))

    # ======================================
    # build the model
    # ======================================

    if use_cpu is False:
        device_str = '/gpu:%d' % device_id
    else:
        device_str = '/cpu:0'

    LR = tf.placeholder(tf.float32, lr_size)
    if (archi1 is not None):
        # if ('resolve_first' in archi):        
        with tf.device(device_str):
            if archi1 == 'dbpn3d':
                resolver = DBPN3d(LR, upscale=False, name="net_s1")
            elif archi1 == 'convnet3d':
                resolver = Conv_net3d(LR, name="net_s1")
            else:
                _raise(ValueError())

            if archi2 == 'rdn3d':
                interpolator = RDN3d(resolver.outputs, factor=factor, conv_kernel=conv_kernel, bn=using_batch_norm,
                                     is_train=False, name="net_s2")
                net = interpolator
            else:
                _raise(ValueError())

    else:
        archi = archi2
        with tf.device(device_str):
            if archi == 'rdn3d':
                net = RDN3d(LR, factor=factor, bn=using_batch_norm, conv_kernel=conv_kernel, name="net_s2")
            elif archi == 'unet3d':
                net = unet3d(LR, upscale=False, is_train=False)
            elif archi == 'dbpn3d':
                net = DBPN3d(LR, upscale=True)
            elif archi == 'drunet3d':
                net = drunet3d(LR, reuse=False, name="net_s2")
            elif archi == 'unet2d':
                net = unet2d(LR, reuse=False, name="net_s2")
            elif archi == 'drunet2d':
                net = drunet2d(LR, reuse=False, name="net_s2")
            else:
                raise Exception('unknow architecture: %s' % archi)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if (archi1 is None):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + ckpt_file, network=net)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + resolve_ckpt_file, network=resolver)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + interp_ckpt_file, network=interpolator)
    return sess, net, LR


denoise_img = {}
sr_img = {}

valid_lr_imgs = get_file_list(path=valid_lr_img_path, regx='.*.tif')


def evaluate_denoise(epoch, half_precision_infer=False, use_cpu=False):
    factor = config_test_denoise.factor
    norm_thres = config_test_denoise.norm_thres
    lr_size = config_test_denoise.valid_lr_img_size
    checkpoint_dir = config_test_denoise.valid_ckpt_dir
    save_dir = config_test_denoise.valid_denoise_saving_path
    archi1 = config_test_denoise.archi1
    archi2 = config_test_denoise.archi2
    conv_kernel = config_test_denoise.valid_conv_kernel

    sess, net, LR = build_model_and_load_npz(archi1, archi2, factor, conv_kernel, lr_size, epoch, checkpoint_dir,
                                             use_cpu=False)

    exists_or_mkdir(save_dir)
    model = Model(net, sess, LR)
    block_size = lr_size[1:-1]
    overlap = 0.2

    dtype = np.float16 if half_precision_infer else np.float32
    predictor = Predictor(factor=factor, block_size=block_size, model=model, dtype=dtype)
    for im_idx, im_file in enumerate(valid_lr_imgs):
        for _, thres in enumerate(norm_thres):
            start_time = time.time()
            print('predicting on %s ' % os.path.join(valid_lr_img_path, im_file))
            im = imageio.volread(os.path.join(valid_lr_img_path, im_file))
            sr = predictor.predict(im, block_size, overlap, normalization='auto', low=0.2, high=thres)
            print('time elapsed : %.4f' % (time.time() - start_time))
            denoise_img[str(im_file)] = sr
            tifffile.imsave(os.path.join(save_dir, ('Denoise_thres%s_' % str(thres).replace('.', 'p')) + im_file), sr)
    model.recycle()
    tf.reset_default_graph()


def evaluate_sr(epoch, half_precision_infer=False, use_cpu=False):
    factor = config_test_sr.factor
    norm_thres = config_test_sr.norm_thres
    lr_size = config_test_sr.valid_lr_img_size
    checkpoint_dir = config_test_sr.valid_ckpt_dir
    save_dir = config_test_sr.valid_sr_saving_path
    archi1 = config_test_sr.archi1
    archi2 = config_test_sr.archi2
    conv_kernel = config_test_sr.valid_conv_kernel

    factor = 1 if archi2 is 'unet3d' else factor
    sess, net, LR = build_model_and_load_npz(archi1, archi2, factor, conv_kernel, lr_size, epoch, checkpoint_dir,
                                             use_cpu=False)
    exists_or_mkdir(save_dir)
    model = Model(net, sess, LR)
    block_size = lr_size[1:-1]
    overlap = 0.2
    dtype = np.float16 if half_precision_infer else np.float32

    predictor = Predictor(factor=factor, block_size=block_size, model=model, dtype=dtype)
    start_time = time.time()
    if has_denoise:
        for im_file, im in denoise_img.items():
            for _, thres in enumerate(norm_thres):
                sr = predictor.predict(im, block_size, overlap, normalization='auto', low=0.2, high=thres)
                sr_img[str(im_file)] = sr
                print('time elapsed : %.4f' % (time.time() - start_time))
                tifffile.imsave(os.path.join(save_dir, ('SR_thres%s_' % str(thres).replace('.', 'p')) + im_file), sr)
    else:
        for im_idx, im_file in enumerate(valid_lr_imgs):
            for _, thres in enumerate(norm_thres):
                im = imageio.volread(os.path.join(valid_lr_img_path, im_file))
                print('predicting on %s ' % os.path.join(valid_lr_img_path, im_file))
                sr = predictor.predict(im, block_size, overlap, normalization='auto', low=0.2, high=thres)
                sr_img[str(im_file)] = sr
                print('time elapsed : %.4f' % (time.time() - start_time))
                tifffile.imsave(os.path.join(save_dir, ('SR_thres%s_' % str(thres).replace('.', 'p')) + im_file), sr)
    model.recycle()
    tf.reset_default_graph()


def evaluate_iso(epoch, z_sub_factor=200. / 97., half_precision_infer=False, use_cpu=False):
    factor = config_test_iso.factor
    norm_thres = config_test_iso.norm_thres
    lr_size = config_test_iso.valid_lr_img_size
    checkpoint_dir = config_test_iso.valid_ckpt_dir
    save_dir = config_test_iso.valid_iso_saving_path
    archi1 = config_test_iso.archi1
    archi2 = config_test_iso.archi2
    conv_kernel = config_test_iso.valid_conv_kernel

    sess, net, LR = build_model_and_load_npz(archi1, archi2, factor, conv_kernel, lr_size, epoch, checkpoint_dir,
                                             use_cpu=False)
    exists_or_mkdir(save_dir)
    model = Model_iso(net, sess, LR)
    block_size = lr_size[0:-1]
    overlap = 0.2
    dtype = np.float16 if half_precision_infer else np.float32

    predictor = Predictor(factor=factor, block_size=block_size, model=model, dtype=dtype)
    start_time = time.time()
    if has_sr:
        for im_file, im in sr_img.items():
            for _, thres in enumerate(norm_thres):
                factor = np.ones(im.ndim)
                factor[0] = z_sub_factor
                im = zoom(im, factor, order=0)

                im_h_d = im.transpose(2, 1, 0)
                im_w_d = im.transpose(1, 2, 0)
                sr_1 = predictor.predict_iso(im_h_d, block_size=block_size, overlap=overlap, normalization='auto',
                                             low=0.2, high=thres)
                sr_1 = sr_1.transpose(2, 1, 0)
                sr_2 = predictor.predict_iso(im_w_d, block_size=block_size, overlap=overlap, normalization='auto',
                                             low=0.2, high=thres)
                sr_2 = sr_2.transpose(2, 0, 1)
                # avg = reverse_norm((sr_1 + sr_2) / 2)  # arithmetic mean
                avg = reverse_norm(np.sqrt(np.maximum(sr_1, 0) * np.maximum(sr_2, 0)))  # geometric mean
                print('time elapsed : %.4f' % (time.time() - start_time))
                tifffile.imsave(os.path.join(save_dir, ('Iso_%s_' % str(thres).replace('.', 'p')) + im_file), avg)
    else:
        for im_idx, im_file in enumerate(valid_lr_imgs):
            for _, thres in enumerate(norm_thres):
                im = imageio.volread(os.path.join(valid_lr_img_path, im_file))
                factor = np.ones(im.ndim)
                factor[0] = z_sub_factor
                im = zoom(im, factor, order=0)

                print('predicting on %s ' % os.path.join(valid_lr_img_path, im_file))
                im_d_h = im.transpose(2, 0, 1)
                im_d_w = im.transpose(1, 0, 2)
                sr_1 = predictor.predict_iso(im_d_h, block_size=block_size, overlap=overlap, normalization='auto',
                                             low=0.2, high=thres)
                sr_1 = sr_1.transpose(1, 2, 0)
                sr_2 = predictor.predict_iso(im_d_w, block_size=block_size, overlap=overlap, normalization='auto',
                                             low=0.2, high=thres)
                sr_2 = sr_2.transpose(1, 0, 2)
                # avg = reverse_norm((sr_1 + sr_2) / 2)  # arithmetic mean
                avg = reverse_norm(np.sqrt(np.maximum(sr_1, 0) * np.maximum(sr_2, 0)))  # geometric mean
                print('time elapsed : %.4f' % (time.time() - start_time))
                tifffile.imsave(os.path.join(save_dir, ('Iso_%s_' % str(thres).replace('.', 'p')) + im_file), avg)
    model.recycle()
    tf.reset_default_graph()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=int, default=0)
    # parser.add_argument("-r", "--rdn", help="use one-stage(RDN) net for inference",
    #                     action="store_true") #if the option is specified, assign True to args.rdn. Otherwise False.
    parser.add_argument("--pb", help="load graph from pb file instead of buiding from API", action="store_true")

    parser.add_argument("--cpu", help="use cpu for inference", action="store_true")

    parser.add_argument("--series", help="inputs normalized as a whole sequence", action="store_true")

    parser.add_argument("-f", "--half_precision", help="use half-precision model for inference", action="store_true")

    parser.add_argument("--large", help="predict on large volume", action="store_true")

    parser.add_argument("-p", "--save_pb", help="save the loaded graph as a half-precision pb file",
                        action="store_true")

    parser.add_argument("-l", "--layer", help="save activations of each layer", action="store_true")

    # parser.add_argument("--stage1", help="run stage1 only",
    #                     action="store_true") 
    # parser.add_argument("--stage2", help="run stage2 only",
    #                     action="store_true") 

    args = parser.parse_args()
    if has_denoise:
        evaluate_denoise(epoch=args.ckpt, half_precision_infer=args.half_precision, use_cpu=args.cpu)
    if has_sr:
        evaluate_sr(epoch=args.ckpt, half_precision_infer=args.half_precision, use_cpu=args.cpu)
    if has_iso:
        evaluate_iso(epoch=args.ckpt, z_sub_factor=z_sub_factor, half_precision_infer=args.half_precision, use_cpu=args.cpu)
