import imageio
import numpy as np
import tensorlayer as tl
from six import string_types
from scipy.ndimage.interpolation import zoom
import os, warnings
import time
import matplotlib

matplotlib.use('TkAgg')
from utils.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def reverse_norm(im):
    max_ = np.max(im)
    min_ = np.percentile(im, 0.2)
    im = np.clip(im, min_, max_)
    im = (im - min_) / (max_ - min_) * 65535
    return im.astype(np.uint16)


def normalize_percentile(im, low=0.2, high=99.8):
    """Normalize the input 'im' by im = (im - p_low) / (p_high - p_low), where p_low/p_high is the 'low'th/'high'th percentile of the im
    Params:
        -im  : numpy.ndarray
        -low : float, typically 0.2
        -high: float, typically 99.8
    return:
        normalized ndarray of which most of the pixel values are in [0, 1]
    """

    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    eps = 1e-10
    im = (im - min_v) / (max_v - min_v + eps)
    return im


def delete_background_patches(data_order, raw_data, patch_tem, threshold=0.9, percentile=99.9):
    patch_tem_good = {}
    thres = threshold * np.percentile(raw_data, 99.9)
    for i in range(len(patch_tem)):
        max_val = np.percentile(patch_tem[i], percentile)
        if max_val > thres:
            patch_tem_good[str(data_order) + '_' + str(i)] = patch_tem[i]
    return patch_tem_good


def creat_patchs(raw_data, patch_size, over_lap):
    patch_tem = []
    if raw_data.ndim == 3:
        d_raw, h_raw, w_raw = raw_data.shape[0], raw_data.shape[1], raw_data.shape[2]
        d_patch, h_patch, w_patch = patch_size[0], patch_size[1], patch_size[2]
        d_raw >= d_patch or _raise(ValueError(
            'The shape of the image must be greater than or equal to the shape of the block, but are %s and %s' % (
                raw_data.shape, patch_size)))
        size_over = 1 - over_lap
        d_mumber = d_raw // int(d_patch * size_over) - 1
        h_number = h_raw // int(h_patch * size_over) - 1
        w_number = w_raw // int(w_patch * size_over) - 1
        for i in range(d_mumber):
            if i == 0:
                patch1 = raw_data[d_patch * i:d_patch * (i + 1), :]
            else:
                patch1 = raw_data[int(d_patch * i * size_over):int(d_patch * (i * size_over + 1)), :]
            for j in range(h_number):
                if j == 0:
                    patch2 = patch1[:, h_patch * j:h_patch * (j + 1), :]
                else:
                    patch2 = patch1[:, int(h_patch * j * size_over):int(h_patch * (j * size_over + 1)), :]
                for k in range(w_number):
                    if k == 0:
                        patch3 = patch2[:, :, w_patch * k:w_patch * (k + 1)]
                    else:
                        patch3 = patch2[:, :, int(w_patch * k * size_over):int(w_patch * (k * size_over + 1))]
                    patch_tem.append(patch3)
    elif raw_data.ndim == 2:
        h_raw, w_raw = raw_data.shape[0], raw_data.shape[1]
        h_patch, w_patch = patch_size[0], patch_size[1]
        size_over = 1 - over_lap
        h_number = h_raw // int(h_patch * size_over) - 1
        w_number = w_raw // int(w_patch * size_over) - 1

        for j in range(h_number):
            if j == 0:
                patch2 = raw_data[h_patch * j:h_patch * (j + 1), :]
            else:
                patch2 = raw_data[int(h_patch * j * size_over):int(h_patch * (j * size_over + 1)), :]
            for k in range(w_number):
                if k == 0:
                    patch3 = patch2[:, w_patch * k:w_patch * (k + 1)]
                else:
                    patch3 = patch2[:, int(w_patch * k * size_over):int(w_patch * (k * size_over + 1))]
                patch_tem.append(patch3)
    return patch_tem


def creat_patchs_iso(raw_data, patch_size, over_lap):
    patch_tem = []
    for i in range(raw_data.shape[0]):
        h_raw, w_raw = raw_data.shape[1], raw_data.shape[2]
        h_patch, w_patch = patch_size[0], patch_size[1]
        size_over = 1 - over_lap
        h_number = h_raw // int(h_patch * size_over) - 1
        w_number = w_raw // int(w_patch * size_over) - 1
        data_slice = raw_data[i, :]
        for j in range(h_number):
            if j == 0:
                patch2 = data_slice[h_patch * j:h_patch * (j + 1), :]
            else:
                patch2 = data_slice[int(h_patch * j * size_over):int(h_patch * (j * size_over + 1)), :]
            for k in range(w_number):
                if k == 0:
                    patch3 = patch2[:, w_patch * k:w_patch * (k + 1)]
                else:
                    patch3 = patch2[:, int(w_patch * k * size_over):int(w_patch * (k * size_over + 1))]
                patch_tem.append(patch3)
    return patch_tem


def generate_transform_data(
        data,
        subsample,
        psf_,
        crop_threshold=0.2,
):
    psf_ is None or isinstance(psf_, np.ndarray) or _raise(ValueError())
    target = data
    zoom_order = 1
    _subsample = subsample
    from scipy.signal import fftconvolve
    def _scale_down_up(data, sub_factor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            factor = np.ones(data.ndim)
            factor[-1] = sub_factor
            result1 = zoom(data, 1 / factor, order=0)
            result2 = zoom(result1, factor, order=zoom_order)
            return result2

    def _make_divisible_by_subsample(x, size):
        def _split_slice(v):
            return slice(None) if v == 0 else slice(v // 2, -(v - v // 2))

        slices = [slice(None) for _ in x.shape]
        slices[2] = _split_slice(x.shape[2] - size)
        return x[slices]

    def _adjust_subsample(d, s, c):
        """length d, subsample s, tolerated crop loss fraction c"""
        from fractions import Fraction

        def crop_size(n_digits, frac):
            _s = round(s, n_digits)
            _div = frac.denominator
            s_multiple_max = np.floor(d / _s)
            s_multiple = (s_multiple_max // _div) * _div
            # print(n_digits, _s,_div,s_multiple)
            size = s_multiple * _s
            assert np.allclose(size, round(size))
            return size

        def decimals(v, n_digits=None):
            if n_digits is not None:
                v = round(v, n_digits)
            s = str(v)
            assert '.' in s
            decimals = s[1 + s.find('.'):]
            return int(decimals), len(decimals)

        s = float(s)
        dec, n_digits = decimals(s)
        frac = Fraction(dec, 10 ** n_digits)
        # a multiple of s that is also an integer number must be
        # divisible by the denominator of the fraction that represents the decimal points

        # round off decimals points if needed
        while n_digits > 0 and (d - crop_size(n_digits, frac)) / d > c:
            n_digits -= 1
            frac = Fraction(decimals(s, n_digits)[0], 10 ** n_digits)

        size = crop_size(n_digits, frac)
        if size == 0 or (d - size) / d > c:
            raise ValueError("subsample factor %g too large (crop_threshold=%g)" % (s, c))

        return round(s, n_digits), int(round(crop_size(n_digits, frac)))

    if psf_ is not None:
        _psf = psf_.astype(np.float32, copy=False)
        np.min(_psf) >= 0 or _raise(ValueError('psf has negative values.'))
        _psf /= np.sum(_psf)
        _psf = _psf[np.newaxis, :]
        lr = fftconvolve(data, _psf, mode='same')
    else:
        lr = data

    if _subsample != 1:
        subsample, subsample_size = _adjust_subsample(data.shape[2], _subsample, crop_threshold)
        target = _make_divisible_by_subsample(data, subsample_size)
        lr = _make_divisible_by_subsample(lr, subsample_size)
        lr = _scale_down_up(lr, subsample)
        assert lr.shape == target.shape, (lr.shape, target.shape)

    return target, lr


def save_training_data(file, X, Y, Z):
    """Save training data in ``.npz`` format.
    """
    from pathlib import Path
    Path().expanduser()
    isinstance(file, (Path, string_types)) or _raise(ValueError())
    file = Path(file).with_suffix('.npz')
    file.parent.mkdir(parents=True, exist_ok=True)
    X.shape[0] == Y.shape[0] or _raise(ValueError())
    np.savez(str(file), X=X, Y=Y, Z=Z)

def generate_training_data(
        hr_path,
        lr_path,
        patch_size,
        factor=1,
        mr_need=True,
        psf=None,
        z_sub_sample=1.0,
        threshold=0.9,
        poisson_noise=True,
        gauss_sigma=0,
        regx='.*.tif',
        printable=False
):
    psf is None or isinstance(psf, np.ndarray) or _raise(ValueError())
    mr_data = None
    if len(patch_size) == 3:
        hr_list = sorted(tl.files.load_file_list(path=hr_path, regx=regx, printable=printable))
        lr_list = sorted(tl.files.load_file_list(path=lr_path, regx=regx, printable=printable))
        len(hr_list) == len(lr_list) or _raise(
            ValueError(
                'The two sets of data should be equal, but you provide %s and %s' % (len(hr_list), len(lr_list))))
        lr_dic = {}
        hr_dic_good = {}
        hr_patch_size = (factor * np.array(patch_size)).tolist() if patch_size[0] > 1 else (
                factor * np.array(patch_size[1:3])).tolist()
        lr_patch_size = patch_size if patch_size[0] > 1 else patch_size[1:3]
        for i, hr_file in enumerate(hr_list):
            _hr = imageio.volread(os.path.join(hr_path, hr_file))  # img:[depth, height, width]
            if _hr.dtype != np.float32:
                _hr = _hr.astype(np.float32, casting='unsafe')
            _hr = normalize_percentile(_hr, 0.2, 99.99)
            hr_patch = creat_patchs(_hr, hr_patch_size, over_lap=0.5)
            hr_patch_good = delete_background_patches(data_order=i, raw_data=_hr, patch_tem=hr_patch,
                                                      threshold=threshold,
                                                      percentile=99.9)
            hr_dic_good.update(hr_patch_good)
            time.sleep(0.1)
            print("Progressing HR data:   {0}%".format(round((i + 1) * 100 / len(hr_list))),
                  end="\r" if i < len(hr_list) - 1 else "\n")
        for j, lr_file in enumerate(lr_list):
            _lr = imageio.volread(os.path.join(lr_path, lr_file))
            if bool(poisson_noise):
                if j == 0:
                    print("apply poisson noise")
                _lr = np.random.poisson(np.maximum(0, _lr).astype(np.int)).astype(np.float32)
                # lr = normalize_percentile(lr)

            if gauss_sigma > 0:
                if j == 0:
                    print("adding gaussian noise with sigma = ", gauss_sigma)
                noise = np.random.normal(0, gauss_sigma, size=_lr.shape).astype(np.float32)
                _lr = np.maximum(0, _lr + noise)
            if (_lr.dtype != np.float32):
                _lr = _lr.astype(np.float32, casting='unsafe')
            _lr = normalize_percentile(_lr, 0.2, 99.99)
            # _lr = add_speckle_noise(_lr)
            lr_patch = creat_patchs(_lr, lr_patch_size, over_lap=0.5)
            for i in range(len(lr_patch)):
                lr_dic[str(j) + '_' + str(i)] = lr_patch[i]
            time.sleep(0.1)
            print("Progressing LR data:   {0}%".format(round((j + 1) * 100 / len(lr_list))),
                  end="\r" if j < len(lr_list) - 1 else "\n")
        hr_select = []
        lr_select = []

        for i, block in hr_dic_good.items():
            hr_select.append(block)
            lr_select.append(lr_dic[i])
        hr_select_array = np.array(hr_select)
        lr_select_array = np.array(lr_select)

        hr_select_array.shape[0] == lr_select_array.shape[0] or _raise(ValueError(
            'The amount of hr must be equal to that of lr, but are %s and %s' % (
                hr_select_array.shape[0], lr_select_array.shape[0])))
        hr_data = np.expand_dims(hr_select_array, 1)
        lr_data = np.expand_dims(lr_select_array, 1)

        if bool(mr_need) and factor > 1 and len(hr_data.shape) == 5:
            mr_select = []
            factor_scale = np.ones(hr_data[0, :].ndim)
            factor_scale[1], factor_scale[2], factor_scale[3] = 1.0 / factor, 1.0 / factor, 1.0 / factor
            for i in range(hr_data.shape[0]):
                mr = zoom(hr_data[i, :], factor_scale, order=3)
                mr_select.append(mr)
                print("Progressing mr data:   {0}%".format(round((i + 1) * 100 / hr_data.shape[0])),
                      end="\r" if i < hr_data.shape[0] - 1 else "\n")
            mr_data = np.array(mr_select)


    elif isinstance(psf, np.ndarray) and len(patch_size) == 2 or _raise(
            ValueError('Please enter the correct PSF and patch size!')):
        hr_list = sorted(tl.files.load_file_list(path=hr_path, regx=regx, printable=printable))
        lr_dic = {}
        hr_dic_good = {}
        number_hr = 0
        hr_patch_size = patch_size if patch_size[0] > 1 else patch_size[1:3]
        for i, hr_file in enumerate(hr_list):
            hr = imageio.volread(os.path.join(hr_path, hr_file))  # img:[depth, height, width]
            if hr.dtype != np.float32:
                hr = hr.astype(np.float32, casting='unsafe')
            hr, lr = generate_transform_data(hr, z_sub_sample, psf)
            if bool(poisson_noise):
                if i == 0:
                    print("apply poisson noise")
                lr = np.random.poisson(np.maximum(0, lr).astype(np.int)).astype(np.float32)
                # lr = normalize_percentile(lr)

            if gauss_sigma > 0:
                if i == 0:
                    print("adding gaussian noise with sigma = ", gauss_sigma)
                noise = np.random.normal(0, gauss_sigma, size=lr.shape).astype(np.float32)
                lr = np.maximum(0, lr + noise)
            hr_patch = creat_patchs_iso(hr, hr_patch_size, over_lap=0.5)
            hr_patch_good = delete_background_patches(data_order=number_hr, raw_data=hr, patch_tem=hr_patch,
                                                      threshold=0.9,
                                                      percentile=99.9)
            lr_patch = creat_patchs_iso(lr, hr_patch_size, over_lap=0.5)
            for j in range(len(lr_patch)):
                lr_dic[str(number_hr) + '_' + str(j)] = lr_patch[j]
            number_hr += 1
            hr_dic_good.update(hr_patch_good)
            print("Progressing ISO data:   {0}%".format(round((i + 1) * 100 / len(hr_list))),
                  end="\r" if i < len(hr_list) - 1 else "\n")

        hr_select = []
        lr_select = []
        for i, block in hr_dic_good.items():
            hr_select.append(block)
            lr_select.append(lr_dic[i])
        hr_select_array = np.array(hr_select)
        lr_select_array = np.array(lr_select)

        hr_data = np.expand_dims(hr_select_array, 1)
        lr_data = np.expand_dims(lr_select_array, 1)
        hr_data = normalize_percentile(hr_data, 0.2, 99.99)
        lr_data = normalize_percentile(lr_data, 0.2, 99.99)

    file = 'data/training_data.npz'
    print('%s groups of training data has been generated' % hr_data.shape[0])
    save_training_data(file, hr_data, lr_data, mr_data)

    return hr_data, lr_data
