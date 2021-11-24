import re
import os
import imageio
import tensorflow as tf
import numpy as np
import random
import platform
import math
import collections
import scipy.io
from six.moves import range, zip, map, reduce, filter

__all__ = ['read_all_images',
    'load_and_assign_ckpt',
    'imread2d',
    'imwrite2d',
    'rearrange3d_fn',
    'write3d',
    'interpolate3d',
    '_raise',
    'get_file_list',
    'load_im',
    'normalize_percentile',
    'normalize_min_max',
    'is_number',
    'exists_or_mkdir',
    'consume',
    'compose',
    'axes_check_and_normalize',
    'axes_dict',
    'move_image_axes',
    'choice'
    ]

def _raise(e):
    raise e

def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def consume(iterator):
    collections.deque(iterator, maxlen=0)

def choice(population, k=1, replace=True):
    ver = platform.sys.version_info
    if replace and (ver.major,ver.minor) in [(2,7),(3,5)]: # python 2.7 or 3.5
        # slow if population is large and not a np.ndarray
        return list(np.random.choice(population, k, replace=replace))
    else:
        try:
            # save state of 'random' and set seed using 'np.random'
            state = random.getstate()
            random.seed(np.random.randint(np.iinfo(int).min, np.iinfo(int).max))
            if replace:
                # sample with replacement
                return random.choices(population, k=k)
            else:
                # sample without replacement
                return random.sample(population, k=k)
        finally:
            # restore state of 'random'
            random.setstate(state)

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None.'))
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes

def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[tuple(slices)]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a

def get_file_list(path, regx):
    import os

    file_list = os.listdir(path)
    file_list = [f for _, f in enumerate(file_list) if re.search(regx, f)]
    return file_list

def load_im(path):
    if re.search('.*.tif', path):
        im = get_tiff_fn(path)
    elif re.search('.*.mat', path):
        im = load_im_from_mat(path)
    else:
        _raise(ValueError('unknown image format : %s' % path))
    return im

def read_all_images(path, z_range, format_out=True, factor=None, transform=None, **kwargs):
    """
    Params:
        - format_out : see function 'transform' for details 
        - factor : useful when format_out == False
    return images in shape [n_images, depth, height, width, channels]
    """
    im_set = []
    img_list = get_file_list(path=path, regx='.*.tif')
    
    if format_out == False:
        z_range = z_range // factor[0]
            
    for img_file in img_list:
        print(path + img_file)
        img = load_im(path + img_file)

        if format_out == False:
            assert factor != None
            img = transform(img, factor=factor, inverse=False) 

        if (img.dtype != np.float32):
            img = img.astype(np.float32, casting='unsafe')
            
        print(img.shape)
        if transform is not None:
            img = transform(img, **kwargs)

        depth = img.shape[0]
        for d in range(0, depth, z_range):
            if d + z_range <= depth:
                im_set.append(img[d:(d+z_range), ...])
    
    if (len(im_set) == 0):
        raise Exception("none of the images have been loaded, please check the config img_size and its real dimension")
    
    print('read %d from %s' % (len(im_set), path)) 
    im_set = np.asarray(im_set)
    print(im_set.shape)
    return im_set

## 
# TODO: is the parameter sess necessary ?
##
def load_and_assign_ckpt(sess, ckpt_file, net): 
    import tensorlayer as tl
    return tl.files.load_and_assign_npz(sess=sess, name=ckpt_file, network=net) 
    
def reformat(output):
    """
    Params : 
        output : tf.Tensor, [batch, depth, height, width, channels=1]
    """ 
    _, _, _, width, channels = output.get_shape().as_list()
    assert channels == 1
    center = width // 2
    sample = output[:,:,:,(center - 1) : (center + 2), :]
    resized_out = tf.image.resize_images(tf.squeeze(sample, [4]), size=[224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    resized_out = (resized_out + 1) / 2  # transform pixel value to [0, 1]
    
    return resized_out

def imread2d(path):
    im = imageio.imread(path)
    return im

def imwrite2d(im, path):
    imageio.imwrite(path, im)

def get_tiff_fn(path):
    """
    return volume in shape of [depth, height, width, channels=1]
    """
    image = imageio.volread(path) # [depth, height, width]
    max_val = 255.
    if image.dtype == np.uint8:
        pass
    elif image.dtype == np.uint16:
        max_val = 65535.
    elif image.dtype == np.float32:
        pass
    else:
        raise Exception('\nunsupported image bitdepth %s\n' % str(image.dtype))
    
    image = image if image.dtype == np.float32 else  normalize_min_max(image, max_val)
    if image.shape[0] == 1:
        # pass
        image = np.transpose(image, axes=[1, 2, 0])
    else:
         image = image[..., np.newaxis]       # [depth, height, width, channels=1]
    return image

def generate_mr_fn(image, mode='ds', **kwargs) :
    """
    Params:
        -image: the source HR image, [depth, height, width, channels=1]
        -mode: 'ds'   -- down-sample only
               'blur' -- blur only
    """ 
    assert mode in ['ds', 'blur']
    depth, height, width, channels = image.shape
    if mode == 'ds':
        factor = kwargs['factor']
        tmp = np.zeros([depth//factor, height//factor, width//factor, channels])
        for i in range(0, depth, factor):
            d = i // factor
            tmp[d, :, :, :] = tmp[d, :, :, :] + image[i, ::factor, ::factor, :]
        tmp = tmp / factor
    else :
        tmp = image
    return tmp


def normalize_percentile(im, low, high):
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

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def rearrange3d_fn(image):
    """
    re-arrange image of shape[depth, height, width, channels] into shape[height, width, depth]
    """
    
    image = np.squeeze(image); # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth]) 
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re    

def _transform(image3d, factor, inverse=False):
    assert len(image3d.shape) == 4
    [d, h, w, c] = image3d.shape

    factors = 1
    for f in factor:
        factors *= f

    if inverse is False:
        assert d % factor[0] == 0 and h % factor[1] == 0 and w % factor[2] == 0

        transformed = np.zeros([d // factor[0], h // factor[1], w // factor[2], c * factors]) 

        for i in range(0, factor[0]):
            for j in range(0, factor[1]):
                for k in range(0, factor[2]):
                    idx = i * (factor[1]*factor[2]) + j * factor[2] + k
                    transformed[..., idx:idx+1] = image3d[i::factor[0], j::factor[1], k::factor[2], :]
    else:
        assert c == factors
        transformed = np.zeros([d*factor[0], h*factor[1], w*factor[2], 1])
        for i in range(0, factor[0]):
            for j in range(0, factor[1]):
                for k in range(0, factor[2]):
                    idx = i * (factor[1]*factor[2]) + j * factor[2] + k
                    transformed[i::factor[0], j::factor[1], k::factor[2], :] = image3d[..., idx:idx+1]

    return transformed


def transform(image3d, factor=[4,4,4], inverse=False):
    '''
    transform a 3D image with 1 channel into a multi-channel one, where the extra channels is filled by the spatial pixels
    Params:
        - image3d : 3D image with shape [depth, height, width, channels=1]
        - factor : super resolution factor for [depth, height, width] respectively
        - inverse : inverse the transform if true
    return:
        - transoformed image with shape[depth/factor[0], height/factor[1], width/factor[2], channels=factor**3]
    '''
    
    if len(image3d.shape) == 4: #[d, h, w, c]
        return _transform(image3d, factor, inverse=inverse)
    if len(image3d.shape) == 5: #[batch, d, h, w, c]
        
        for i in range(0, image3d.shape[0]):
            tmp = image3d[i]
            tmp = _transform(tmp, factor=factor, inverse=inverse)
            tmp = tmp[np.newaxis, ...]
            if i == 0:
                ret = tmp
            else:
                ret = np.concatenate((ret, tmp), axis=0)
    return ret

def _write3d(x, filename, scale_pixel_value=True):
    """
    Params:
        -x : [depth, height, width]
        -max_val : possible maximum pixel value (65535 for 16-bit or 255 for 8-bit)
    """
    min_val = np.min(x)
    max_val = np.max(x)
    if scale_pixel_value:
        x = x - np.min(x)
        x = x / np.max(x)
        #print('min: %.2f max: %.2f\n' % (np.min(x), np.max(x)))
        #x = x + 1.2  #[0, 2]
        x = x * (255. / 2.)
        x = x.astype(np.uint8)

    else:
        x = x.astype(np.float32)

    imageio.volwrite(filename, x)
    #stack = sitk.GetImageFromArray(x)
    #sitk.WriteImage(stack, filename)
    return max_val, min_val

def write3d(x, path, scale_pixel_value=True, savemat=False):
    """
    Params:
        -x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
        -scale_pixel_value : scale pixels value to [0, 65535] if True
        -savemat : if to save x as an extra .mat file.
    """
    
    #print(x.shape)
    new_path = ''
    fragments = path.split('.')
    for i in range(len(fragments) - 1):
        new_path = new_path + fragments[i]

    if savemat:
        save_mat(x, new_path)

    dims = len(x.shape)
    batch = x.shape[0]
    n_channels = x.shape[-1]
    
    min_val = 1e10
    max_val = -1e10
    if dims == 4:
        x_re = np.transpose(x, axes=[0, 1, 2, 3])
        for b in range(batch):
            x_re_b = x_re[b, ...]
            tmp_max, tmp_min = _write3d(x_re_b, new_path + '_{}.{}'.format(b, fragments[-1]) , scale_pixel_value)  
            min_val = min_val if min_val < tmp_min else tmp_min
            max_val = max_val if max_val > tmp_max else tmp_max
            
    elif dims == 5:
        x_re = x
        for b in range(batch):
            x_re_b = x_re[b, ...]
            for c in range(n_channels):
                x_re_c = x_re_b[..., c]
                tmp_max, tmp_min = _write3d(x_re_c, new_path + '_b{}_c{}.{}'.format(b, c, fragments[-1]) , scale_pixel_value) 
                min_val = min_val if min_val < tmp_min else tmp_min   
                max_val = max_val if max_val > tmp_max else tmp_max  
                #print(image.shape)
    else:
        raise Exception('unsupported dims : %s' % str(x.shape))
    
    return max_val, min_val

def load_im_from_mat(filename):
    mat = scipy.io.loadmat(filename)
    for key, val in mat.items():
        if key == 'block_norm':
            block = np.asarray(val)                         # [height, width, depth]
            block = np.transpose(block, axes=[2,0,1])       # [depth, height, width]
            block = block[..., np.newaxis]                  # [depth, height, width, channel==1]
            return block

    return None

def exists_or_mkdir(path, verbose=True):
    """Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True. Copied from TensorLayer.files

    Parameters
    ----------
    path : a string
        A folder path.
    verbose : boolean
        If True, prints results, deaults is True

    Returns
    --------
    True if folder exist, otherwise, returns False and create the folder

    """
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True


def save_mat(im, filename):
    """save the image as .mat file.
    """
    scipy.io.savemat(filename, mdict={'data' : im})

def interpolate3d(img, factor=4, order=1):
    """
    Params:
        -img : [batch, depth, height, width, channels] or [depth, height, width, channels]
    """
    from scipy.ndimage.interpolation import zoom
    if len(img.shape) == 5:
        zoom_factor = [1,factor,factor,factor,1]
    elif len(img.shape) == 4:
        zoom_factor = [factor,factor,factor,1]
    else:
        raise Exception("interpolate3d : unsupported image dims : %d" % len(img.shape))

    return zoom(img, zoom=zoom_factor, order=order)


'''
    def makeGaussianKernel1D(sigma):
    """
    params:
        -sigma : standard deviation of the 1-D Gaussian function, measured in `pixels`
    """
    kernel_radius = (int) (2 * sigma + 1)
    kernel = np.zeros(kernel_radius * 2 - 1)
    k_sum = 0.
    for i in range(0, kernel_radius * 2 - 1):
        x = i - kernel_radius
        kernel[i] = math.exp(-0.5 * x * x / sigma / sigma)
        k_sum += kernel[i]
    kernel = kernel / k_sum # normalization
    return kernel

def conv1d(line, kernel):
    """1-D convolution
    Params:
        -line : 1-D array, a row or colume from a image
        -kernel: 1-D normalized kernel
    return:
        convolved line
    """

    k_radius = kernel.size // 2
    conv = np.zeros(line.size)
    for i in range(0, line.size):
        tmp = 0.
        for j in range(-k_radius, k_radius + 1):
            idx = i + j
            if (idx >= 0 and idx < line.size):
                tmp += line[idx] * kernel[j + k_radius]
        conv[i] = tmp
    return conv

def gaussianBlur2D(image, kernel):
    """
    Perform a 2-D Gaussian blur by x-conv and y-conv seperately. The image channels remain unchanged.
    Params:
        -image: [height, width, channels=1]
        -kernel: 1-D Gaussian kernel
    """
    height, width, channels = image.shape
    blurred = np.zeros([height, width, channels])
    for c in range(0, channels):
        image_c = image[:,:, c]
        image_bc = np.zeros([height, width])
        for h in range(0, height):
            image_bc[h, :] = conv1d(image_c[h, :], kernel)
        for w in range(0, width):
            image_bc[:, w] = conv1d(image_bc[:, w], kernel)
    
        blurred[:,:,c] = image_bc
    
    return blurred

def gaussianBlur3D(image, sigma_xy, sigma_z):
    """ 
    Params:
        -image: [depth, height, width, channels] 
        -sigma: standard deviations of the 3D Gussian function, measured in pixels
    return: the blurred image
    """

    depth, height, width, channels = image.shape
    blurred = np.zeros([depth, height, width, channels])
    kernel_xy = makeGaussianKernel1D(sigma_xy)
    kernel_z = makeGaussianKernel1D(sigma_z)
    ## blur xy
    for d in range(0, depth):
        blurred[d, ...] = gaussianBlur2D(image[d, ...], kernel_xy)    
    ## blur z 
    for h in range(0, height):
        for w in range(0, width):
            for c in range(0, channels):
                blurred[:, h, w, c] = conv1d(blurred[:, h, w, c], kernel_z)
    return blurred
'''