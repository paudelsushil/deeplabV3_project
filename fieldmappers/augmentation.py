from skimage import transform as trans
import random
import numpy as np
import numpy.ma as ma

import cv2


def uniShape(img, label, mask, dsize, tlX=0, tlY=0):

    '''
    Unify dimension of images and labels to specified data size

    Params:

    img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
    label (narray): Ground truth with a dimension of (H,W)
    mask (narray): Binary mask represents valid pixels in images and label, in a dimension of (H,W)
    dsize (int): Target data size
    tlX (int): Vertical offset by pixels
    tlY (int): Horizontal offset by pixels

    Returns:

    (narray, narray, narray) tuple of shape unified image, label and mask

    '''

    resizeH, resizeW, c = img.shape

    canvas_img = np.zeros((dsize, dsize, c), dtype=img.dtype)
    canvas_label = np.zeros((dsize, dsize), dtype=label.dtype)
    canvas_mask = np.zeros((dsize, dsize), dtype=label.dtype)

    canvas_img[tlX:tlX + resizeH, tlY:tlY + resizeW] = img
    canvas_label[tlX:tlX + resizeH, tlY:tlY + resizeW] = label
    canvas_mask[tlX:tlX + resizeH, tlY:tlY + resizeW] = mask

    return canvas_img, canvas_label, canvas_mask



def centerRotate(img, label, mask, degree):
    '''
    Synthesize new image chips by rotating the input chip around its center.

    Args:

    img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
    label (narray): Ground truth with a dimension of (H,W)
    mask (narray): Binary mask represents valid pixels in images and label, in a dimension of (H,W)
    degree (tuple or list): Range of degree for rotation

    Returns:

    (narray, narray, narray) tuple of rotated image, label and mask

    '''

    if isinstance(degree, tuple) or isinstance(degree, list):
        degree = random.uniform(degree[0], degree[1])

    # Get the dimensions of the image (e.g. number of rows and columns).
    h, w,_ = img.shape

    # Determine the image center.
    center = (w // 2, h // 2)

    # Grab the rotation matrix
    rotMtrx = cv2.getRotationMatrix2D(center, degree, 1.0)

    # perform the actual rotation for both raw and labeled image.
    img = cv2.warpAffine(img, rotMtrx, (w, h))
    label = cv2.warpAffine(label, rotMtrx, (w, h))
    label = np.rint(label)
    mask = cv2.warpAffine(mask, rotMtrx, (w, h))
    mask = np.rint(mask)

    return img, label, mask



def flip(img, label, mask, ftype):
    '''
    Synthesize new image chips by flipping the input chip around a user defined axis.

    Args:

        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (narray): Ground truth with a dimension of (H,W)
        mask (narray): Binary mask represents valid pixels in images and label, in a dimension of (H,W)
        ftype (str): Flip type from ['vflip','hflip','dflip']

    Returns:

        (narray, narray, narray) tuple of flipped image, label and mask

    Note:

        Provided transformation are:
            1) 'vflip', vertical flip
            2) 'hflip', horizontal flip
            3) 'dflip', diagonal flip

    '''

    def diagonal_flip(img):
        flipped = np.flip(img, 1)
        flipped = np.flip(flipped, 0)
        return flipped


    # Horizontal flip
    if ftype == 'hflip':

        img = np.flip(img, 0)
        label = np.flip(label, 0)
        mask = np.flip(mask, 0)

    # Vertical flip
    elif ftype == 'vflip':

        img = np.flip(img, 1)
        label = np.flip(label, 1)
        mask = np.flip(mask, 1)

    # Diagonal flip
    elif ftype == 'dflip':

        img = diagonal_flip(img)
        label = diagonal_flip(label)
        mask = diagonal_flip(mask)

    else:

        raise ValueError("Bad flip type")

    return img.copy(), label.copy(), mask.copy()



def reScale(img, label, mask, scale=(0.8, 1.2), randResizeCrop=False, diff=False, cenLocate=True):
    '''
    Synthesize new image chips by rescaling the input chip.

    Params:

        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        label (narray): Ground truth with a dimension of (H,W)
        mask (narray): Binary mask represents valid pixels in images and label, in a dimension of (H,W)
        scale (tuple or list): Range of scale ratio
        randResizeCrop (bool): Whether crop the rescaled image chip randomly or at the center if the chip is larger than inpput ones
        diff (bool): Whether change the aspect ratio
        cenLocate (bool): Whether locate the rescaled image chip at the center or a random position if the chip is smaller than input

    Returns:

        (narray, narray, narray) tuple of rescaled image, label and mask

    '''

    h, w, _ = img.shape
    if isinstance(scale, tuple) or isinstance(scale, list):
        resizeH = round(random.uniform(scale[0], scale[1]) * h)
        if diff:
            resizeW = round(random.uniform(scale[0], scale[1]) * w)
        else:
            resizeW = resizeH
    else:
        raise Exception('Wrong scale type!')

    imgRe = trans.resize(img, (resizeH, resizeW), preserve_range=True)
    labelRe = trans.resize(label, (resizeH, resizeW), preserve_range=True)
    maskRe = trans.resize(mask, (resizeH, resizeW), preserve_range=True)


    # crop image when length of side is larger than input ones
    if randResizeCrop:
        x_off = random.randint(0, max(0, resizeH - h))
        y_off = random.randint(0, max(0, resizeW - w))
    else:
        x_off = max(0, (resizeH - h) // 2)
        y_off = max(0, (resizeW - w) // 2)

    imgRe = imgRe[x_off:x_off + min(h, resizeH), y_off:y_off + min(w, resizeW), :]
    labelRe = labelRe[x_off:x_off + min(h, resizeH), y_off:y_off + min(w, resizeW)]
    labelRe = np.rint(labelRe)
    maskRe = maskRe[x_off:x_off + min(h, resizeH), y_off:y_off + min(w, resizeW)]
    maskRe = np.rint(maskRe)

    # locate image when it is smaller than input
    if resizeH < h or resizeW < w:
        if cenLocate:
            tlX = max(0, (h - resizeH) // 2)
            tlY = max(0, (w - resizeW) // 2)
        else:
            tlX = random.randint(0, max(0, h - resizeH))
            tlY = random.randint(0, max(0, w - resizeW))

        # resized result
        imgRe, labelRe, maskRe = uniShape(imgRe, labelRe, maskRe, h, tlX, tlY)

    return imgRe, labelRe, maskRe


def shiftBrightness(img, gammaRange=(0.2, 2.0), shiftSubset=(4, 4), patchShift=True):
    '''
    Shift image brightness through gamma correction

    Params:

        img (narray): Concatenated variables or brightness value with a dimension of (H, W, C)
        gammaRange (tuple): Range of gamma values
        shiftSubset (tuple): Number of bands or channels for each shift
        patchShift (bool): Whether apply the shift on small patches

     Returns:

        narray, brightness shifted image

    '''


    c_start = 0

    if patchShift:
        for i in shiftSubset:
            gamma = random.triangular(gammaRange[0], gammaRange[1], 1)

            h, w, _ = img.shape
            rotMtrx = cv2.getRotationMatrix2D(center=(random.randint(0, h), random.randint(0, w)),
                                              angle=random.randint(0, 90),
                                              scale=random.uniform(1, 2))
            mask = cv2.warpAffine(img[:, :, c_start:c_start + i], rotMtrx, (w, h))
            mask = np.where(mask, 0, 1)
            # apply mask
            img_ma = ma.masked_array(img[:, :, c_start:c_start + i], mask=mask)
            img[:, :, c_start:c_start + i] = ma.power(img_ma, gamma)
            # default extra step -- shift on image
            gamma_full = random.triangular(0.5, 1.5, 1)
            img[:, :, c_start:c_start + i] = np.power(img[:, :, c_start:c_start + i], gamma_full)

            c_start += i
    else:
        # convert image dimension to (C, H, W) if len(img.shape)==3
        img = np.transpose(img, list(range(img.ndim)[-1:]) + list(range(img.ndim)[:-1]))
        for i in shiftSubset:
            gamma = random.triangular(gammaRange[0], gammaRange[1], 1)
            img[c_start:c_start + i, ] = np.power(img[c_start:c_start + i, ], gamma)

            c_start += i
        img = np.transpose(img, list(range(img.ndim)[-img.ndim + 1:]) + [0])

    return img


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    jittered = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return jittered


def shift_tstack_brightness(img, gamma_range=(0.2, 2.0), sample_number=5, patch_shift=False, mode = 'random'):
    '''
    Shift image brightness on sampled timestamps

    Params:

        img (narray): Concatenated time series cube in a dimension of (N, T), where N could be either nothing or  C, H, W
        gamma_range (tuple): Range of gamma values
        sample_number (int): Nubmer of timestamps to sample
        patch_shift (bool): Whether apply the shift on small patches; will be ignored if input is point stack
        mode {int): 'random' or 'uni'
     Returns:

        narray, brightness shifted time series cube
    '''

    # parameters
    assert img.ndim == 4 or img.ndim == 1
    if img.ndim == 4:
        c, h, w, t = img.shape
        if h < 2 or w < 2:
            patch_shift = False
    else:
        t = img.shape[-1]
    # reset sample number 
    sample_number = min(sample_number, t)

    # sample timestamos
    if mode == 'random':
        # convert t dimension to the first dimension
        img = np.transpose(img, list(range(img.ndim)[-1:]) + list(range(img.ndim)[:-1])) # T, C, H, W
        tsamples = random.sample(range(t), sample_number)
        for tsample in tsamples:
            img_t = img[tsample, ]
            img_t = img_t.transpose(list(range(img_t.ndim)[1:]) + [0]) # H, W, C
            shifted = shiftBrightness(img_t, gamma_range, [c],  patch_shift) # H, W C
            img[tsample,] = shifted.transpose(list(range(img_t.ndim)[-1:]) + list(range(img_t.ndim)[:-1])) # C, H, W
        # transpose back to C, H, W, T
        img = np.transpose(img, list(range(img.ndim)[-img.ndim + 1:]) + [0]) # C, H, W, T
    else:
        # convert c dimension to the last
        img = np.transpose(img, list(range(img.ndim)[1:]) + [0]) # H, W, T, C
        shifted = shiftBrightness(img, gamma_range, [c], patchShift = False)
        # transpose back to  C, H, W, T
        img = np.transpose(shifted, list(range(shifted.ndim)[-1:]) + list(range(shifted.ndim)[:-1]))

    return img


def jitter_tstack(img, sample_number=5):
    '''
    Apply jitter augmentation on sampled timestamps

    Params:
        img(narray): Concatenated time series cube in a dimension of (N, T), where N could be either nothing or  C, H, W
        sample_number (int): Nubmer of timestamps to sample

    Returns:

        narray, jittered time series cube
    '''
    assert img.ndim == 4 or img.ndim == 1
   
    # convert t dimension to the first dimension
    t = img.shape[-1]
    # reset sample number 
    sample_number = min(sample_number, t)
    img = np.transpose(img, list(range(img.ndim)[-1:]) + list(range(img.ndim)[:-1]))

    # jitter
    tsamples = random.sample(range(t), sample_number)
    for tsample in tsamples:
        img[tsample, ] = jitter(img[tsample, ])

    img = np.transpose(img, list(range(img.ndim)[-img.ndim + 1:]) + [0])

    return img
