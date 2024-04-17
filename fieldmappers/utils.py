import numpy as np
import pandas as pd
import os
import random
import itertools
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import logging
import pickle
from datetime import datetime

import torch

from .tools import InputError


def load_data(data_path, usage="train", window=None, norm_stats_type=None, 
              is_label=False):
    '''
    Read geographic data into numpy array
    Params:
        data_path : str
            Path of data to load
        usage : str
            Usage of the data: "train", "validate", or "predict"
        window : tuple
            The view onto a rectangular subset of the data, in the format of
            (column offsets, row offsets, width in pixel, height in pixel)
        norm_stats_type : str
            How the normalization statistics is calculated.
        is_label : binary
            Decide whether to saturate data with tested threshold
    Returns:
        narray
    '''

    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise InputError("Label shape not applicable: \
                                 expected 1 channel")
            img = src.read(1)
        else:
            nodata = src.nodata
            assert norm_stats_type in ["local_per_tile", "local_per_band", 
                                       "global_per_band"]
            if norm_stats_type == "local_per_tile":
                img = mmnorm1(src.read(), nodata=nodata)
            elif norm_stats_type == "local_per_band":
                img = mmnorm2(src.read(), nodata=nodata, clip_val=1.5)
            elif norm_stats_type == "global_per_band":
                img = mmnorm3(src.read(), nodata=nodata, clip_val=1.5)

            if usage in ['train', 'validate']:
               img = img[:, max(0, window[1]): window[1] + window[3], 
                         max(0, window[0]): window[0] + window[2]]

    return img


def get_stacked_img(img_paths, usage, norm_stats_type="local_per_tile", 
                    window=None):
    '''
    Read geographic data into numpy array
    Params:
        gsPath :str 
            Path of growing season image
        osPath : str
            Path of off season image
        img_paths : list
            List of paths for imgages
        usage : str
            Usage of the image: "train", "validate", or "predict"
        norm_stats_type : str
            How the normalization statistics is calculated.
        window : tuple
            The view onto a rectangular subset of the data, in the 
            format of (column offsets, row offsets, width in pixel, height in 
            pixel)
    Returns:
        ndarray
    '''

    if len(img_paths) > 1:
        img_ls = [load_data(m, usage, window, norm_stats_type) 
                  for m in img_paths]
        img = np.concatenate(img_ls, axis=0).transpose(1, 2, 0)
    else:
        img = load_data(img_paths[0], usage, window, norm_stats_type)\
            .transpose(1, 2, 0)

    if usage in ["train", "validate"]:
        col_off, row_off, col_target, row_target = window
        row, col, c = img.shape

        if row < row_target or col < col_target:

            row_off = abs(row_off) if row_off < 0 else 0
            col_off = abs(col_off) if col_off < 0 else 0

            canvas = np.zeros((row_target, col_target, c))
            canvas[row_off: row_off + row, col_off : col_off + col, :] = img

            return canvas

        else:
            return img

    elif usage == "predict":
        return img

    else:
        raise ValueError


def get_buffered_window(src_path, dst_path, buffer):
    '''
    Get bounding box representing subset of source image that overlaps with 
    bufferred destination image, in format of (column offsets, row offsets,
    width, height)

    Params:
        src_path : str 
            Path of source image to get subset bounding box
        dst_path : str 
            Path of destination image as a reference to define the 
            bounding box. Size of the bounding box is
            (destination width + buffer * 2, destination height + buffer * 2)
        buffer :int 
            Buffer distance of bounding box edges to destination image 
            measured by pixel numbers

    Returns:
        tuple in form of (column offsets, row offsets, width, height)
    '''

    with rasterio.open(src_path, "r") as src:
        gt_src = src.transform

    with rasterio.open(dst_path, "r") as dst:
        gt_dst = dst.transform
        w_dst = dst.width
        h_dst = dst.height

    col_off = round((gt_dst[2] - gt_src[2]) / gt_src[0]) - buffer
    row_off = round((gt_dst[5] - gt_src[5]) / gt_src[4]) - buffer
    width = w_dst + buffer * 2
    height = h_dst + buffer * 2

    return col_off, row_off, width, height


def get_meta_from_bounds(file, buffer):
    '''
    Get metadata of unbuffered region in given file
    Params:
        file (str):  File name of a image chip
        buffer (int): Buffer distance measured by pixel numbers
    Returns:
        dictionary
    '''

    with rasterio.open(file, "r") as src:

        meta = src.meta
        dst_width = src.width - 2 * buffer
        dst_height = src.height - 2 * buffer

        window = Window(buffer, buffer, dst_width, dst_height)
        win_transform = src.window_transform(window)

    meta.update({
        'width': dst_width,
        'height': dst_height,
        'transform': win_transform,
        'count': 1,
        'nodata': -128,
        'dtype': 'int8'
    })

    return meta



def display_hist(img):
    '''
    Display data distribution of input image in a histogram
    Params:
        img (narray): Image in form of (H,W,C) to display data distribution
    '''

    img = mmnorm1(img)
    im = np.where(img == 0, np.nan, img)

    plt.hist(img.ravel(), 500, [np.nanmin(im), img.max()])
    plt.figure(figsize=(20, 20))
    plt.show()


def mmnorm1(img, nodata):
    '''
    Data normalization with min/max method
    Params:
        img (narray): The targeted image for normalization
    Returns:
        narrray
    '''

    img_tmp = np.where(img == nodata, np.nan, img)
    img_max = np.nanmax(img_tmp)
    img_min = np.nanmin(img_tmp)
    normalized = (img - img_min) / (img_max - img_min)
    normalized = np.clip(normalized, 0, 1)

    return normalized


def mmnorm2(img, nodata, clip_val=None):
    r"""
    Normalize the input image pixels to [0, 1] ranged based on the
    minimum and maximum statistics of each band per tile.
    Arguments:
            img : numpy array 
                Stacked image bands with a dimension of (C,H,W).
            nodata : str 
                Value reserved to represent NoData in the image chip.
            clip_val : int 
                Defines how much of the distribution tails to be cut off.
    Returns:
            img : numpy array 
                Normalized image stack of size (C,H,W).
    Note 1: If clip then min, max are calculated from the clipped image.
    """

    # filter out zero pixels in generating statistics.
    nan_corr_img = np.where(img == nodata, np.nan, img)
    nan_corr_img = np.where(img == 0, np.nan, img)

    if clip_val > 0:
        left_tail_clip = np.nanpercentile(nan_corr_img, clip_val)
        right_tail_clip = np.nanpercentile(nan_corr_img, 100 - clip_val)

        left_clipped_img = np.where(img < left_tail_clip, left_tail_clip, img)
        clipped_img = np.where(left_clipped_img > right_tail_clip, 
                               right_tail_clip, left_clipped_img)

        normalized_bands = []
        for i in range(img.shape[0]):
            band_min = np.nanmin(clipped_img[i, :, :])
            band_max = np.nanmax(clipped_img[i, :, :])
            normalized_band = (clipped_img[i, :, :] - band_min) /\
                (band_max - band_min)
            normalized_bands.append(np.expand_dims(normalized_band, 0))
        normal_img = np.concatenate(normalized_bands, 0)

    elif clip_val == 0 or clip_val is None:
        normalized_bands = []
        for i in range(img.shape[0]):
            band_min = np.nanmin(nan_corr_img[i, :, :])
            band_max = np.nanmax(nan_corr_img[i, :, :])
            normalized_band = (nan_corr_img[i, :, :] - band_min) /\
                (band_max - band_min)
            normalized_bands.append(np.expand_dims(normalized_band, 0))
        normal_img = np.concatenate(normalized_bands, 0)

    else:
        raise ValueError("clip must be a non-negative decimal.")

    normal_img = np.clip(normal_img, 0, 1)
    return normal_img


def mmnorm3(img, nodata, clip_val=None):
    hardcoded_stats = {
        "mins": np.array([331.0, 581.0, 560.0, 1696.0]),
        "maxs": np.array([1403.0, 1638.0, 2076.0, 3652.0])
    }

    num_bands = img.shape[0]
    mins = hardcoded_stats["mins"]
    maxs = hardcoded_stats["maxs"]

    if clip_val:
        normalized_bands = []
        for i in range(num_bands):
            nan_corr_img = np.where(img[i, :, :] == nodata, np.nan, 
                                    img[i, :, :])
            nan_corr_img = np.where(img[i, :, :] == 0, np.nan, img[i, :, :])
            left_tail_clip = np.nanpercentile(nan_corr_img, clip_val)
            right_tail_clip = np.nanpercentile(nan_corr_img, 100 - clip_val)
            left_clipped_band = np.where(img[i, :, :] < left_tail_clip, 
                                         left_tail_clip, img[i, :, :])
            clipped_band = np.where(left_clipped_band > right_tail_clip, 
                                    right_tail_clip, left_clipped_band)
            normalized_band = (clipped_band - mins[i]) / (maxs[i] - mins[i])
            normalized_bands.append(np.expand_dims(normalized_band, 0))
        img = np.concatenate(normalized_bands, 0)

    else:
        for i in range(num_bands):
            img[i, :, :] = (img[i, :, :] - mins[i]) / (maxs[i] - mins[i])

    img = np.clip(img, 0, 1)
    return img

def get_chips(img, dsize, buffer):
    '''
    Generate small chips from input images and the corresponding index of each 
    chip The index marks the location of corresponding upper-left pixel of a 
    chip.
    Params:
        img (narray): Image in format of (H,W,C) to be crop, in this case it is 
            the concatenated image of growing season and off season
        dsize (int): Cropped chip size
        buffer (int):Number of overlapping pixels when extracting images chips
    Returns:
        list of cropped chips and corresponding coordinates
    '''

    h, w, _ = img.shape
    x_ls = range(0,h - 2 * buffer, dsize - 2 * buffer)
    y_ls = range(0, w - 2 * buffer, dsize - 2 * buffer)

    index = list(itertools.product(x_ls, y_ls))

    img_ls = []
    for i in range(len(index)):
        x, y = index[i]
        img_ls.append(img[x:x + dsize, y:y + dsize, :])

    return img_ls, index



def display(img, label, mask):

    '''
    Display composites and their labels
    Params:
        img (torch.tensor): Image in format of (C,H,W)
        label (torch.tensor): Label in format of (H,W)
        mask (torch.tensor): Mask in format of (H,W)
    '''

    gsimg = (comp432_dis(img, "GS") * 255).permute(1, 2, 0).int()
    osimg = (comp432_dis(img, "OS") * 255).permute(1, 2, 0).int()


    _, figs = plt.subplots(1, 4, figsize=(20, 20))

    label = label.cpu()

    figs[0].imshow(gsimg)
    figs[1].imshow(osimg)
    figs[2].imshow(label)
    figs[3].imshow(mask)

    plt.show()


# color composite
def comp432_dis(img, season):
    '''
    Generate false color composites
    Params:
        img (torch.tensor): Image in format of (C,H,W)
        season (str): Season of the composite to generate, be  "GS" or "OS"
    '''

    viewsize = img.shape[1:]

    if season == "GS":

        b4 = mmnorm1(img[3, :, :].cpu().view(1, *viewsize),0)
        b3 = mmnorm1(img[2, :, :].cpu().view(1, *viewsize),0)
        b2 = mmnorm1(img[1, :, :].cpu().view(1, *viewsize),0)

    elif season == "OS":
        b4 = mmnorm1(img[7, :, :].cpu().view(1, *viewsize), 0)
        b3 = mmnorm1(img[6, :, :].cpu().view(1, *viewsize), 0)
        b2 = mmnorm1(img[5, :, :].cpu().view(1, *viewsize), 0)

    else:
        raise ValueError("Bad season value")

    img = torch.cat([b4, b3, b2], 0)

    return img

def make_reproducible(seed=42, cudnn=True):
    """Make all the randomization processes start from a shared seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def pickle_dataset(dataset, file_path):
    with open(file_path, "wb") as fp:
        pickle.dump(dataset, fp)


def load_pickle(file_path):
    return pd.read_pickle(file_path)


def progress_reporter(msg, verbose, logger=None):
    """Helps control print statements and log writes
    Parameters
    ----------
    msg : str
      Message to write out
    verbose : bool
      Prints or not to console
    logger : logging.logger
      logger (defaults to none)
      
    Returns:
    --------  
        Message to console and or log
    """
    
    if verbose:
        print(msg)

    if logger:
        logger.info(msg)


def setup_logger(log_dir, log_name, use_date=False):
    """Create logger
    """
    if use_date:
        dt = datetime.now().strftime("%d%m%Y_%H%M")
        log = "{}/{}_{}.log".format(log_dir, log_name, dt)
    else: 
        log = "{}/{}.log".format(log_dir, log_name)
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = (
        f"%(asctime)s::%(levelname)s::%(name)s::%(filename)s::"
        f"%(lineno)d::%(message)s"
    )
    logging.basicConfig(filename=log, filemode='w',
                        level=logging.INFO, format=log_format)
    
    return logging.getLogger()


