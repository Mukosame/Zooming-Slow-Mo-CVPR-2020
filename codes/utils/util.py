# this code is modified from https://github.com/xinntao/EDVR/blob/master/codes/utils/util.py
import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import glob
import re

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################
def get_model_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return (1.0*params/(1000*1000))


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info(
            'Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(
            root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print('img1-2')
    # print(img2)
    mse = np.mean((img1 - img2)**2)
    # print(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <=
                          max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            # clean the output (remove extra chars since last display)
            sys.stdout.write('\033[J')
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

####################
# read image
####################


def read_image(img_path):
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img_GT = cv2.imread(img_path)
    img = img_GT.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images'''
    img_path_l = glob.glob(img_seq_path + '/*')
    img_path_l.sort(key=lambda x: int(
        re.search(r'\d+', os.path.basename(x)).group()))
    return read_seq_imgs_by_list(img_path_l)


def read_seq_imgs_by_list(img_path_l):
    '''read a sequence of images from the given list'''
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1], torch
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(
        np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def test_index_generation(skip, N_out, len_in):
    '''
    params: 
    skip: if skip even number; 
    N_out: number of frames of the network; 
    len_in: length of input frames

    example:
  len_in | N_out  | times | (no skip)                  |   (skip)
    5    |   3    |  4/2  | [0,1], [1,2], [2,3], [3,4] | [0,2],[2,4]
    7    |   3    |  5/3  | [0,1],[1,2][2,3]...[5,6]   | [0,2],[2,4],[4,6] 
    5    |   5    |  2/1  | [0,1,2] [2,3,4]            | [0,2,4]
    '''
    # number of input frames for the network
    N_in = 1 + N_out // 2
    # input length should be enough to generate the output frames
    assert N_in <= len_in

    sele_list = []
    if skip:
        right = N_out  # init
        while (right <= len_in):
            h_list = [right-N_out+x for x in range(N_out)]
            l_list = h_list[::2]
            right += (N_out - 1)
            sele_list.append([l_list, h_list])
    else:
        right = N_out  # init
        right_in = N_in
        while (right_in <= len_in):
            h_list = [right-N_out+x for x in range(N_out)]
            l_list = [right_in-N_in+x for x in range(N_in)]
            right += (N_out - 1)
            right_in += (N_in - 1)
            sele_list.append([l_list, h_list])
    # check if it covers the last image, if not, we should cover it
    if (skip) and (right < len_in - 1):
        h_list = [len_in - N_out + x for x in range(N_out)]
        l_list = h_list[::2]
        sele_list.append([l_list, h_list])
    if (not skip) and (right_in < len_in - 1):
        right = len_in * 2 - 1
        h_list = [right-N_out+x for x in range(N_out)]
        l_list = [len_in - N_in + x for x in range(N_in)]
        sele_list.append([l_list, h_list])
    return sele_list


####################
# video
####################

def extract_frames(ffmpeg_dir, video, outDir):
    """
    Converts the `video` to images.
    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.
    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """

    error = ""
    print('{} -i {} -vsync 0 {}/%06d.png'.format(os.path.join(ffmpeg_dir,
          "ffmpeg"), video, outDir))
    retn = os.system('{} -i "{}" -vsync 0 {}/%06d.png'.format(
        os.path.join(ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error


def create_video(ffmpeg_dir, dir, output, fps):
    error = ""
    print('{} -r {} -f image2 -i {}/%6d.png {}'.format(os.path.join(ffmpeg_dir,
          "ffmpeg"), fps, dir, output))
    retn = os.system('{} -r {} -f image2 -i {}/%6d.png {}'.format(
        os.path.join(ffmpeg_dir, "ffmpeg"), fps, dir, output))
    if retn:
        error = "Error creating output video. Exiting."
    return error


# combine frames to a video
def combine_frames(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(
        pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    for i in range(len(files)):
        filename = os.path.join(pathIn, files[i])
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
