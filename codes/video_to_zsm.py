#!/usr/bin/env python3
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import re

import utils.util as util
import data.util as data_util
import models.modules.Sakuya_arch as Sakuya_arch

import argparse
from shutil import rmtree

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="",
                    help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, required=True,
                    help='path of video to be converted')
parser.add_argument("--model", type=str, required=True,
                    help='path of pretrained model')
parser.add_argument("--fps", type=float, default=24,
                    help='specify fps of output video. Default: 24.')
parser.add_argument("--N_out", type=int, default=3,
                    help='Specify size of output frames of the network for faster conversion. This will depend on your cpu/gpu memory. Default: 7')
parser.add_argument("--output", type=str, default="output.mp4",
                    help='Specify output file name. Default: output.mp4')
args = parser.parse_args()


def check():
    """
    Checks the validity of commandline arguments.
    Parameters
    ----------
        None
    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """

    error = ""
    if (args.batch_size not in [3, 5, 7]):
        error = "Error: --N_out has to be 3 or 5 or 7"
    # if ".mkv" not in args.output:
        # error = "output needs to have a video container"
    return error


def main():
    scale = 4
    N_ot = args.N_out
    N_in = 1 + N_ot//2

    # model
    model_path = args.model
    model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)

    # extract the input video to temporary folder
    save_folder = osp.join(osp.dirname(args.output), '.delme')
    save_out_folder = osp.join(osp.dirname(args.output), '.hr_delme')
    if os.path.isdir(save_folder):
        rmtree(save_folder)
    util.mkdirs(save_folder)
    if os.path.isdir(save_out_folder):
        rmtree(save_out_folder)
    util.mkdirs(save_out_folder)
    error = util.extract_frames(args.ffmpeg_dir, args.video, save_folder)
    if error:
        print(error)
        exit(1)

    # temporal padding mode
    padding = 'replicate'
    save_imgs = True

    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def single_forward(model, imgs_in):
        with torch.no_grad():
            # print(imgs_in.size()) # [1,5,3,270,480]
            b, n, c, h, w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
            imgs_temp[:, :, :, 0:h, 0:w] = imgs_in
            model_output = model(imgs_temp)
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output

    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)
    # zsm images
    img_path_l = glob.glob(save_folder + '/*')
    img_path_l.sort(key=lambda x: int(
        re.search(r'\d+', os.path.basename(x)).group()))
    select_idx_list = util.test_index_generation(False, N_ot, len(img_path_l))
    for select_idxs in select_idx_list:
        # get input images
        select_idx = select_idxs[0]
        imgs_in = util.read_seq_imgs_by_list(
            [img_path_l[x] for x in select_idx]).unsqueeze(0).to(device)
        output = single_forward(model, imgs_in)
        outputs = output.data.float().cpu().squeeze(0)
        # save imgs
        out_idx = select_idxs[1]
        for idx, name_idx in enumerate(out_idx):
            output_f = outputs[idx, ...].squeeze(0)
            if save_imgs:
                output = util.tensor2img(output_f)
                cv2.imwrite(osp.join(save_out_folder,
                            '{:06d}.png'.format(name_idx)), output)

    # now turn output images to video
    # generate mp4
    util.combine_frames(save_out_folder,
                        args.output, args.fps)

    # remove tmp folder
    rmtree(save_folder)
    rmtree(save_out_folder)

    exit(0)


if __name__ == '__main__':
    main()
