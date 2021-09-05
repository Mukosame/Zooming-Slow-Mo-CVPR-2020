import cv2
import numpy as np
import os
from os.path import isfile, join
import re

# extract frames from a video


def extract_frames(pathIn, pathOut, cnt=0):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    print('Start to extract frames from {}.'.format(os.path.basename(pathIn)))
    while success:
        cv2.imwrite(join(pathOut, "{:06d}.png".format(cnt)), image)
        success, image = vidcap.read()
        cnt += 1
    print('Successfully extract {} frames from {}.'.format(
        cnt, os.path.basename(pathIn)))

# combine frames to a video


def combine_frames(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    for i in range(len(files)):
        filename = join(pathIn, files[i])
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == "__main__":
    pathIn = 'demo_vid/'
    pathOut = 'out.mp4'
    fps = 29.98
    combine_frames(pathIn, pathOut, fps)
