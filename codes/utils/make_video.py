import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    # TODO: can be changed depending on your data
    files.sort(key = lambda x: int(x[2:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each file
        img = cv2.imread(filename)
        print(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    # TODO: change your input & output path here
    pathIn= './input_path'
    pathOut = './output_path/2kcut_hfrsr_slow.mp4'
    fps = 23.98
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()