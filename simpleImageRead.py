import cv2
import numpy
import glob
import os
import sys
import time
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--path1", type=str, help="image dir to read, source 1", required=True)
parser.add_argument("--path2", type=str, help="image dir to read, source 2", required=False, default="./")
args = parser.parse_args()

def readOpenCV(pathDir):
    for i, item in enumerate(glob.glob(pathDir + "*.jpg")):
        image = cv2.imread(item, cv2.IMREAD_COLOR)
def readPIL(pathDir):
    for i, item in enumerate(glob.glob(pathDir + "*.jpg")):
        image = Image.open(item)
        image = image.load()
def verifyImages(pathDir_1, pathDir_2):
    #  Using pathDir_2 to capture corresponding pathDir_1 files
    if pathDir_2 == "./":
        print("[ WARNING ], It seems that you didn't explicitly specify --path2. \
                If your path2 images are in current directory, then that's fine; \
                just be careful to give both two paths for function verifyImages().\n")
    for i, item in enumerate(glob.glob(pathDir_2 + "*.bmp")):
        filename_1 = os.path.basename(item)
        filepath_1 = pathDir_1 + filename_1.split(".", 1)[0] + ".jpg"
        image_1 = cv2.imread(filepath_1, cv2.IMREAD_COLOR)
        image_2 = cv2.imread(item, cv2.IMREAD_COLOR)
        height1, width1, depth1 = image_1.shape
        height2, width2, depth2 = image_2.shape
        if height1 != height2 or width1 != width2 or depth1 != depth2:
            print("Verification failed, please check.")
            sys.exit(1)
        for i in range(height1):
            for j in range(width1):
                for k in range(depth1):
                    if image_1[i][j][k] != image_2[i][j][k]:
                        print("Verification failed, please check.")
                        sys.exit(2)
        print("Well done, all correct.")

pathDir_1 = args.path1
pathDir_2 = args.path2
verifyImages(pathDir_1, pathDir_2)
#  start = time.time()
#  readOpenCV(pathDir_1)
#  print("Time elapsed on decoding through OpenCV: ", time.time()-start)
#  start = time.time()
#  readPIL(pathDir_1)
#  print("Time elapsed on decoding through Pillow: ", time.time()-start)
