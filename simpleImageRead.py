import cv2
import numpy
import glob
import os
import time
from PIL import Image

def readOpenCV():
    for i, item in enumerate(glob.glob("./data/*/*.jpg")):
        image = cv2.imread(item, cv2.IMREAD_COLOR)
def readPIL():
    for i, item in enumerate(glob.glob("./data/*/*.jpg")):
        image = Image.open(item)
        image = image.load()
start = time.time()
readOpenCV()
print("Time elapsed on decoding through OpenCV: ", time.time()-start)
start = time.time()
readPIL()
print("Time elapsed on decoding through Pillow: ", time.time()-start)
