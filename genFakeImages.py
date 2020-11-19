import cv2
import numpy as np


def create_blank(width, height, rgb):
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb))
    image[:] = color
    return image


# Create new blank 300x300 red image
width = 2048
height = 2048
pathDir = "./data/2kResolution/"
blue = (0, 0, 255)
for i in range(1000):
    image = create_blank(width, height, blue)
    cv2.imwrite(pathDir+"image_"+str(i)+".jpg", image)
    print("Index: ", i)
