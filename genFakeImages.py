import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--row", type=int, default=2048)
parser.add_argument("--col", type=int, default=2048)
parser.add_argument("--iteration", type=int, help="How many images to generate", default=1000)
parser.add_argument("--path", type=str, help="image dir to store", required=True)
args = parser.parse_args()

def create_blank(width, height, rgb):
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb))
    image[:] = color
    return image


# Create new blank 300x300 red image
width = args.row
height = args.col
pathDir = args.path
iteration = args.iteration
blue = (0, 0, 255)
for i in range(iteration):
    image = create_blank(width, height, blue)
    cv2.imwrite(pathDir+"image_"+str(i)+".jpg", image)
    print("Index: ", i)
