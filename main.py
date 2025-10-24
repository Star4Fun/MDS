import cv2
import os

IMAGES_PATH = "ImageCLEFmed2007_test"

images = []

def read_img(name):
    return cv2.imread(IMAGES_PATH + "/" + name + ".png")

for file in os.listdir(IMAGES_PATH):
    img_name = file.split(".")[0]
    images.append(img_name)

image = read_img(images[0])

cv2.imshow('image', image)

cv2.waitKey(0)