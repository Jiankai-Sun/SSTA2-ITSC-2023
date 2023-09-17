import cv2
import numpy as np
import os, sys


def resize_img(dir=''):
    for img_name in sorted(os.listdir(dir)):
        if img_name[-4:] != '.png':
            continue
        img = cv2.imread(os.path.join(dir, img_name))
        resized_img = cv2.resize(img, (128, 128))
        cv2.imwrite(os.path.join(dir, img_name), resized_img)
        print('Save to {}'.format(os.path.join(dir, img_name)))

if __name__ == "__main__":
    resize_img(dir='/home/jsun/Downloads/')