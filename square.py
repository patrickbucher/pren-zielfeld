#!/usr/bin/python3

import cv2
import numpy as np
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

def process(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(image_gray, (5, 5))
    threshold = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

    _, contours, hierarchy = cv2.findContours(threshold.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    w, h = threshold.shape
    max_area = w * h * 0.75
    hi = 0
    for cont in contours:
        hier = hierarchy[hi]
        hi += 1
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.01 * peri, True)
        area = cv2.contourArea(cont)
        if area > max_area or area < 500:
            continue
        m = cv2.moments(cont)
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        color = (0, 255, 0) # default: green
        if hier[2] < 0:
            color = (0, 0, 255) # innermost: blue
        if hier[3] < 0:
            color = (255, 0, 0) # outermost: red
        if ar >= 0.85 and ar <= 1.15:
            cont = cont.astype('int')
            cv2.drawContours(image_rgb, [cont], -1, color, 10)

    plt.imshow(image_rgb)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("usage: {} [file]".format(sys.argv[0]))
        sys.exit(1)
    process(sys.argv[1])

if __name__ == '__main__':
    main()
