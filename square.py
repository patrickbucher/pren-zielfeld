#!/usr/bin/python3

import cv2
import numpy as np
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

def center(cont):
    cx, cy = -1, -1
    m = cv2.moments(cont)
    m00 = m['m00']
    if m00 > 0:
        cx = int(m['m10']/m00)
        cy = int(m['m01']/m00)
    return (cx, cy)

def process(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(image_gray, (5, 5))
    threshold = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

    _, contours, hierarchy = cv2.findContours(threshold.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    height, width = threshold.shape
    img_mid = int(height/2)
    cv2.line(image_rgb, (0, img_mid), (width, img_mid), (255,0,255), 5)
    max_area = width * height * 0.75
    hi = 0
    for cont in contours:
        hier = hierarchy[hi]
        hi += 1
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.01 * peri, True)
        area = cv2.contourArea(cont)
        if area > max_area or area < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(approx)
        (cx, cy) = center(cont)
        ar = w / float(h)
        color = (0, 255, 0) # default: green
        innermost = False
        if hier[2] < 0:
            color = (0, 0, 255) # innermost: blue
            innermost = True
        if hier[3] < 0:
            color = (255, 0, 0) # outermost: red
        if ar >= 0.85 and ar <= 1.15:
            cont = cont.astype('int')
            cv2.drawContours(image_rgb, [cont], -1, color, 10)
            if innermost:
                (cx, cy) = center(cont)
                dist = height/2 - cy
                print('distance={}px'.format(dist))
                if cx > 0 and cy > 0:
                    cv2.circle(image_rgb, (cx, cy), 10, (255, 255, 0), 5)
                    if height*0.5 > dist >= height*0.3:
                        print('far away')
                    elif height*0.3 > dist >= height*0.2:
                        print('away')
                    elif height*0.2 > dist >= height*0.1:
                        print('close')
                    elif height*0.1 > dist >= height*0.05:
                        print('closer')
                    elif height*0.05 > dist >= height*0.01:
                        print('very close')
                    elif height*0.01 > dist >= 0:
                        print('extremely close')
                    elif dist < 0:
                        print('over')
                    elif dist > 0:
                        print('not there yet')

    plt.imshow(image_rgb)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("usage: {} [file]".format(sys.argv[0]))
        sys.exit(1)
    process(sys.argv[1])

if __name__ == '__main__':
    main()
