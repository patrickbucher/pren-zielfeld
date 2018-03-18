#!/usr/bin/python3

import cv2
import numpy as np
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
purple = (255, 0, 255)

innermost_square_height_cm = 5.0

def write_text(img, msg, col):
    cv2.putText(img, msg, (50,2400), cv2.FONT_HERSHEY_SIMPLEX, 4, col, 10)

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
                perimeter = cv2.arcLength(cont, True)
                innermost_square_height_px = perimeter / 4.0
                print('perimeter={}px', perimeter)
                cm_in_pixels = innermost_square_height_px / innermost_square_height_cm
                print('1cm={}px'.format(cm_in_pixels))
                (cx, cy) = center(cont)
                dist = height/2 - cy
                dist_cm = dist / cm_in_pixels
                print('distance={}px'.format(dist))
                if cx > 0 and cy > 0:
                    i = image_rgb
                    d = threshold.shape
                    cv2.circle(image_rgb, (cx, cy), 10, (255, 255, 0), 5)
                    if height*0.5 > dist >= height*0.3:
                        write_text(i, 'far away: {:.3f}cm'.format(dist_cm), green)
                    elif height*0.3 > dist >= height*0.2:
                        write_text(i, 'away: {:.3f}cm'.format(dist_cm), green)
                    elif height*0.2 > dist >= height*0.1:
                        write_text(i, 'close: {:.3f}cm'.format(dist_cm), yellow)
                    elif height*0.1 > dist >= height*0.05:
                        write_text(i, 'closer: {:.3f}cm'.format(dist_cm), yellow)
                    elif height*0.05 > dist >= height*0.01:
                        write_text(i, 'very close: {:.3f}cm'.format(dist_cm), yellow)
                    elif height*0.01 > dist >= 0:
                        write_text(i, 'extremely close: {:.3f}cm'.format(dist_cm), purple)
                    elif dist < 0:
                        write_text(i, 'over by: {:.3f}cm'.format(dist_cm), red)
                    elif dist > 0:
                        write_text('not there yet')

    plt.imshow(image_rgb)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("usage: {} [file]".format(sys.argv[0]))
        sys.exit(1)
    process(sys.argv[1])

if __name__ == '__main__':
    main()
