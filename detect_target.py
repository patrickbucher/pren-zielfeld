#!/usr/bin/python3

"""
PREN Target Field Detection
Copyright 2018 by Patrick Bucher, patrick.bucher@stud.hslu.ch

This program processes pictures of a target field consisting of concentric
black and white squares. The borders of detected squares are marked as follows:

    - innermost square: blue
    - outermost square: red
    - other squares: green

Further features are displayed:

    - center of the innermost square: small yellow circle
    - vertical middle: purple line
    - distance of the aforementioned: orange line

The distance (in centimeters) will also be displayed on the bottom left of the
image with a descriptive string.

The target field is considered to enter the image from the top, thus a positive
distance means the yellow circle is over the purple middle line, and a negative
distance means the yellow circle is under the purple line.

The actual size of the physical innermost circle must be measured manually and
stored in INNERMOST_SQUARE_HEIGHT_CM.

Usage (with file input):

    ./detect_target.py [image file]

Usage (with Raspi Cam input):

    ./detect_target.py

"""

import cv2
import math
import numpy as np
import os
import sys

# constants relevant for detection
INNERMOST_SQUARE_HEIGHT_CM = 6.0
GRAY_THRESHOLD = 120

MIN_SQUARE_AREA_RATIO = 0.005 # 0.5% of the image
MAX_SQUARE_AREA_RATIO = 0.75 # 75% of the image

MIN_SQUARE_XY_RATIO = 0.75
MAX_SQUARE_XY_RATIO = 1.25

MAX_PERIMETER_DELTA_RATIO = 0.1

# constants for output
RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
YELLOW  = (255, 255, 0)
ORANGE  = (255, 165, 0)
PURPLE  = (255, 0, 255)

CAPTION_BORDER_PX = 100

def write_text(img, msg, col):
    h, _, _ = img.shape
    x, y = CAPTION_BORDER_PX, h - CAPTION_BORDER_PX
    cv2.putText(img, msg, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 4, col, 10)

def determine_center(contours):
    cx, cy = -1, -1
    m = cv2.moments(contours)
    m00 = m['m00']
    if m00 > 0:
        cx = int(m['m10']/m00)
        cy = int(m['m01']/m00)
    return (cx, cy)

def draw_contours(image, contures, color, thickness):
    cv2.drawContours(image, [contures], 0, color, thickness)

def find_threshold(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.blur(image_gray, (5, 5))
    threshold = cv2.threshold(blurred, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return threshold

def is_square_shaped(contours):
    peri = cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, MAX_PERIMETER_DELTA_RATIO * peri, True)
    if len(approx) != 4:
        return False
    (x, y, w, h) = cv2.boundingRect(approx)
    (cx, cy) = determine_center(contours)
    width_height_ratio = w / float(h)
    sensible_width_height_ratio = (MAX_SQUARE_XY_RATIO > width_height_ratio> MIN_SQUARE_XY_RATIO) 

    # L-shaped areas are also polygons, but have a much smaller area
    square_area_px = cv2.contourArea(contours)
    max_area = w*MAX_SQUARE_XY_RATIO * h*MAX_SQUARE_XY_RATIO
    min_area = w*MIN_SQUARE_XY_RATIO * h*MIN_SQUARE_XY_RATIO
    sensible_area = (max_area > square_area_px > min_area)

    return sensible_width_height_ratio and sensible_area

def estimate_distance_to_center(image, contours):
    h, _, _ = image.shape
    perimeter = cv2.arcLength(contours, True)
    innermost_square_height_px = perimeter / 4.0
    cm_in_pixels = innermost_square_height_px / INNERMOST_SQUARE_HEIGHT_CM
    _, cy = determine_center(contours)
    dist_px = h/2 - cy
    dist_cm = dist_px / cm_in_pixels
    return int(dist_px), dist_cm

def has_square_area(image, contours):
    h, w, _ = image.shape
    square_area_px = cv2.contourArea(contours)
    image_area_px = w * h
    min = MIN_SQUARE_AREA_RATIO * image_area_px
    max = MAX_SQUARE_AREA_RATIO * image_area_px
    return max > square_area_px > min

def has_neighbours(hierarchy):
    # hierarchy[0]: next, hierarchy[1]: previous
    return hierarchy[0] != -1 or hierarchy[1] != -1

def classify(cont_hier_list):
    min_area, max_area = math.inf, -1
    min_cont, max_cont = None, None
    others = []
    for item in cont_hier_list:
        cont = item[0]
        hier = item[1]
        area = cv2.contourArea(cont)
        # hier[3] = parent: smallest must have no child but a parent
        if area < min_area and hier[2] == -1 and hier[3] != -1:
            min_area = area
            min_cont = cont
        # hier[2] = child: biggest must have a child (checking for no parent is
        # unreliable, because the image frame could be the outermost parent)
        if area > max_area and hier[2] != -1:
            max_area = area
            max_cont = cont

    for item in cont_hier_list:
        cont = item[0]
        is_min = np.array_equal(cont, min_cont)
        is_max = np.array_equal(cont, max_cont)
        if not is_min and not is_max:
            others.append(cont)

    return (min_cont, max_cont, others)

def calc_middle(image):
    h, _, _ = image.shape
    middle = int(h/2)
    return middle

def draw_middle_line(image):
    _, w, _ = image.shape
    middle = calc_middle(image)
    cv2.line(image, (0, middle), (w, middle), PURPLE, 5)

def draw_distance_line(image, square_center):
    middle = calc_middle(image)
    x = square_center[0]
    cv2.line(image, (x, square_center[1]), (x, middle), ORANGE, 5)

def store(image, dir, filename):
    outfile = os.path.join(dir, os.path.split(filename)[-1])
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outfile, image_bgr)

def process(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    threshold = find_threshold(image_rgb)[1]

    _, contours, hierarchy = cv2.findContours(threshold.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    draw_middle_line(image_rgb)
    square_candidates = []
    for item in zip(contours, hierarchy[0]):
        cont = item[0]
        hier = item[1]
        if has_neighbours(hier):
            continue
        if not has_square_area(image_rgb, cont):
            continue
        if not is_square_shaped(cont):
            continue
        square_candidates.append((cont, hier))

    smallest, biggest, others = classify(square_candidates)
    if biggest is not None:
        draw_contours(image_rgb, biggest, RED, 10)
    if len(others) > 0:
        for contours in others:
            draw_contours(image_rgb, contours, GREEN, 10)
    if smallest is not None:
        draw_contours(image_rgb, smallest, BLUE, 10)
        (cx, cy) = determine_center(smallest)
        dist_px, dist_cm = estimate_distance_to_center(image_rgb, smallest)
        print('distance: {:4d}px {:7.3f}cm'.format(dist_px, dist_cm))
        if cx > 0 and cy > 0:
            i = image_rgb
            cv2.circle(image_rgb, (cx, cy), 10, YELLOW, 5)
            draw_distance_line(image_rgb, (cx, cy))
            h, _, _ = image_rgb.shape
            if h*0.5 > dist_px >= h*0.3:
                write_text(i, 'far away: {:.3f}cm'.format(dist_cm), GREEN)
            elif h*0.3 > dist_px >= h*0.2:
                write_text(i, 'away: {:.3f}cm'.format(dist_cm), GREEN)
            elif h*0.2 > dist_px >= h*0.1:
                write_text(i, 'close: {:.3f}cm'.format(dist_cm), YELLOW)
            elif h*0.1 > dist_px >= h*0.05:
                write_text(i, 'closer: {:.3f}cm'.format(dist_cm), YELLOW)
            elif h*0.05 > dist_px >= h*0.01:
                write_text(i, 'very close: {:.3f}cm'.format(dist_cm), YELLOW)
            elif h*0.01 > dist_px >= 0:
                write_text(i, 'extremely close: {:.3f}cm'.format(dist_cm), PURPLE)
            elif dist_px < 0:
                write_text(i, 'passed: {:.3f}cm'.format(dist_cm), RED)
            elif dist_px > 0:
                write_text('way too far away')
            else:
                write_text('unknown distance')
    store(image_rgb, 'demo', filename)

def main():
    if len(sys.argv) == 2:
        process(sys.argv[1])
    else:
        import picamera
        x, y = 2400, 2400 
        cam = picamera.PiCamera()
        cam.resolution = (x, y)
        for f in cam.capture_continuous('img{counter:04d}.jpg'):
            process(f)
            os.remove(f)

if __name__ == '__main__':
    main()
