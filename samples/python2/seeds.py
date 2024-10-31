#!/usr/bin/env python

'''
This sample demonstrates SEEDS Superpixels segmentation
Use [space] to toggle output mode

Usage:
  seeds.py [<video source>]

'''

import numpy as np
import cv2

# relative module
import video

# built-in module
import sys


if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('SEEDS')
    cv2.createTrackbar('# Superpixels', 'SEEDS', 100, 1000, nothing)
    cv2.createTrackbar('Use Prior', 'SEEDS', 0, 10, nothing) # 2
    cv2.createTrackbar('# Levels', 'SEEDS',0, 10, nothing) # 1
    cv2.createTrackbar('Iterations', 'SEEDS',0, 20, nothing) # 20

    seeds = None
    display_mode = 0 
    num_superpixels = 100 # default 400
    prior = 1 # default 2
    num_levels = 1 # default 4
    num_histogram_bins = 5 # default 5

    cap = video.create_capture(fn)
    while True:
        flag, img = cap.read()
        converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height,width,channels = converted_img.shape
        num_superpixels = cv2.getTrackbarPos('# Superpixels', 'SEEDS')
        num_levels = cv2.getTrackbarPos('# levels', 'SEEDS')
        prior = cv2.getTrackbarPos('Use Prior', 'SEEDS')
        num_iterations = cv2.getTrackbarPos('Iterations', 'SEEDS')

        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
                num_superpixels, num_levels, prior, num_histogram_bins)
        color_img = np.zeros((height,width,3), np.uint8)
        color_img[:] = (0, 0, 255)

        seeds.iterate(converted_img, num_iterations)

        # retrieve the segmentation result
        labels = seeds.getLabels()


        # labels output: use the last x bits to determine the color
        num_label_bits = 2
        labels &= (1<<num_label_bits)-1
        labels *= 1<<(16-num_label_bits)


        mask = seeds.getLabelContourMask(False)

        # stitch foreground & background together
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
        result = cv2.add(result_bg, result_fg)

        if display_mode == 0:
            cv2.imshow('SEEDS', result)
        elif display_mode == 1:
            cv2.imshow('SEEDS', mask)
        else:
            cv2.imshow('SEEDS', labels)

        ch = cv2.waitKey(1)
        if ch == 27:
            break
        elif ch & 0xff == ord(' '):
            display_mode = (display_mode + 1) % 3
    cv2.destroyAllWindows()
