import matplotlib.pyplot as plt
import os
import numpy as np
from imageio import imread, imsave
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks, corner_fast, corner_subpix, ORB, canny
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
import green_thresholds as gt
from tabulate import tabulate

# Path to dirs here - change it!
path_to_dirs = 'C:\\Users\\Daria\\Documents\\Обработка сигналов\\Big_lab\\'

# Name of dirs - constant!
dirs = ['above',
        'below',
        'none']
name_dirs = ['Above', 'Below', 'None']


def k_closest(sample, pivot, k):
    return sorted(sample, key=lambda i: abs(i - pivot))[:k]


def show_hough_transform(image):
    # Hough from Canny borders
    h, theta, d = hough_line(canny(image))

    # Find horizontal lines (dist + angle)
    h_up = image.shape[0]
    dist_hor = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if np.abs(angle) > 1:
            dist_hor.append((np.abs(dist), angle))

            # Find heights from left corner
    heights = [np.abs(dist[0] / np.sin(dist[1])) for dist in dist_hor]

    # Find 2 closest among heights (here return closest for every height)
    result = [k_closest(heights, i, 2) for i in heights]
    # Find smallest distance
    distances = [(i, np.abs(i[0] - i[1])) for i in result]

    h_up, h_low = sorted(min(distances, key=lambda dist: dist[1])[0])

    return h_up, h_low


def sift_с(img):
    sift = cv2.SIFT_create()
    # Keypoints and descriptors.
    kp1, des1 = sift.detectAndCompute(img, None)
    return kp1


def get_green_point_height(kp, img, rgb, is_green):
    # Convert key points to coordinates
    pts = cv2.KeyPoint_convert(kp)
    # Round coordinates to get pixel
    pts = np.round(pts, 0)

    # Here find height of green key points
    green_points_height = is_green(img, pts, rgb)

    if len(green_points_height) == 0:
        return -1, -1
    max_h = np.min(green_points_height)
    min_h = np.max(green_points_height)

    return max_h, min_h


define_green = [gt.is_green_clast, gt.is_green_threshold, gt.is_green_from_first]
rows = []
headers = ['Method', 'Accuracy above', 'Accuracy below', 'Impossible accuracy', 'Total accuracy']

for method in define_green:
    acc = []
    acc.clear()
    i = 0
    for dir_ in dirs:
        dir_actual = os.path.join(path_to_dirs, dir_)
        os.chdir(dir_actual)
        images = [img_ for img_ in os.listdir(dir_actual)
                  if img_.endswith(".jpg") or
                  img_.endswith(".jpeg") or
                  img_.endswith("png")]
        sum_acc = 0
        for j in range(len(images)):
            image = imread(images[j])
            if j == 0:
                rgb = gt.find_colors(image)
            gray = rgb2gray(image)
            table_up, table_low = show_hough_transform(gray)
            key_points = sift_с(image)
            chair_up, chair_low = get_green_point_height(key_points, image, rgb, method)
            if chair_low < 0 or chair_up < 0:
                res = 'none'
            elif chair_low < table_up:
                res = 'above'
            elif chair_up > table_low:
                res = 'below'
            else:
                res = 'none'
            if res == name_dirs[i].lower():
                sum_acc += 1
        acc.append(sum_acc/len(images))
        i += 1
    rows.append((method.__name__, acc[0], acc[1], acc[2], np.round(np.sum(acc) / 3, decimals=3)))
print(tabulate(rows, headers, tablefmt="pipe"))