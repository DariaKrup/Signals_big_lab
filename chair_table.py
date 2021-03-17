import matplotlib.pyplot as plt
import os
import numpy as np
from imageio import imread, imsave
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks, corner_fast, corner_subpix, ORB, canny
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import cv2
import opencv as oc
from sklearn.cluster import KMeans
from collections import Counter

# Directories here - change name(!)
dirs = ['C:\\Users\\Daria\\Documents\\Обработка сигналов\\Big_lab\\above',
        'C:\\Users\\Daria\\Documents\\Обработка сигналов\\Big_lab\\below',
        'C:\\Users\\Daria\\Documents\\Обработка сигналов\\Big_lab\\none']
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


number_of_colors = 6
def find_colors(image):
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    return rgb_colors


def get_green_point_height(kp, img):
    # Convert key points to coordinates
    pts = cv2.KeyPoint_convert(kp)
    # Round coordinates to get pixel
    pts = np.round(pts, 0)

    # Find main 8 colors of picture
    rgb = find_colors(img)

    # Green color found if max is in 1st coordinate(RGB)
    green_colors = []
    for i in rgb:
        i = i.tolist()
        if (i.index(max(i)) == 1):
            green_colors.append(np.round(i, 0))

    def is_green(b, g, r):
        for green in green_colors:
            if np.abs(green[0] - r) <= 20 and np.abs(green[1] - g) <= 20 and np.abs(green[2] - b) <= 20:
                return True

    # Here find height of green key points
    green_points_height = []
    for point in pts:
        if (int(point[0]) < img.shape[0] and int(point[1]) < img.shape[1]):
            (r, g, b) = img[int(point[0]), int(point[1])]
            if is_green(b, g, r):
                green_points_height.append(int(point[0]))

    if len(green_points_height) == 0:
        return -1, -1
    max_h = np.min(green_points_height)
    min_h = np.max(green_points_height)

    return max_h, min_h


i = 0
for dir_ in dirs:
    images = [img_ for img_ in os.listdir(dir_)
              if img_.endswith(".jpg") or
              img_.endswith(".jpeg") or
              img_.endswith("png")]
    print(name_dirs[i])
    sum_acc = 0
    for j in range(len(images)):
        os.chdir(dir_)
        #print(images[j])
        image = imread(images[j])
        #print(image)
        gray = rgb2gray(image)
        table_up, table_low = show_hough_transform(gray)
        #print(table_up, table_low)
        key_points = sift_с(image)
        chair_up, chair_low = get_green_point_height(key_points, image)
        #print(chair_up, chair_low)
        print('Image ' + str(j) + ': ', end ="")
        if chair_low < 0 or chair_up < 0:
            res = 'none'
        elif chair_low < table_up:
            res = 'above'
        elif chair_up > table_low:
            res = 'below'
        else:
            res = 'none'
        print(res)
        if res == name_dirs[i].lower():
            sum_acc += 1
    #print(sum_acc)
    print('For ' + name_dirs[i] + ' accuracy is: ' + str(sum_acc/len(images)))
    i += 1
