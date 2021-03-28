import cv2
from sklearn.cluster import KMeans
from collections import Counter

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


def is_green_threshold(img, pts, rgb):
    green_points_height = []
    for point in pts:
        if (int(point[0]) < img.shape[0] and int(point[1]) < img.shape[1]):
            (r, g, b) = img[int(point[0]), int(point[1])]
            if b <= g and r <= g:
                green_points_height.append(int(point[0]))
    return green_points_height


def is_green_clast(img, pts, b):
    green_points_height = []
    # Find main 6 colors of picture
    rgb = find_colors(img)
    # Green color found if max is in 1st coordinate(RGB)
    green_colors = []
    for i in rgb:
        i = i.tolist()
        if (i.index(max(i)) == 1):
            green_colors.append(np.round(i, 0))
    for point in pts:
        if (int(point[0]) < img.shape[0] and int(point[1]) < img.shape[1]):
            (r, g, b) = img[int(point[0]), int(point[1])]
            for green in green_colors:
                if np.abs(green[0] - r) <= 20 and np.abs(green[1] - g) <= 20 and np.abs(green[2] - b) <= 20:
                    green_points_height.append(int(point[0]))
    return green_points_height


def is_green_from_first(img, pts, b):
    green_points_height = []
    green_colors = []
    # Green color found if max is in 1st coordinate(RGB)
    for i in rgb:
        i = i.tolist()
        if (i.index(max(i)) == 1):
            green_colors.append(np.round(i, 0))

    for point in pts:
        if (int(point[0]) < img.shape[0] and int(point[1]) < img.shape[1]):
            (r, g, b) = img[int(point[0]), int(point[1])]
            for green in green_colors:
                if np.abs(green[0] - r) <= 20 and np.abs(green[1] - g) <= 20 and np.abs(green[2] - b) <= 20:
                    green_points_height.append(int(point[0]))
    return green_points_height