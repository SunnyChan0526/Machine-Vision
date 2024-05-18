import cv2
import numpy as np

image_path = ['HW5/hw5/images/img1.jpg',
              'HW5/hw5/images/img2.jpg',
              'HW5/hw5/images/img3.jpg']

def gen_neighbor_position(min, max):
    neighbor_position = []
    for x in range(min, max+1):
        for y in range(min, max+1):
            neighbor_position.append((x, y))
    return neighbor_position

def mean_filter(MeanFilter, height, width, min, max):
    for y in range(height):
        for x in range(width):
            neighbor_value = []
            neighbor_position = gen_neighbor_position(min, max)
            for dx, dy in neighbor_position:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_value.append(int(img[ny, nx, 0]))

            MeanFilter[y, x] = sum(neighbor_value) / len(neighbor_value)

def median_filter(MedianFilter, height, width, min, max):
    for y in range(height):
        for x in range(width):
            neighbor_value = []
            neighbor_position = gen_neighbor_position(min, max)
            for dx, dy in neighbor_position:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_value.append(int(img[ny, nx, 0]))

            mask_length = (max - min + 1)**2
            median = int(mask_length / 2) + 1
            median_position = median - (mask_length - len(neighbor_value)) - 1
            neighbor_value.sort()
            MedianFilter[y, x] = neighbor_value[median_position] if median_position >= 0 else 0

for no, img_path in enumerate(image_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        MeanFilter_3X3 = img.copy()
        MedianFilter_3X3 = img.copy()
        GaussianFilter_3X3 = img.copy()
        mean_filter(MeanFilter_3X3, height, width, -1, 1)
        median_filter(MedianFilter_3X3, height, width, -1, 1)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_3.jpg', MeanFilter_3X3)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_3.jpg', MeanFilter_3X3)

        MeanFilter_7X7 = img.copy()
        MedianFilter_7X7 = img.copy()
        GaussianFilter_7X7 = img.copy()
        mean_filter(MeanFilter_7X7, height, width, -3, 3)
        median_filter(MedianFilter_7X7, height, width, -3, 3)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_7.jpg', MeanFilter_7X7)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_7.jpg', MedianFilter_7X7)
        