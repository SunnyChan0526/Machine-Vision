import cv2
import numpy as np
import math

def zero_pad(image, pad_size):
    padded_image = np.zeros((image.shape[0] + 2 * pad_size, image.shape[1] + 2 * pad_size, image.shape[2]), dtype=np.uint8)
    padded_image[pad_size:-pad_size, pad_size:-pad_size] = image
    return padded_image

def mean_filter(image, mask_size):
    pad_size = mask_size // 2
    padded_image = zero_pad(image, pad_size)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                sum_val = 0
                for m in range(mask_size):
                    for n in range(mask_size):
                        sum_val += padded_image[i + m, j + n, k]
                filtered_image[i, j, k] = sum_val // (mask_size * mask_size)
    
    return filtered_image

### 2. 中值濾波器 (Median Filter) 使用Zero Padding

def median_filter(image, mask_size):
    pad_size = mask_size // 2
    padded_image = zero_pad(image, pad_size)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                neighbors = []
                for m in range(mask_size):
                    for n in range(mask_size):
                        neighbors.append(padded_image[i + m, j + n, k])
                neighbors.sort()
                filtered_image[i, j, k] = neighbors[len(neighbors) // 2]

    return filtered_image

### 3. 高斯濾波器 (Gaussian Filter) 使用Zero Padding

def gaussian_filter(image, mask_size, sigma):
    def gaussian(x, y, sigma):
        return (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    pad_size = mask_size // 2
    padded_image = zero_pad(image, pad_size)
    filtered_image = np.zeros_like(image)

    # 生成高斯核
    gaussian_kernel = np.zeros((mask_size, mask_size), dtype=np.float32)
    sum_val = 0
    for i in range(mask_size):
        for j in range(mask_size):
            x = i - pad_size
            y = j - pad_size
            gaussian_kernel[i, j] = gaussian(x, y, sigma)
            sum_val += gaussian_kernel[i, j]

    # 正規化高斯核
    gaussian_kernel /= sum_val

    # 應用高斯濾波器
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                sum_val = 0
                for m in range(mask_size):
                    for n in range(mask_size):
                        sum_val += padded_image[i + m, j + n, k] * gaussian_kernel[m, n]
                filtered_image[i, j, k] = sum_val
    
    return filtered_image

def meanNmedian_filter(image, mask_size):
    return mean_filter(median_filter(image, 3), 3)

image_path = ['HW5/hw5/images/img1.jpg',
              'HW5/hw5/images/img2.jpg',
              'HW5/hw5/images/img3.jpg']

for no, img_path in enumerate(image_path):
    image = cv2.imread(img_path)
    
    MeanFilter_3X3 = mean_filter(image, 3)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_3.jpg', MeanFilter_3X3)
    
    MeanFilter_7X7 = mean_filter(image, 7)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_7.jpg', MeanFilter_7X7)

    MedianFilter_3X3 = median_filter(image, 3)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_3.jpg', MedianFilter_3X3)
    
    MedianFilter_7X7 = median_filter(image, 7)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_7.jpg', MedianFilter_7X7)
    
    GaussianFilter_5X5 = gaussian_filter(image, 5, 1.0)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_q3.jpg', GaussianFilter_5X5)

    MeanNMedianFilter_3X3 = meanNmedian_filter(image, 3)
    cv2.imwrite(f'HW5/hw5/results/img{no+1}_extra.jpg', MeanNMedianFilter_3X3)
