mread(img_path)
        height, width, channels = img.shape
        MeanFilter_3X3 = img.copy()
        MedianFilter_3X3 = img.copy()
        GaussianFilter_3X3 = img.copy()
        mean_filter(MeanFilter_3X3, height, width, -1, 1)
        median_filter(MedianFilter_3X3, height, width, -1, 1)
        gaussian_filter(GaussianFilter_3X3, height, width, -1, 1)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_3.jpg', MeanFilter_3X3)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_3.jpg', MeanFilter_3X3)

        MeanFilter_7X7 = img.copy()
        MedianFilter_7X7 = img.copy()
        GaussianFilter_7X7 = img.copy()
        mean_filter(MeanFilter_7X7, height, width, -3, 3)
        median_filter(MedianFilter_7X7, height, width, -3, 3)
        gaussian_filter(GaussianFilter_7X7, height, width, -3, 3)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q1_7.jpg', MeanFilter_7X7)
        cv2.imwrite(f'HW5/hw5/results/img{no+1}_q2_7.jpg', MedianFilter_7X7)