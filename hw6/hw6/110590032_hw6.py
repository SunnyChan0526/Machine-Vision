import cv2
import numpy as np
import os

def convert_to_greyscale(image):
    """將彩色圖像轉換為灰度圖像"""
    height, width, _ = image.shape
    grey_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):  # 遍歷圖像的每一行
        for j in range(width):  # 遍歷圖像的每一列
            b, g, r = image[i, j]  # 獲取像素的藍、綠、紅分量
            grey_level = 0.3 * r + 0.59 * g + 0.11 * b  # 計算灰度值
            grey_image[i, j] = grey_level  # 將灰度值賦給灰度圖像

    return grey_image

def create_gaussian_kernel(size, sigma):
    """創建高斯核"""
    k = size // 2  # 計算核的半徑
    kernel = np.zeros((size, size), dtype=np.float32)
    
    sum_val = 0
    for x in range(size):  # 遍歷核的每一行
        for y in range(size):  # 遍歷核的每一列
            x_distance = (x - k) ** 2  # 計算x方向的距離平方
            y_distance = (y - k) ** 2  # 計算y方向的距離平方
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_distance + y_distance) / (2 * sigma**2))  # 計算高斯核值
            sum_val += kernel[x, y]  # 累加核值

    kernel /= sum_val  # 正規化核值

    return kernel

def apply_gaussian_filter(image, kernel_size, sigma):
    """應用高斯濾波器對圖像進行平滑處理"""
    kernel = create_gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    filtered_image = np.copy(image)
    
    for i in range(pad_size, image.shape[0] - pad_size):  # 遍歷圖像的每一行
        for j in range(pad_size, image.shape[1] - pad_size):  # 遍歷圖像的每一列
            region = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]  # 獲取當前像素的鄰域
            filtered_pixel = 0
            for m in range(kernel_size):  # 遍歷核的每一行
                for n in range(kernel_size):  # 遍歷核的每一列
                    filtered_pixel += region[m, n] * kernel[m, n]  # 對應位置相乘後累加
            filtered_image[i, j] = filtered_pixel  # 將結果賦值給濾波後的圖像

    return filtered_image

def compute_sobel_gradients(image):
    """計算Sobel梯度"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X方向核
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel Y方向核
    height, width = image.shape
    Gx = np.zeros((height, width))
    Gy = np.zeros((height, width))

    for i in range(1, height - 1):  # 遍歷圖像的每一行，邊界除外
        for j in range(1, width - 1):  # 遍歷圖像的每一列，邊界除外
            region = image[i - 1:i + 2, j - 1:j + 2]  # 獲取當前像素的3x3鄰域
            Gx[i, j] = np.sum(Kx * region)  # 計算X方向梯度
            Gy[i, j] = np.sum(Ky * region)  # 計算Y方向梯度

    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)  # 計算梯度幅值
    gradient_direction = np.arctan2(Gy, Gx)  # 計算梯度方向
    
    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    """非最大值抑制"""
    height, width = gradient_magnitude.shape
    suppressed_image = np.zeros((height, width))
    angle = np.rad2deg(gradient_direction)
    angle[angle < 0] += 180

    for i in range(1, height - 1):  # 遍歷圖像的每一行，邊界除外
        for j in range(1, width - 1):  # 遍歷圖像的每一列，邊界除外
            a = angle[i, j]
            
            if (0 <= a < 22.5) or (157.5 <= a <= 180):  # 水平方向
                before = gradient_magnitude[i, j - 1]
                after = gradient_magnitude[i, j + 1]
            elif (22.5 <= a < 67.5):  # 45度方向
                before = gradient_magnitude[i - 1, j - 1]
                after = gradient_magnitude[i + 1, j + 1]
            elif (67.5 <= a < 112.5):  # 垂直方向
                before = gradient_magnitude[i - 1, j]
                after = gradient_magnitude[i + 1, j]
            elif (112.5 <= a < 157.5):  # 135度方向
                before = gradient_magnitude[i - 1, j + 1]
                after = gradient_magnitude[i + 1, j - 1]
            else:
                print(f"角度超出範圍: {a}")

            if (gradient_magnitude[i, j] >= before) and (gradient_magnitude[i, j] >= after):  # 若當前像素是局部最大值
                suppressed_image[i, j] = gradient_magnitude[i, j]
            else:
                suppressed_image[i, j] = 0

    return suppressed_image

def double_thresholding(image, low_threshold, high_threshold):
    """雙閾值處理"""
    height, width = image.shape
    thresholded_image = np.zeros_like(image)

    for i in range(height):  # 遍歷圖像的每一行
        for j in range(width):  # 遍歷圖像的每一列
            if image[i, j] >= high_threshold:  # 強邊緣
                thresholded_image[i, j] = 255
            elif image[i, j] >= low_threshold:  # 弱邊緣
                thresholded_image[i, j] = 128

    return thresholded_image

def edge_tracking_hysteresis(image):
    """滯後閾值法邊緣追蹤"""
    height, width = image.shape
    visited_edges = np.zeros((height, width), dtype=bool)
    
    def depth_first_search(stack):
        """深度優先搜索"""
        while stack:
            y, x = stack.pop()
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited_edges[ny, nx]:
                        if image[ny, nx] == 128:  # 連接弱邊緣
                            visited_edges[ny, nx] = True
                            image[ny, nx] = 255
                            stack.append((ny, nx))
    
    for y in range(1, height - 1):  # 遍歷圖像的每一行，邊界除外
        for x in range(1, width - 1):  # 遍歷圖像的每一列，邊界除外
            if image[y, x] == 255 and not visited_edges[y, x]:  # 強邊緣像素且未訪問過
                visited_edges[y, x] = True
                depth_first_search([(y, x)])
    
    for y in range(1, height - 1):  # 遍歷圖像的每一行，邊界除外
        for x in range(1, width - 1):  # 遍歷圖像的每一列，邊界除外
            if image[y, x] == 128:  # 孤立的弱邊緣像素
                image[y, x] = 0
    
    return image

def canny_edge_detection(image_path, output_path):
    """Canny邊緣檢測"""
    image = cv2.imread(image_path)
    grey_image = convert_to_greyscale(image)  # 將圖像轉換為灰度圖像

    blurred_image = apply_gaussian_filter(grey_image, 5, 1.0)  # 高斯濾波
    gradient_magnitude, gradient_direction = compute_sobel_gradients(blurred_image)  # 計算梯度
    non_max_image = non_max_suppression(gradient_magnitude, gradient_direction)  # 非最大值抑制
    thresholded_image = double_thresholding(non_max_image, 50, 170)  # 雙閾值處理
    final_image = edge_tracking_hysteresis(thresholded_image)  # 滯後閾值法邊緣追蹤
    
    cv2.imwrite(output_path, final_image)  # 保存結果圖像
    return final_image

def main():
    """主函數"""
    input_folder = 'hw6/images'
    output_folder = 'hw6/results'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.jpg', '_sobel.jpg'))
            canny_edge_detection(input_path, output_path)

if __name__ == '__main__':
    main()