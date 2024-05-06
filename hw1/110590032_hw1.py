import cv2
import numpy as np

# # q1_1
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     # print(height, width, channels)
#     # print(type(img[0, 0]))
#     for y in range(height):
#         for x in range(width):
#             blue = img[y, x, 0]
#             green = img[y, x, 1]
#             red = img[y, x, 2]
            
#             img[y, x] = (0.3 * red)+(0.59 * green)+(0.11 * blue)
    
#     name = "img" + str(no+1)
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q1-1.jpg', img)


# # q1_2
# image_path = ['results/img1_q1-1.jpg',
#               'results/img2_q1-1.jpg',
#               'results/img3_q1-1.jpg']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     for y in range(height):
#         for x in range(width):
#             threshold = 128
#             if(img[y, x, 0] >= threshold):
#                 img[y, x] = 255
#             else:
#                 img[y, x] = 0
#     name = "img" + str(no+1)
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q1-2.jpg', img)


# # q1_3
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']
# custom_colors = [
#     [255, 0, 0],     # Blue
#     [0, 255, 0],     # Green
#     [0, 0, 255],     # Red
#     [255, 255, 0],   # Yellow
#     [255, 0, 255],   # Magenta
#     [0, 255, 255],   # Cyan
#     [128, 0, 0],     # Maroon
#     [0, 128, 0],     # Olive
#     [0, 0, 128],     # Navy
#     [128, 128, 0],   # Yellow Green
#     [128, 0, 128],   # Purple
#     [0, 128, 128],   # Teal
#     [128, 128, 128], # Gray
#     [192, 192, 192], # Silver
#     [255, 165, 0],   # Orange
#     [128, 0, 0]      # Brown
# ]

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     index_img = np.zeros((height, width, 3), dtype=np.uint8)
#     for y in range(height):
#         for x in range(width):
#             color = img[y, x]
#             min_distance = float('inf')
#             index = 0
#             for i, custom_color in enumerate(custom_colors):
#                 distance = sum((c1 - c2) ** 2 for c1, c2 in zip(color, custom_color)) ** 0.5
#                 if distance < min_distance:
#                     min_distance = distance
#                     index = i
#             index_img[y][x] = np.array(custom_colors[index])
#     name = "img" + str(no+1)
#     cv2.imshow(name, index_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q1-3.jpg', index_img)


# # q2_1-double
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     ResizingImage_double = np.zeros((height*2, width*2, 3), dtype=np.uint8)
#     for y in range(height):
#         for x in range(width):
#             color = img[y, x]
#             for i in range(2*y, 2*y+2):
#                 for j in range(2*x, 2*x+2):
#                     ResizingImage_double[i][j] = color
#     name = "img" + str(no+1)
#     cv2.imshow(name, ResizingImage_double)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q2-1-double.jpg', ResizingImage_double)
  
    
# # q2_1-half
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     ResizingImage_double = np.zeros((height//2, width//2, 3), dtype=np.uint8)
#     for y in range(0, height, 2):
#         for x in range(0, width, 2):
#             color = img[y, x]
#             ResizingImage_double[y//2][x//2] = color
#     name = "img" + str(no+1)
#     cv2.imshow(name, ResizingImage_double)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q2-1-half.jpg', ResizingImage_double)
    
    
# # q2_2-double
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     ResizingImage_double = np.zeros((height*2, width*2, 3), dtype=np.uint8)
#     for y in range(height):
#         for x in range(width):
#             ResizingImage_double[y*2][x*2] = img[y, x]
#             if(x != width-1):
#                 ResizingImage_double[y*2][x*2+1] = (1/2)*img[y, x] + (1/2)*img[y, x+1]
#     for x in range(width):
#         for y in range(height):
#             ResizingImage_double[y*2][x*2] = img[y, x]
#             if(y != height-1):
#                 ResizingImage_double[y*2+1][x*2] = (1/2)*img[y, x] + (1/2)*img[y+1, x]
#     for x in range(1, width*2-1, 2):
#         for y in range(1, height*2-1, 2):
#             ResizingImage_double[y][x] = (1/2)*ResizingImage_double[y-1, x] + (1/2)*ResizingImage_double[y+1, x]
#     for y in range(height*2):
#         ResizingImage_double[y][-1] = ResizingImage_double[y][-2]
#     for x in range(width*2):
#         ResizingImage_double[-1][x] = ResizingImage_double[-2][x]
#     name = "img" + str(no+1)
#     cv2.imshow(name, ResizingImage_double)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q2-2-double.jpg', ResizingImage_double)
    
    
# # q2_2-half
# image_path = ['images/img1.png',
#               'images/img2.png',
#               'images/img3.png']

# for no, img_path in enumerate(image_path):
#     img = cv2.imread(img_path)
#     height, width, channels = img.shape
#     ResizingImage_double = np.zeros((height//2, width//2, 3), dtype=np.uint8)
#     for y in range(0, height, 2):
#         for x in range(0, width, 2):
#             color = img[y, x]
#             ResizingImage_double[y//2][x//2] = color
#     name = "img" + str(no+1)
#     cv2.imshow(name, ResizingImage_double)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite('results/' + name + '_q2-2-half.jpg', ResizingImage_double)