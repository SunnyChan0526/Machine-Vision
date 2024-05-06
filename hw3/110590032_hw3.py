import cv2
import numpy as np

image_path = ['hw3/images/img1.jpg',
              'hw3/images/img2.jpg',
              'hw3/images/img3.jpg',
              'hw3/images/img4.jpg']

binary_image_path = ['hw3/binary_images/img1.jpg',
              'hw3/binary_images/img2.jpg',
              'hw3/binary_images/img3.jpg',
              'hw3/binary_images/img4.jpg']

# image_path = ['hw3/images/img1.jpg']

def color2binary_image():
    for no, img_path in enumerate(image_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        
        # binary image
        for y in range(height):
            for x in range(width):
                blue = img[y, x, 0]
                green = img[y, x, 1]
                red = img[y, x, 2]          
                img[y, x] = (0.3 * red)+(0.59 * green)+(0.11 * blue)
                
                threshold = 170
                if(img[y, x, 0] >= threshold):
                    img[y, x] = 0
                else:
                    img[y, x] = 255
        name = "img" + str(no+1)
        cv2.imwrite('hw3/binary_images/' + name + '.jpg', img)
        # cv2.imshow(name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

         
def eight_distance_transform():
    for no, img_path in enumerate(binary_image_path):
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img[img == 255] = 1
        fm = img.copy()
        max_distance = 0
        while 1:
            prv = fm.copy()
            fm = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    min_8Neighbors = min(prv[y, x-1, 0] if x-1>-1 else 0, prv[y-1, x-1, 0] if y-1>-1 and x-1>-1 else 0
                                        , prv[y-1, x, 0] if y-1>-1 else 0, prv[y-1, x+1, 0] if y-1>-1 and x+1<width else 0
                                        , prv[y, x+1, 0] if x+1<width else 0, prv[y+1, x, 0] if y+1<height else 0
                                        , prv[y+1, x+1, 0] if y+1<height and x+1<width else 0, prv[y+1, x-1, 0] if y+1<height and x-1>-1 else 0)
                    fm[y, x] = img[y, x, 0] + min_8Neighbors
                    if(int(fm[y, x, 0]) > max_distance):
                        max_distance = int(fm[y, x, 0])
            if np.array_equal(prv, fm):
                break
        print(max_distance)
        DF_img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                DF_img[y, x] = fm[y, x, 0]*(256/max_distance) - 1 if fm[y, x, 0]*(256/max_distance) - 1 > 0 else 0
                    
        name = "img" + str(no+1)
        # cv2.imwrite('hw3/binary_image/' + name + '.jpg', img)
        cv2.imshow(name, DF_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
color2binary_image()
eight_distance_transform()