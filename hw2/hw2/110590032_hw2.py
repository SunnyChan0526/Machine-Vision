import cv2
import numpy as np

class unionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        # if self.parent[x] != x:
        #     return self.find(self.parent[x])
        # else:
        #     return x  
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        self.parent[self.find(y)] = self.find(x)   
    def get_parent(self):
        return self.parent
    
class componentLabeling:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.imgLabel = np.zeros((height, width, 3), dtype=np.uint8)
        self.label = np.zeros((height, width), dtype=int)
        self.uf_parent = unionFind(height*width)
        self.label_p = 0
        self.colors = []
    def fourConnected(self, x, y):
        a = self.label[y][x-1] if x-1>=0 else 0
        b = self.label[y-1][x] if y-1>=0 else 0
        if(a == 0 and b == 0):
            self.label_p += 1
            self.label[y][x] = self.label_p
        elif(not(a != 0 and b != 0)):
            self.label[y][x] = a if a!=0 else b
        elif(a != 0 and b != 0):
            if(a == b):
                self.label[y][x] = a
            else:
                self.label[y][x] = a if a<b else b
                self.uf_parent.union(a, b)
    def eightConnected(self, x, y):
        a = self.label[y][x-1] if x-1>=0 else 0
        b = self.label[y-1][x-1] if y-1>=0 and x-1>=0 else 0
        c = self.label[y-1][x] if y-1>=0 else 0
        d = self.label[y+1][x+1] if y+1<self.height and x+1<self.width else 0
        if(not(a or b or c or d)): # all not labeled
            self.label_p += 1
            self.label[y][x] = self.label_p
        else:
            labelList = [i for i in [a, b, c, d] if i != 0]
            minLabel = min(labelList)
            self.label[y][x] = minLabel
            for i in labelList:
                if(i != minLabel):
                    self.uf_parent.union(i, minLabel)
    def genUniqColors(self):
        while len(self.colors) < self.label_p+1:
            color = np.random.randint(0, 256, size=(3,))
            unique = True
            for existing_color in self.colors:
                if np.array_equal(color, existing_color):
                    unique = False
                    break
            if unique:
                self.colors.append(color)
    def imgFillColor(self):
        for y in range(self.height):
            for x in range(self.width):
                if(self.label[y][x] != 0):
                    self.imgLabel[y, x] = self.colors[self.uf_parent.find(self.label[y][x])]
                else:
                    self.imgLabel[y, x] = 255

def convertColorToGrayscale(img, threshold, x, y):
    # Convert the color image to the grayscale image
    blue = img[y, x, 0]
    green = img[y, x, 1]
    red = img[y, x, 2] 
    imgConvert = (0.3 * red)+(0.59 * green)+(0.11 * blue)
    # Convert the grayscale image to the binary image
    return 255 if imgConvert >= threshold else 0

image_path = ['/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/images/img1.png',
              '/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/images/img2.png',
              '/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/images/img3.png',
              '/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/images/img4.png']
threshold = [110, 200, 230, 230]

for no, img_path in enumerate(image_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    img_4 = componentLabeling(height, width)
    img_8 = componentLabeling(height, width)
    
    for y in range(height):
        for x in range(width):
            img_4.imgLabel[y, x] = convertColorToGrayscale(img, threshold[no], x, y) 
            img_8.imgLabel[y, x] = convertColorToGrayscale(img, threshold[no], x, y) 

            # component labeling
            if(img_4.imgLabel[y, x, 0] == 0):
                img_4.fourConnected(x, y)
                img_8.eightConnected(x, y)

    img_4.genUniqColors()
    img_4.imgFillColor()
    img_8.genUniqColors()
    img_8.imgFillColor()
      
    name = "img" + str(no+1)
    # # cv2.imshow(name, img_4.imgLabel)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    cv2.imwrite('/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/results/' + name + '_4.jpg', img_4.imgLabel)
    cv2.imwrite('/Users/chantsaiching/Desktop/Machine Vision/HW2/hw2/results/' + name + '_8.jpg', img_8.imgLabel)