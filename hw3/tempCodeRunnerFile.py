    # # 8-distance transform
    # img[img == 255] = 1
    # fm = img.copy()
    # max_distance = 0
    # while 1:
    #     prv = fm.copy()
    #     fm = np.zeros((height, width, 3), dtype=np.uint8)
    #     for y in range(height):
    #         for x in range(width):
    #             min_8Neighbors = min(prv[y, x-1, 0] if x-1>-1 else 0, prv[y-1, x-1, 0] if y-1>-1 and x-1>-1 else 0
    #                                  , prv[y-1, x, 0] if y-1>-1 else 0, prv[y-1, x+1, 0] if y-1>-1 and x+1<width else 0
    #                                  , prv[y, x+1, 0] if x+1<width else 0, prv[y+1, x, 0] if y+1<height else 0
    #                                  , prv[y+1, x+1, 0] if y+1<height and x+1<width else 0, prv[y+1, x-1, 0] if y+1<height and x-1>-1 else 0)
    #             fm[y, x] = img[y, x, 0] + min_8Neighbors
    #             if(int(fm[y, x, 0]) > max_distance):
    #                 max_distance = int(fm[y, x, 0])
    #     if np.array_equal(prv, fm):
    #         break
    # print(max_distance)
    # DF_img = np.zeros((height, width, 3), dtype=np.uint8)
    # for y in range(height):
    #     for x in range(width):
    #         DF_img[y, x] = fm[y, x, 0]*(256/max_distance) - 1 if fm[y, x, 0]*(256/max_distance) - 1 > 0 else 0
                
    # name = "img" + str(no+1)
    # cv2.imshow(name, DF_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()