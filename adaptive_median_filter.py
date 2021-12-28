import cv2
import numpy as np
import pathlib
import random

def add_noise(img):
    noise = np.zeros(img.shape)          

    for y in range(img.shape[0]):              
        for x in range(img.shape[1]):
            randomnum = random.random()     
            if randomnum < 0.25:           
                noise[y,x] = 0
            elif randomnum > 0.25 and randomnum < 0.5:        
                noise[y,x] = 255
            else:                                   
                noise[y,x] = img[y,x]

    return noise

def adaptive_median_filter(img, s = 3, sMax = 7):
    out = np.zeros(img.shape)
    
    for y in range(4, img.shape[0] - sMax//2 + 1):
        for x in range(4, img.shape[1] - sMax//2 + 1):
            # print(y,x)
            out[y, x] = level_A(img, y, x, s, sMax)
    return out
            

def level_A(img, y, x, s, sMax):
    subimg = img[y - (s//2):y + (s//2) + 1, x - (s//2):x + (s//2) + 1]
    # print(y - (s//2), y + (s//2) + 1, x - (s//2), x + (s//2) + 1)
    # print(subimg.shape)
    # print("================")

    Zmin = np.min(subimg)
    Zmed = np.median(subimg)
    Zmax = np.max(subimg)
    
    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return level_B(subimg, Zmin, Zmed, Zmax)
    else:
        s += 2 
        if s <= sMax:
            # print("Increase the window size: " + str(s))
            return level_A(img,y,x,s,sMax)
        else:
            return Zmed


def level_B(img, Zmin, Zmed, Zmax):
    h,w = img.shape
    Zxy = img[h // 2, w //2]

    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0:
        return Zxy
    else:
        return Zmed


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.resolve()
    img = cv2.imread(str(path) + "/img/test_img_wang.tif", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(str(path) + "/lenna_512.jpg", cv2.IMREAD_GRAYSCALE)
    # print(img.shape)

    noise = add_noise(img)
    noise = noise.astype(np.uint8)
    cv2.imwrite(str(path) + '/out/add_noise.jpg',noise)

    out = adaptive_median_filter(noise)
    out = out.astype(np.uint8)
    cv2.imwrite(str(path) + '/out/result.jpg', out)