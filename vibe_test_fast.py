
import numpy as np
import os
import cv2
import time




def getRandomNeighborCoordinate(N):
    x, y = 0, 0
    if N == 1:
        x, y = -1, -1
    elif N == 2:
        x, y = 0, -1
    elif N == 3:
        x, y = 1, -1
    elif N == 4:
        x, y = -1, 0
    elif N == 5:
        x, y = 1, 0
    elif N == 6:
        x, y = -1, 1
    elif N == 7:
        x, y = 0, 1
    elif N == 8:
        x, y = 1, 1

    return x, y


def initial_background(I_gray, N):
    
    # I_pad = np.pad(I_gray, 1, 'symmetric')
    I_pad = I_gray
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height,width,N))



    # for i in range(1, height - 1):
    #     for j in range(1, width - 1):
    #         for n in range(N):
    #             x, y = 0, 0
    #             while(x == 0 and y == 0):
    #                 x = np.random.randint(-1, 1)
    #                 y = np.random.randint(-1, 1)
    #             ri = i + x
    #             rj = j + y
    #             samples[i, j, n] = I_pad[ri, rj]
    # samples = samples[1:height-1, 1:width-1]
    for i in range(N):
            samples[:, :, i] = I_gray
    samples_list = samples.tolist()
    return samples_list
    
def vibe_detection(I_gray, samples, _min, N, R, pi):
    
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    segMap_list = segMap.tolist()
    I_gray_list = I_gray.tolist()

    
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            pixel_value = I_gray_list[i][j]
            while count < _min and index < N:
                # dist = np.abs(pixel_value - samples[i,j,index])
                dist = np.abs(int(pixel_value) - int(samples[i][j][index]))
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                r = np.random.randint(0, pi)
                if r == 0:
                    r = np.random.randint(0, N)
                    samples[i][j][r] = pixel_value
                r = np.random.randint(0, pi)
                if r == 0:
                    v = np.random.randint(1, 9)
                    x, y = getRandomNeighborCoordinate(v)
                    r = np.random.randint(0, N)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri][rj][r] = pixel_value
                    except:
                        pass
            else:
                segMap_list[i][j] = 255

    segMap = np.array(segMap_list).astype(np.uint8)
    return segMap, samples
    
rootDir = r'data/input'
image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
image = cv2.imread(image_file, 0)

N = 20
R = 20
_min = 2
pi = 16

samples = initial_background(image, N)

frm = 0
for lists in os.listdir(rootDir):
    path = os.path.join(rootDir, lists)
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    startTime = time.time()

    segMap, samples = vibe_detection(gray, samples, _min, N, R, pi)


    endTime = time.time() - startTime
    cv2.imshow('segMap', segMap)
    print("{} is done".format(lists))
    print(endTime)
    frm = frm + 1
    if cv2.waitKey(1) and 0xff == ord('q'):
        break
cv2.destroyAllWindows()
