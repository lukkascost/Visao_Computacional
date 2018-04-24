from Queue import Queue

import cv2
import numpy as np

orig = cv2.imread("img/fill2.png", 0)
cv2.imwrite("originalImage.jpg", orig)

line = orig.shape[0]
col = orig.shape[1]

result = np.zeros(orig.shape)
result[int(line / 2), int(col / 2)] = 255
cv2.imwrite("semente.png",result)
parar = 0
continuar = 1
it = 0
valPix = orig[int(line / 2), int(col / 2)]
valPix = int(valPix * 0.8)


while continuar != parar:
    continuar = parar
    parar = 0
    for x in range(1,orig.shape[0]-1):
        for y in range(1,orig.shape[1]-1):
            if result[x, y] == 255:
                if orig[x - 1, y - 1] >= valPix:
                    result[x - 1, y - 1] = 255
                    parar += 1
                if orig[x - 1, y] >= valPix:
                    result[x - 1, y] = 255
                    parar += 1
                if orig[x - 1, y + 1] >= valPix:
                    result[x - 1, y + 1] = 255
                    parar += 1
                if orig[x , y - 1] >= valPix:
                    result[x , y - 1] = 255
                    parar += 1
                if orig[x, y + 1] >= valPix:
                    result[x , y + 1] = 255
                    parar += 1
                if orig[x + 1, y - 1] >= valPix:
                    result[x + 1, y - 1] = 255
                    parar += 1
                if orig[x + 1, y] >= valPix:
                    result[x + 1, y] = 255
                    parar += 1
                if orig[x + 1, y + 1] >= valPix:
                    result[x + 1, y + 1] = 255
                    parar += 1
    cv2.imwrite("it/result1{:04d}.png".format(it), result)
    it+=1
#img = cv2.floodFill(orig, None, seed, 0)
