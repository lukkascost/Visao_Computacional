import cv2
import numpy as np

#img = cv2.imread('img/tri.png')
img = cv2.imread('../DataSet-Baumann/IMAGENS_RECORTADAS_30/c1_p1_00003.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 450)


print lines
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imwrite('houghlines3.jpg', img)
