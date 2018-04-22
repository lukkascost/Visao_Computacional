import cv2
import numpy as np

orig = cv2.imread("img/text.bmp", 0)
img = np.copy(orig)

line = img.shape[0]
col = img.shape[1]

a = img[0, :10]
b = np.flip(img[0, :10], 0)

for i in range(1, line, 2):
    img[i] = np.flip(img[i], 0)
array = np.reshape(img, img.size)

result = np.copy(array)
k = 4
for i in range(k, array.size):
    result[i] = int(np.average(array[i - k:i]))
result = np.reshape(result, [line, col])

for i in range(1, line, 2):
    result[i] = np.flip(result[i], 0)

print result[0, :10]

result2 = orig.astype(int) - (result.astype(int) * 0.5)
result2[result2 > 0] = 255
result2[result2 <= 0] = 0

print orig[0, :10], result2[0, :10], array[:10]

cv2.imwrite("result.jpg", result2)
