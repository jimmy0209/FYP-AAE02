import cv2
import numpy as np

image = cv2.imread('/home/jimmy/PycharmProjects/edgedetectionnewnew/jjj.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
pixels = cv2.countNonZero(thresh)
# pixels = len(np.column_stack(np.where(thresh > 0)))

image_area = image.shape[0] * image.shape[1]
area_ratio = (pixels / image_area) * 100

print('pixels', pixels)
print('area ratio', area_ratio)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)