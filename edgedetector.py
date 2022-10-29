import cv2
import numpy as np

#gray scale
img = cv2.imread(r"/home/vk/Desktop/opencv/3.png",cv2.IMREAD_GRAYSCALE)

#gradient: directional change in intensity of image

# prewitt kernel

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

prewittx = cv2.filter2D(img,-1,kernelx)
prewitty = cv2.filter2D(img,-1,kernely)

#sobel is inbuild function opencv
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

#canny (Reduce the noise)
img_canny = cv2.Canny(img,100,200)


cv2.imwrite("input1.png",img)
cv2.imwrite("prewittx1.png",prewittx)
cv2.imwrite("prewitty1.png",prewitty)
cv2.imwrite("Sobelx1.png",sobelx)
cv2.imwrite("Sobely1.png",sobely)
cv2.imwrite("canny1.png",img_canny)


cv2.imshow("Input",img)
cv2.imshow("Prewittx",prewittx)
cv2.imshow("prewitty",prewitty)
cv2.imshow("prewitt combine",prewittx+prewitty)
cv2.imshow("sobelx",sobelx)
cv2.imshow("sobely",sobely)
cv2.imshow("combine sobel",sobelx+sobely)
cv2.imshow("canny",img_canny)
cv2.waitKey()
cv2. destroyAllWindows()

