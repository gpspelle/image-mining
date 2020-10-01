# import library
import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
#Code execution
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE) #Adding borders
for y in range(1,h-1):
  for x in range(1,w-1):
    #perform a convolution operation with a 3×3 filter
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    img2[y,x] = min(max(val,0),255)

t2 = cv2.getTickCount()
#Find the time of execution in seconds
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(img2, cmap='gray')
plt.title('Convolution - Direct method')
plt.axis('off')
plt.savefig("conv_direct_sharpen.png", bbox_inches='tight')
plt.close()

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(img3, cmap='gray')
plt.title('Convolution - filter2D')
plt.axis('off')
plt.savefig("conv_filter2D_sharpen.png", bbox_inches='tight')
plt.close()

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img4 = signal.convolve2d(img, kernel, boundary='symm', mode='same')
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method scipy.signal.convolve2D :",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(img3, cmap='gray')
plt.title('Convolution - scipy')
plt.axis('off')
plt.savefig("conv_scipy_sharpen.png", bbox_inches='tight')
plt.close()

img_diff = img3 - img2
img_diff *= 255.0 / np.max(img_diff) 
plt.figure(figsize=(8, 6))
plt.imshow(img_diff, cmap='gray')
plt.title("Result difference between the direct and filter2D")
plt.axis('off')
plt.savefig("difference_sharpen_direct-filter2D.png", bbox_inches='tight')
plt.close()

img_diff = img4 - img2
img_diff *= 255.0 / np.max(img_diff) 
plt.figure(figsize=(8, 6))
plt.imshow(img_diff, cmap='gray')
plt.title("Result difference between the direct and scipy")
plt.axis('off')
plt.savefig("difference_sharpen_direct-scipy.png", bbox_inches='tight')
plt.close()
