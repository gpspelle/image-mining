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
direct_method = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE) #Adding borders
for y in range(1,h-1):
  for x in range(1,w-1):
    #perform a convolution operation with a 3×3 filter
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    direct_method[y,x] = min(max(val,0),255)


t2 = cv2.getTickCount()
#Find the time of execution in seconds
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(direct_method, cmap='gray')
plt.title('Sharpen Convolution - Direct method')
plt.axis('off')
plt.savefig("conv_direct_sharpen.png", bbox_inches='tight')
plt.close()

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
filter2d_method = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(filter2d_method, cmap='gray')
plt.title('Sharpen Convolution - filter2D')
plt.axis('off')
plt.savefig("conv_filter2D_sharpen.png", bbox_inches='tight')
plt.close()

#normalized_filter2d = cv2.normalize(filter2d_method, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#normalized_filter2d.astype(np.uint8)

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
scipy_method = signal.convolve2d(img, kernel, boundary='symm', mode='same')
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method scipy.signal.convolve2D :",time,"s")

#normalized_scipy = cv2.normalize(scipy_method, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#normalized_scipy.astype(np.uint8)

plt.figure(figsize=(8, 6))
plt.imshow(scipy_method, cmap='gray')
plt.title('Sharpen Convolution - scipy')
plt.axis('off')
plt.savefig("conv_scipy_sharpen.png", bbox_inches='tight')
plt.close()

img_diff = filter2d_method - direct_method
img_diff *= 255.0 / np.max(img_diff) 
plt.figure(figsize=(8, 6))
plt.imshow(img_diff, cmap='gray')
plt.title("Sharpen result difference between the direct and filter2D")
plt.axis('off')
plt.savefig("difference_sharpen_direct-filter2D.png", bbox_inches='tight')
plt.close()

img_diff = scipy_method - direct_method
img_diff *= 255.0 / np.max(img_diff) 
plt.figure(figsize=(8, 6))
plt.imshow(img_diff, cmap='gray')
plt.title("Sharpen result difference between the direct and scipy")
plt.axis('off')
plt.savefig("difference_sharpen_direct-scipy.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(filter2d_method, cmap='gray', vmax=255.0, vmin=0.0)
plt.title('Normalized Sharpen Convolution - filter2d')
plt.axis('off')
plt.savefig("normalized_conv_filter2d_sharpen.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(scipy_method, cmap='gray', vmax=255.0, vmin=0.0)
plt.title('Normalized Sharpen Convolution - scipy')
plt.axis('off')
plt.savefig("normalized_conv_scipy_sharpen.png", bbox_inches='tight')
plt.close()



