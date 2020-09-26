# import library
import numpy as np
import cv2
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

plt.subplot(311)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Direct method')

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

#plotting the figures
plt.subplot(312)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

plt.subplot(313)
plt.imshow(img3 - img2, cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title("Difference between the two methods")

plt.tight_layout(pad=1.0)
plt.savefig("q11.png")
