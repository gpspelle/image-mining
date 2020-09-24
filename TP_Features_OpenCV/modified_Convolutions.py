import numpy as np
import cv2

from matplotlib import pyplot as plt

def SI(img, x, y, p):
    return np.sum(img[y-p:y+p, x-p:x+p])

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h):
  for x in range(1,w):
    val = img[y, x] - img[y-1, x] 
    img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.subplot(311)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Direct method')

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([-1, 1])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.subplot(312)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

plt.subplot(313)
plt.imshow(img3 - img2, cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title("Difference between the two methods")

plt.tight_layout(pad=1.0)
plt.savefig("q12.png")

plt.close()


center_y = h // 2
center_x = x // 2
p = 1
q = 50

img4 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

for i in range(-q//2, q//2 + 1, 1):
    for j in range(-q//2, q//2 + 1, 1):
        val = SI(img, center_y + i, center_x + j, p)
        print(val)
        img4[center_y + i, center_x + j] = val

img4 *= 255.0/img4.max()

plt.subplot(211)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title("Original image")

plt.subplot(212)
plt.imshow(img4, cmap="gray", vmin=0.0, vmax=255.0)
plt.title("SI function applied with p = 1 on the image center, a 50x50 square")

plt.tight_layout(pad=1.0)
plt.savefig("q13.png")

plt.close()
