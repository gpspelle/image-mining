import numpy as np
import cv2

from matplotlib import pyplot as plt

def SI(img, x, y, p):
    val = np.sum(img[y-p:y+p, x-p:x+p])
    return min(max(val, 0), 255)

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
direct_method = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h):
  for x in range(1,w):
    val = img[y, x] - img[y-1, x] 
    direct_method[y,x] = min(max(val,0),255)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(direct_method, cmap='gray')
plt.title('Y derivate convolution - Direct method')
plt.axis('off')
plt.savefig("conv_direct_y_derivate.png", bbox_inches='tight')
plt.close()

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([-1, 1])
filter2d_method = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.figure(figsize=(8, 6))
plt.imshow(direct_method, cmap='gray')
plt.title('Y derivate convolution - filter 2D')
plt.axis('off')
plt.savefig("conv_filter2D_y_derivate.png", bbox_inches='tight')
plt.close()

img_diff = filter2d_method - direct_method
plt.figure(figsize=(8, 6))
plt.imshow(img_diff, cmap='gray', vmax=255, vmin=0)
plt.title("Y derivate result difference between the direct and filter2D")
plt.axis('off')
plt.savefig("difference_y_derivate_direct-filter2D.png", bbox_inches='tight')
plt.close()

center_y = h // 2
center_x = x // 2
p = 1
q = 50

SI_image = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

for i in range(-q//2, q//2 + 1, 1):
    for j in range(-q//2, q//2 + 1, 1):
        SI_image[center_y + i, center_x + j] = SI(img, center_y + i, center_x + j, p)

plt.figure(figsize=(8, 6))
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig("original_image.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(SI_image, cmap='gray')
plt.title('SI Function with p=1 on a square of size 50 on the center')
plt.axis('off')
plt.savefig("SI_function.png", bbox_inches='tight')
plt.close()
