import numpy as np
import cv2
from matplotlib import pyplot as plt

#Reading grayscale image and conversion to float64
#img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))

# Reading the image directly with openCV and not converting it with numpy later
img = cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_GRAYSCALE)
(h,w) = img.shape

print("Dimension of image:", h, "rows x", w, "columns")
print("Type of image:", img.dtype)

#Beginning of calculus
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0, 0, 0, 0, cv2.BORDER_REPLICATE)

# Put here Harris interest function calculation
# Detector parameters
blockSize = 2
apertureSize = 3
k = 0.04

# Detecting corners
Theta = cv2.cornerHarris(img, blockSize*2, apertureSize, k)

# Computing local maxima and thresholding
Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc, d_maxloc), np.uint8)
Theta_dil = cv2.dilate(Theta, se)

#Suppression of non-local-maxima
Theta_maxloc[Theta < Theta_dil] = 0.0

#Values to small are also removed
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()

print("My computation of Harris points:", time, "s")
print("Number of cycles per pixel:", (t2 - t1)/(h * w), "cpp")

plt.subplot(131)
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.title('Original image')

plt.subplot(132)
plt.imshow(Theta, cmap = 'gray')
plt.axis('off')
plt.title('Harris function')

se_croix = np.uint8([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0],
                     [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])

Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)

#Re-read image for colour display
img_rgb = cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_COLOR)

(h, w, c) = img_rgb.shape
print("Dimension of image:", h, "rows x", w, "columns x", c, "channels")
print("Type of image:", img_rgb.dtype)

#Points are displayed as red crosses
img_rgb[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.axis('off')
plt.imshow(img_rgb)
plt.title('Harris points')

plt.savefig("harris.png", dpi=480, bbox_inches='tight', pad_inches=0)
plt.show()
