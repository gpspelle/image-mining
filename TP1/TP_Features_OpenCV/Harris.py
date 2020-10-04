import numpy as np
import cv2
from matplotlib import pyplot as plt

def harris(input_img, k, window_size, threshold):

    harris_function = input_img.copy()
    harris_function[:,] = 0 # make it black
    
    thresh_harris_function = input_img.copy()
    thresh_harris_function[:,] = 0 # make it black

    offset = int(window_size/2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    
    dy, dx = np.gradient(input_img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            #The variable names are representative to 
            #the variable of the Harris corner equation
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2)
            harris_function[y,x] = r

            if r > threshold:
                thresh_harris_function[y,x] = 1
    
    return harris_function, thresh_harris_function 

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
alpha = 0.04

# Detecting corners
Theta = cv2.cornerHarris(img, blockSize*2, apertureSize, alpha)

# Computing local maxima and thresholding
Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc, d_maxloc), np.uint8)
Theta_dil = cv2.dilate(Theta, se)

#Suppression of non-local-maxima
Theta_maxloc[Theta < Theta_dil] = 0.0

#Values too small are also removed
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()

print("My computation of Harris points:", time, "s")
print("Number of cycles per pixel:", (t2 - t1)/(h * w), "cpp")

se_croix = np.uint8([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0],
                     [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])

# 1 0 0 0 1
# 0 1 0 1 0
# 0 0 1 0 0  se_croix represents a x that will be drawn with red
# 0 1 0 1 0 
# 1 0 0 0 1

Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)

#Re-read image for colour display
img_rgb = cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_COLOR)

(h, w, c) = img_rgb.shape
print("Dimension of image:", h, "rows x", w, "columns x", c, "channels")
print("Type of image:", img_rgb.dtype)

#Points are displayed as red crosses
img_rgb[Theta_ml_dil > 0] = [255,0,0]

plt.subplot(131)
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.title('Original image', fontsize=8)

plt.subplot(132)
plt.imshow(Theta, cmap = 'gray')
plt.axis('off')
plt.title('Harris function', fontsize=8)

plt.subplot(133)
plt.axis('off')
plt.imshow(img_rgb)
plt.title('Harris points', fontsize=8)

plt.savefig("harris-opencv.png", dpi=480, bbox_inches='tight', pad_inches=0)
plt.close()

threshold = 3000000.00
alpha = 0.04
apertureSize = 3
# Detecting corners
harris_function, thresh_harris_function = harris(img, alpha, apertureSize, threshold)

# Computing local maxima and thresholding
Theta_maxloc = cv2.copyMakeBorder(harris_function, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc, d_maxloc), np.uint8)
Theta_dil = cv2.dilate(Theta, se)

#Suppression of non-local-maxima
Theta_maxloc[Theta < Theta_dil] = 0.0

#Values too small are also removed
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0

Theta_ml_dil = cv2.dilate(Theta_maxloc, se_croix)

#Re-read image for colour display
img_rgb = cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_COLOR)

#Points are displayed as red crosses
img_rgb[Theta_ml_dil > 0] = [255,0,0]

plt.subplot(141)
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.title('Original image', fontsize=8)

plt.subplot(142)
plt.imshow(harris_function, cmap = 'gray')
plt.axis('off')
plt.title('Harris function (H)', fontsize=8)

plt.subplot(143)
plt.imshow(thresh_harris_function, cmap = 'gray')
plt.axis('off')
plt.title('H with threshold', fontsize=8)

plt.subplot(144)
plt.axis('off')
plt.imshow(img_rgb)
plt.title('Harris Points', fontsize=8)

plt.savefig("harris-mine.png", dpi=480, bbox_inches='tight', pad_inches=0)
plt.close()
