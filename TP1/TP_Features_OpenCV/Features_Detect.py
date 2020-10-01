import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage:",sys.argv[0],"detector(= orb or kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage:",sys.argv[0],"detector(= orb or kaze)")
  sys.exit(2)

#Read the image pair
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension of image 1:",img1.shape[0],"rows x",img1.shape[1],"columns")
print("Type of image 1:",img1.dtype)
img2 = cv2.imread('../Image_Pairs/torb_small2.png')
print("Dimension of image 2:",img2.shape[0],"rows x",img2.shape[1],"columns")
print("Type of image 2:",img2.dtype)

#Beggining the calculus...
t1 = cv2.getTickCount()
#Creation of objects "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 250,#By default : 500
                       scaleFactor = 2,#By default : 1.2
                       nlevels = 3)#By default : 8
  kp2 = cv2.ORB_create(nfeatures=250,
                       scaleFactor = 2,
                       nlevels = 3)
  print("Detector: ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#By default : false
    		        threshold = 0.001,#By default : 0.001
  		            nOctaves = 4,#By default : 4
		            nOctaveLayers = 4,#By default : 4
		            diffusivity = 2)#By default : 2
  kp2 = cv2.KAZE_create(upright = False,#By default : false
	  	        threshold = 0.001,#By default : 0.001
		        nOctaves = 4,#By default : 4
		        nOctaveLayers = 4,#By default : 4
		        diffusivity = 2)#By default : 2
  print("Detector: KAZE")
#Conversion to grayscale
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Keypoints detection
pts1 = kp1.detect(gray1,None)
pts2 = kp2.detect(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Detection of key points:",time,"s")

#Displaying the keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags defines the information level on key points
# 0: position only; 4: position + scale + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()
