from core.data_aug import *
from core.bbox_util import draw_rect
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl

# Storage Format
# First we define how the storage formats required for images to work
# The image: An open-cv numpy array of shape (HxHxC)
# Annotations: A numpy array of shape N x 5 where N is the number of objects, one represented by each row. 
# 5 columns represent the top-left x-coordinate, top-left y-coordinate, bottom-right x-coordinate, bottom-right y-coordinate, and the class of the object. 

img = cv2.imread("messi.jpg")[:,:,::-1] # opencv loads images in bgr. 
bboxes = pkl.load(open("messi_ann.pkl", "rb"))
# print(bboxes)

plotte_img = draw_rect(img, bboxes)
plt.imshow(plotte_img)

plt.show()


#Horizontal Flipping.
img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#Scaling
img_, bboxes_ = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()


#Translation
img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

#Rotation
img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()


#Shearing
img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

# Resizing
img_, bboxes_ = Resize(608)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

# HSV transforms are supported as well.
img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()

# You can combine multiple transforms together by using the Sequence class as follows.
seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()])
img_, bboxes_ = seq(img.copy(), bboxes.copy())

plotted_img = draw_rect(img_, bboxes_)
plt.imshow(plotted_img)
plt.show()