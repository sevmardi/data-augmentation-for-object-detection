from core.data_aug import *
from core.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle as pkl
# https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/quick-start.ipynb
# Storage Format
# First we define how the storage formats required for images to work
# The image: An open-cv numpy array of shape (HxHxC)
# Annotations: A numpy array of shape N x 5 where N is the number of objects, one represented by each row. 
# 5 columns represent the top-left x-coordinate, top-left y-coordinate, bottom-right x-coordinate, bottom-right y-coordinate, and the class of the object. 

img = cv2.imread("messi.jpg")[:,:,::-1] # opencv loads images in bgr. 
bboxes = pkt.load(open("messi_aan.pkl", "rb"))
print(bboxes)
