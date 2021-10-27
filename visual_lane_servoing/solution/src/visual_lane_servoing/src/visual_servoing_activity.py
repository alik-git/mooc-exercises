#!/usr/bin/env python
# coding: utf-8

# In[33]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    # print(shape)
    # print(*shape)
    steer_matrix_left_lane = np.ones(shape)
    steer_matrix_left_lane[:,:] = -0.001

    return steer_matrix_left_lane


# In[34]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    
    steer_matrix_right_lane = np.ones(shape)
    steer_matrix_right_lane[:,:] = 0.001

    return steer_matrix_right_lane


# In[74]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    
    
    h, w, _ = image.shape
    
    # OpenCV uses BGR by default, whereas matplotlib uses RGB, so we generate an RGB version for the sake of visualization
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask_ground = np.ones(image.shape, dtype=np.uint8) # TODO: CHANGE ME
    h_mask_ratio = 3
    print("hello")
    print(mask_ground.shape)
    mask_ground[:int(image.shape[0]/h_mask_ratio),:] = 0
    mask_ground[:int(image.shape[0]/h_mask_ratio),:] = np.zeros((int(image.shape[0]/h_mask_ratio), int(image.shape[1])), dtype=uint8)
    mask_ground[:int(image.shape/h_mask_ratio),:] = np.zeros((int(image.shape/h_mask_ratio), int(image.shape[1])), dtype=uint8)

    
    sigma = 2
    img_gaussian_filter = cv2.GaussianBlur(image,(0,0), sigma)
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)
    threshold = 50 # CHANGE ME
    mask_mag = (Gmag > threshold)
    
    
    white_lower_hsv = np.array([0, 0,150])         # CHANGE ME
    white_upper_hsv = np.array([179, 40, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([15, 80, 170])        # CHANGE ME
    yellow_upper_hsv = np.array([40, 255, 255])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    # mask_ground = np.array([mask_ground, mask_ground, mask_ground])
    # print(mask_ground.shape)
    print(mask_left.shape)
    print(mask_mag.shape)
    print(mask_sobelx_neg.shape)
    print(mask_sobely_neg.shape)
    
    a = np.array([[1, 2], [1, 2]])
    print(a.shape)
    # (2,  2)

    # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
    # array along, (you can achieve the same effect by indexing with None, see below)
    mask_yellow_3 = np.repeat(mask_yellow[:, :, np.newaxis], 3, axis=2)
    mask_white_3 = np.repeat(mask_white[:, :, np.newaxis], 3, axis=2)

    
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow_3
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white_3
    
    # mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow_3
    # mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white_3
     
    # mask_left_edge = mask_yellow
    # mask_right_edge = mask_white

    # mask_left_edge = np.random.rand(h, w)
    # mask_left_edge[:,:] = 1
    mask_left_edge = np.mean(mask_left_edge,axis=2)
    mask_right_edge = np.mean(mask_right_edge,axis=2)
    
    return (mask_left_edge, mask_right_edge)

