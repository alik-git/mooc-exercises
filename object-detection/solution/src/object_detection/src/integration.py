#!/usr/bin/env python
# coding: utf-8

# In[10]:


def DT_TOKEN():
    # todo change this to your duckietown token
    dt_token = "dt1-3nT8KSoxVh4MguCtCLhmRq1o9tihmBNRqUf9x1T3bF5jze8-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbffAbB1yPboQBiqYm4RdGRFe7KXh3gqjEkM"
    return dt_token

def MODEL_NAME():
    # todo change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5"
    return "yolov5"


# In[11]:


def NUMBER_FRAMES_SKIPPED():
    # todo change this number to drop more frames
    # (must be a positive integer)
    return 1


# In[18]:


# `class` is the class of a prediction
def filter_by_classes(clas):
    # Right now, this returns True for every object's class
    # Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return (clas == 0)


# In[16]:


# `scor` is the confidence score of a prediction
def filter_by_scores(scor):
    # Right now, this returns True for every object's confidence
    # Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    print("Here the score is: ", scor)
    return scor > 0.5


# In[17]:


# `bbox` is the bounding box of a prediction, in xyxy format
# So it is of the shape (leftmost x pixel, topmost y pixel, rightmost x pixel, bottommost y pixel)
def filter_by_bboxes(bbox):
    # Like in the other cases, return False if the bbox should not be considered.
    box = bbox
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height
    return area > 5000

