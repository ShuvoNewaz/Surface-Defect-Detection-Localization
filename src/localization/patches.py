import cv2
import numpy as np
from skimage.segmentation import chan_vese


def activeContours(image):
    contours = chan_vese(image, mu=0.5, lambda1=1.0, lambda2=1.0, tol=0.001, max_num_iter=500, dt=0.5, init_level_set='checkerboard', extended_output=False)

    return contours
